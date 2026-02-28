"""v1: Simple CNN baseline for cross-subject fMRI decoding."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import datasets as hfds
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).parents[2]
SCRIPT = Path(__file__).stem
NUM_CLASSES = 24


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (1, 215, 200) -> (32, 107, 100)
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: (32, 107, 100) -> (64, 53, 50)
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: (64, 53, 50) -> (128, 26, 25)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 4: (128, 26, 25) -> (256, 13, 12)
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_split_tensors(ds):
    """Load a HF dataset split into tensors with per-sample z-normalization."""
    activity = torch.tensor(np.array(ds["activity"]), dtype=torch.float32)  # (N, 215, 200)
    if activity.ndim == 3:
        activity = activity.unsqueeze(1)  # (N, 1, 215, 200)
    # Per-sample z-normalization (critical for cross-subject generalization)
    mean = activity.mean(dim=(2, 3), keepdim=True)
    std = activity.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    activity = (activity - mean) / std
    targets = torch.tensor(np.array(ds["target"]), dtype=torch.long)
    return activity, targets


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # Random horizontal flip augmentation
        if torch.rand(1).item() > 0.5:
            x = x.flip(-1)
        # Gaussian noise augmentation
        x = x + 0.1 * torch.randn_like(x)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, 100 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        all_preds.append(out.argmax(1).cpu())
    return torch.cat(all_preds).numpy()


def main(args):
    start_t = time.monotonic()
    sha, is_clean = get_sha()
    print(f"sha: {sha}, clean: {is_clean}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Load data
    dataset_dict = hfds.load_dataset("clane9/nsd-flat-cococlip")

    print("Loading tensors...")
    splits = {}
    for split in dataset_dict:
        act, tgt = load_split_tensors(dataset_dict[split])
        splits[split] = (act, tgt)
        print(f"  {split}: {act.shape}, targets: {tgt.shape}")

    train_loader = DataLoader(
        TensorDataset(*splits["train"]),
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    eval_loaders = {
        split: DataLoader(TensorDataset(act, tgt), batch_size=256, num_workers=4, pin_memory=True)
        for split, (act, tgt) in splits.items()
    }

    # Model
    model = SimpleCNN(NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None
    for epoch in range(args.epochs):
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        # Evaluate on validation
        val_preds = evaluate(model, eval_loaders["validation"], device)
        val_acc = 100 * accuracy_score(splits["validation"][1].numpy(), val_preds)

        elapsed = time.monotonic() - start_t
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | loss={loss:.4f} | train_acc={train_acc:.1f}% | val_acc={val_acc:.1f}% | {elapsed:.0f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best model and evaluate all splits
    model.load_state_dict(best_state)
    model.to(device)
    preds = {}
    for split in dataset_dict:
        preds[split] = evaluate(model, eval_loaders[split], device)

    scores = score_predictions(splits, preds)

    result = {
        "script": SCRIPT,
        "args": vars(args),
        "sha": sha,
        "clean": is_clean,
        "wall_t": round(time.monotonic() - start_t, 3),
        **scores,
    }
    print(json.dumps(result))


def score_predictions(splits, preds):
    scores = {}
    for split, (act, tgt) in splits.items():
        targets = tgt.numpy()
        score = accuracy_score(targets, preds[split])
        scores[f"acc_{split}"] = round(100 * score, 3)
    return scores


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    clean = True
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        clean = not _run(["git", "diff-index", "HEAD"])
    except Exception:
        pass
    return sha, clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--notes", type=str, default=None)
    args = parser.parse_args()
    main(args)
