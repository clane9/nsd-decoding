"""v2: Shallow MLP on masked (non-background) flat map voxels."""

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


class ShallowMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_split_tensors(ds, mask):
    """Load a HF dataset split, apply mask, per-sample z-normalize."""
    activity = np.array(ds["activity"])  # (N, 215, 200) or (N, 1, 215, 200)
    if activity.ndim == 4:
        activity = activity[:, 0]
    # Apply mask: (N, 215, 200) -> (N, num_voxels)
    activity = activity[:, mask]
    activity = torch.tensor(activity, dtype=torch.float32)
    # Per-sample z-normalization
    mean = activity.mean(dim=1, keepdim=True)
    std = activity.std(dim=1, keepdim=True).clamp(min=1e-6)
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

    # Load mask
    mask = np.load(ROOT / "metadata/nsd_flat_mask.npy")
    num_voxels = int(mask.sum())
    print(f"Mask: {num_voxels} voxels out of {mask.size}")

    # Load data
    dataset_dict = hfds.load_from_disk(ROOT / "datasets/nsd_flat_cococlip")

    print("Loading tensors...")
    splits = {}
    for split in dataset_dict:
        act, tgt = load_split_tensors(dataset_dict[split], mask)
        splits[split] = (act, tgt)
        print(f"  {split}: {act.shape}, targets: {tgt.shape}")

    train_loader = DataLoader(
        TensorDataset(*splits["train"]),
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    eval_loaders = {
        split: DataLoader(TensorDataset(act, tgt), batch_size=512, num_workers=4, pin_memory=True)
        for split, (act, tgt) in splits.items()
    }

    # Model
    model = ShallowMLP(num_voxels, args.hidden, NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
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
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--notes", type=str, default=None)
    args = parser.parse_args()
    main(args)
