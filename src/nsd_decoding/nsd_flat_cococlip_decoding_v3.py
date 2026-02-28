"""v3: Residual MLP with low-dim projection on masked flat map voxels."""

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


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        mask = torch.rand(x.shape[0], 1, device=x.device) < keep_prob
        return x * mask / keep_prob


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.5, drop_path=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        return x + self.drop_path(self.block(x))


class ResidualMLP(nn.Module):
    def __init__(
        self, input_dim, latent_dim, depth, num_classes=NUM_CLASSES, dropout=0.5, drop_path=0.1
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, latent_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(latent_dim, dropout=dropout, drop_path=drop_path) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


def load_split_tensors(ds, mask):
    """Load a HF dataset split, apply mask, per-sample z-normalize."""
    activity = np.array(ds["activity"])
    if activity.ndim == 4:
        activity = activity[:, 0]
    activity = activity[:, mask]
    activity = torch.tensor(activity, dtype=torch.float32)
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
    dataset_dict = hfds.load_dataset("clane9/nsd-flat-cococlip")

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
    model = ResidualMLP(
        input_dim=num_voxels,
        latent_dim=args.latent_dim,
        depth=args.depth,
        dropout=args.dropout,
        drop_path=args.drop_path,
    ).to(device)
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
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--notes", type=str, default=None)
    args = parser.parse_args()
    main(args)
