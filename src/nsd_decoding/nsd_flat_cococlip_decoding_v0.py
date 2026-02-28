import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import datasets as hfds
import numpy as np
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).parents[2]

SCRIPT = Path(__file__).stem


def main(args):
    start_t = time.monotonic()
    sha, is_clean = get_sha()
    print(f"sha: {sha}, clean: {is_clean}")

    dataset_dict = hfds.load_dataset("clane9/nsd-flat-cococlip")
    dataset_dict.set_format("torch")

    preds = {split: np.full(len(ds), 0, dtype=np.int64) for split, ds in dataset_dict.items()}
    scores = score_predictions(dataset_dict, preds)

    result = {
        "script": SCRIPT,
        "args": vars(args),
        "sha": sha,
        "clean": is_clean,
        "wall_t": round(time.monotonic() - start_t, 3),
        **scores,
    }
    print(json.dumps(result))


def score_predictions(dataset_dict: hfds.DatasetDict, preds: dict[str, np.ndarray]):
    scores = {}
    for split, ds in dataset_dict.items():
        targets = np.asarray(ds["target"])
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
    parser.add_argument("--notes", type=str, default=None)
    args = parser.parse_args()
    main(args)
