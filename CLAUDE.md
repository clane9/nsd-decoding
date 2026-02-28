# NSD decoding

The goal of this small project is to train models for visual category decoding from fMRI data using the natural scenes dataset.

The dataset is hosted on HuggingFace at [`clane9/nsd-flat-cococlip`](https://huggingface.co/datasets/clane9/nsd-flat-cococlip). Each sample has the following fields:

- `subject_id`: NSD subject ID (0, ..., 7)
- `trial_id`: NSD trial ID
- `nsd_id`: NSD stimulus ID
- `activity`: visual cortex fMRI activity represented in a flat map format (shape `(1, 215, 200)`)
- `target`: target category ID (0, ..., 23)

A model should predict the target category given the activity map. Constraints: single H100 GPU, wall time at most 20 minutes per run, no additional data. Track results in `RESULTS.md`.

## Data splits

The validation and test splits contain held-out subjects. The task is cross-subject decoding.

- Train: subjects 0, 1, 2, 5, 6, 7 (32,539 samples)
- Validation: subject 3 (5,418 samples)
- Test: subject 4 (5,390 samples)
- Testid: subjects 0, 1, 2, 5, 6, 7 held-out trials (5,187 samples)

## Preprocessing pipeline

1. Mask background voxels (value 127) using `metadata/nsd_flat_mask.npy` (18,577 of 43,000 survive)
2. Per-sample z-normalization of masked voxels
3. PCA projection + whitening using `metadata/nsd_flat_pca.npz` (fit on training data only)
   - `components`: (512, 18577), `mean`: (18577,), `scale`: (512,) for whitening

## Key files

- Scripts: `src/nsd_decoding/nsd_flat_cococlip_decoding_v{0,1,2,3,4}.py`
- Preprocessing: `notebooks/nsd_flat_masking.ipynb`, `notebooks/nsd_flat_pca.ipynb`
- Metadata: `metadata/nsd_flat_mask.npy`, `metadata/nsd_flat_pca.npz`, `metadata/nsd_include_coco_categories.json`
- Experiments: `experiments/sweep_v4/`, `experiments/sweep_v4_nc/`

## Current status

Best cross-subject test accuracy: 27.8% (v4, n_components=96, depth=6, dropout=0.7, drop_path=0.2, lr=5e-4). PCA dimensionality is the dominant factor for cross-subject generalization. In-distribution and out-of-distribution performance are anticorrelated across n_components.

## Collaboration notes

- Connor prefers to run training scripts himself and watch the logs
- Be direct; minimal style (avoid excessive emojis and emphatic language)
- Each script version is a separate file (v0.py, v1.py, ...) for reproducibility
- Use sbatch array jobs for sweeps (see `experiments/` for the pattern)
