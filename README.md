# NSD Decoding

Cross-subject visual category decoding from fMRI data using the [Natural Scenes Dataset](https://naturalscenesdataset.org/).

## Task

Predict which of 24 visual categories a subject is viewing, given their fMRI cortical flat map activity. The challenge is cross-subject generalization: models are trained on 6 subjects and evaluated on 2 held-out subjects.

## Dataset

[`clane9/nsd-flat-cococlip`](https://huggingface.co/datasets/clane9/nsd-flat-cococlip) on HuggingFace. Each sample contains:

- `activity`: fMRI flat map (215 x 200)
- `target`: category ID (0-23)
- `subject_id`, `trial_id`, `nsd_id`

## Setup

```bash
uv sync
```

## Usage

```bash
# Run the current best model (v4: PCA + residual MLP)
python src/nsd_decoding/nsd_flat_cococlip_decoding_v4.py --n_components 96 --depth 6 --dropout 0.7 --drop_path 0.2 --lr 5e-4

# Run a hyperparameter sweep
sbatch experiments/sweep_v4_nc/launch.sh
```

## Results

See [RESULTS.md](RESULTS.md) for detailed results. Current best: 27.8% cross-subject test accuracy.
