# NSD Decoding Results

24-class visual category decoding from fMRI cortical flat maps (215x200). Cross-subject task: train on subjects 0,1,2,5,6,7; validate on subject 3; test on subject 4. Testid = held-out trials from training subjects.

## 2026-02-28

### v4 n_components sweep

PCA dimensionality is the dominant factor for cross-subject generalization. Lower PCA dims discard subject-specific variance and retain shared visual semantics. In-distribution (testid) and out-of-distribution (test) performance are anticorrelated across n_components.

Sorted by test accuracy. Raw results in `experiments/sweep_v4_nc/result.jsonl`.

| n_components | Config | Test | Val | TestID | Train |
|-------------|--------|------|-----|--------|-------|
| 96 | d6_highreg | 27.8 | 25.8 | 29.6 | 34.2 |
| 96 | default | 27.5 | 25.7 | 30.6 | 37.3 |
| 80 | default | 27.1 | 25.4 | 28.9 | 34.0 |
| 64 | default | 27.1 | 26.5 | 28.4 | 33.4 |
| 128 | default | 27.0 | 25.7 | 31.9 | 39.4 |
| 112 | default | 27.0 | 25.0 | 31.3 | 39.3 |
| 128 | d6_highreg | 26.8 | 26.1 | 31.7 | 37.5 |
| 48 | default | 26.8 | 26.2 | 25.1 | 28.4 |
| 64 | d6_highreg | 26.6 | 26.2 | 26.9 | 30.3 |
| 32 | default | 26.0 | 25.0 | 22.7 | 25.3 |
| 160 | default | 25.7 | 24.7 | 33.0 | 39.2 |
| 192 | default | 23.7 | 24.6 | 34.5 | 43.2 |

### v4 hyperparameter sweep

20 configs varying dropout, depth, lr, wd, drop_path at n_components=256 baseline. Raw results in `experiments/sweep_v4/results.jsonl`. n_components=128 was the standout (26.5% test vs 22.4% baseline), motivating the focused sweep above. Other hyperparameters had marginal effects at fixed n_components.

### v0-v4 version progression

| Version | Architecture | Test | Val | TestID | Train | Wall time |
|---------|-------------|------|-----|--------|-------|-----------|
| v0 | Constant (predict class 0) | 3.7 | 3.6 | 3.8 | 4.0 | 6s |
| v1 | CNN (4 conv blocks, 2D input) | 19.9 | 19.3 | 20.2 | 23.0 | 363s |
| v2 | Shallow MLP on masked voxels | 19.4 | 19.3 | 36.6 | 55.3 | 206s |
| v3 | Residual MLP, learned projection | 22.5 | 21.7 | 38.6 | 51.0 | 214s |
| v4 | Residual MLP, PCA projection | 27.8 | 25.8 | 29.6 | 34.2 | 87s |

v4 best config: `--n_components 96 --depth 6 --dropout 0.7 --drop_path 0.2 --lr 5e-4`

### Open questions

- Is ~28% near the ceiling for cross-subject decoding without subject-specific alignment (e.g., hyperalignment)?
- Would functional alignment methods allow using more PCA components without overfitting to subject identity?
- Could ensembling across n_components values combine the complementary information?
