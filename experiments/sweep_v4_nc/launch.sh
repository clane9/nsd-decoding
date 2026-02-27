#!/usr/bin/env bash
#SBATCH --job-name=sweep_v4_nc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --nodelist=n-1,n-2,n-3,n-4
#SBATCH --account=training
#SBATCH --array=0-11

set -euo pipefail

ROOT="/data/connor/nsd-decoding"
cd $ROOT

EXP_NAME="sweep_v4_nc"
EXP_DIR="experiments/${EXP_NAME}"

configs=(
    # --- pure n_components sweep (default hparams) ---
    # 0-8
    "--n_components 32  --notes nc32"
    "--n_components 48  --notes nc48"
    "--n_components 64  --notes nc64"
    "--n_components 80  --notes nc80"
    "--n_components 96  --notes nc96"
    "--n_components 112 --notes nc112"
    "--n_components 128 --notes nc128"
    "--n_components 160 --notes nc160"
    "--n_components 192 --notes nc192"
    # --- best combos: low nc + d6_highreg ---
    # 9-11
    "--n_components 64  --depth 6 --dropout 0.7 --drop_path 0.2 --lr 5e-4 --notes nc64_d6hr"
    "--n_components 96  --depth 6 --dropout 0.7 --drop_path 0.2 --lr 5e-4 --notes nc96_d6hr"
    "--n_components 128 --depth 6 --dropout 0.7 --drop_path 0.2 --lr 5e-4 --notes nc128_d6hr"
)
config=${configs[SLURM_ARRAY_TASK_ID]}

uv run python src/nsd_decoding/nsd_flat_cococlip_decoding_v4.py $config
