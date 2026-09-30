#!/usr/bin/env bash
#SBATCH --job-name=lab4-prefix
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lab4-prefix-%j.out
#SBATCH --error=logs/lab4-prefix-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
uv run --project /work/nat1andotxyz/study-group-online-judge --env-file /work/nat1andotxyz/study-group-online-judge/.env --no-sync python -u "$SLURM_SUBMIT_DIR/src/labs/prepare_lab4_prefix.py"
