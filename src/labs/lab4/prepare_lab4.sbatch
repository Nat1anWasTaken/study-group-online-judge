#!/usr/bin/env bash
#SBATCH --job-name=lab4-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lab4-prepare-%j.out
#SBATCH --error=logs/lab4-prepare-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

uv run --no-sync python -u src/labs/lab4/prepare_lab4.py
