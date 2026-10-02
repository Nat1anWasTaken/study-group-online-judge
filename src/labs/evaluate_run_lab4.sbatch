#!/usr/bin/env bash
#SBATCH --job-name=lab4-w3-eval
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/lab4-w3-eval-%j.out
#SBATCH --error=logs/lab4-w3-eval-%j.err

set -euo pipefail

module purge
module load cuda/13.0

LAB4_SOURCE_DIR="$PWD"
export WANDB_DIR="$LAB4_SOURCE_DIR"
export OMP_NUM_THREADS=4
export HF_HUB_DISABLE_XET=1
uv run --project /work/nat1andotxyz/study-group-online-judge --env-file /work/nat1andotxyz/study-group-online-judge/.env --no-sync python -u "$LAB4_SOURCE_DIR/src/labs/evaluate_run_lab4.py" "$1"
