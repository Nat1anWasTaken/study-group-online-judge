#!/usr/bin/env bash
#SBATCH --job-name=lab4-oj
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lab4-oj-%j.out
#SBATCH --error=logs/lab4-oj-%j.err

set -euo pipefail
LAB4_SOURCE_DIR="$PWD"
module purge
module load cuda/13.0
export PYTHONPATH=/work/nat1andotxyz/study-group-online-judge/src
export WANDB_DIR="$LAB4_SOURCE_DIR"
uv run --project /work/nat1andotxyz/study-group-online-judge --env-file /work/nat1andotxyz/study-group-online-judge/.env --no-sync python -u "$LAB4_SOURCE_DIR/src/labs/evaluate_lab4.py" "$1" "$2"
