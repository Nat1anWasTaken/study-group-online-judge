#!/usr/bin/env bash
#SBATCH --job-name=lab4-publish
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/lab4-publish-%j.out
#SBATCH --error=logs/lab4-publish-%j.err

set -euo pipefail
LAB4_SOURCE_DIR="$PWD"
export WANDB_DIR="$LAB4_SOURCE_DIR"
export OMP_NUM_THREADS=4
uv run --project /work/nat1andotxyz/study-group-online-judge --env-file /work/nat1andotxyz/study-group-online-judge/.env --no-sync python -u "$LAB4_SOURCE_DIR/src/labs/publish_lab4.py" "$1"
