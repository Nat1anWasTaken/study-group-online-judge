#!/usr/bin/env bash
#SBATCH --job-name=lab4-s-data
#SBATCH --partition=dev
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dedup-prepare-%j.out
#SBATCH --error=logs/dedup-prepare-%j.err

set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_DATASETS_CACHE=/work/nat1andotxyz/lab4/dedup-s-download-cache
uv run --project /work/nat1andotxyz/study-group-online-judge --env-file /work/nat1andotxyz/study-group-online-judge/.env --no-sync python -u "$PWD/src/labs/prepare_dedup_lab4.py"
