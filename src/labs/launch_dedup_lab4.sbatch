#!/usr/bin/env bash
#SBATCH --job-name=lab4-s-launch
#SBATCH --partition=dev
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/dedup-launch-%j.out
#SBATCH --error=logs/dedup-launch-%j.err

set -euo pipefail
uv run --project /work/nat1andotxyz/study-group-online-judge --env-file /work/nat1andotxyz/study-group-online-judge/.env --no-sync python -u "$PWD/src/labs/launch_dedup_lab4.py"
