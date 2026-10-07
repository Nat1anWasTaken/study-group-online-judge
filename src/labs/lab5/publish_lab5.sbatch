#!/usr/bin/env bash
#SBATCH --job-name=lab5-publish
#SBATCH --account=ACD115198
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/lab5-publish-%j.out
#SBATCH --error=logs/lab5-publish-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_PROJECT_ENVIRONMENT="$PWD/work/lab4-fa2-env"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
srun --ntasks=1 --cpus-per-task=2 \
    uv run --env-file "$PWD/.env" --no-sync \
    python -u src/labs/lab5/publish_lab5.py "$@"
