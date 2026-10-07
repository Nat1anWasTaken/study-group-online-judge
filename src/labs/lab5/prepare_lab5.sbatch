#!/usr/bin/env bash
#SBATCH --job-name=lab5-prepare
#SBATCH --account=ACD115198
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=H200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lab5-prepare-%j.out
#SBATCH --error=logs/lab5-prepare-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export LAB5_DATA="${LAB5_DATA:-/home/nat1andotxyz/lab5/dolma-seed42-8192-v1}"
export UV_PROJECT_ENVIRONMENT="$PWD/work/lab4-fa2-env"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -s <<'EOF_COMPUTE'
set -euo pipefail
export LAB5_CACHE="$(mktemp -d /tmp/lab5-prepare-$SLURM_JOB_ID-XXXXXX)"
export HF_HUB_CACHE="$LAB5_CACHE/hub"
export HF_DATASETS_CACHE="$LAB5_CACHE/datasets"
[[ "$(df -Pk "$LAB5_CACHE" | awk 'NR==2 {print $4}')" -ge 1468006400 ]]
mkdir -p "$(dirname "$LAB5_DATA")"
[[ "$(df -Pk "$(dirname "$LAB5_DATA")" | awk 'NR==2 {print $4}')" -ge 62914560 ]]
printf 'Host: %s; cache: %s\n' "$(hostname)" "$LAB5_CACHE"
uv run --env-file "$PWD/.env" --no-sync python -u src/labs/lab5/prepare_lab5.py
rm -rf -- "$LAB5_CACHE"
EOF_COMPUTE
