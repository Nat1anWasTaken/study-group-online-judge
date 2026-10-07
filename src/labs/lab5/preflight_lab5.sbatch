#!/usr/bin/env bash
#SBATCH --job-name=lab5-preflight
#SBATCH --account=ACD115198
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=H200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/lab5-preflight-%j.out
#SBATCH --error=logs/lab5-preflight-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load cuda/13.0
export UV_PROJECT_ENVIRONMENT="$PWD/work/lab4-fa2-env"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=2
export NCCL_SOCKET_IFNAME="=lo"
srun --ntasks=1 --gpus-per-task=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    uv run --no-sync python -u -m torch.distributed.run --standalone --nproc-per-node=1 \
    src/labs/lab5/preflight_lab5.py
