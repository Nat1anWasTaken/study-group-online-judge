#!/usr/bin/env bash
#SBATCH --job-name=lab5-lr-tune
#SBATCH --account=ACD115198
#SBATCH --partition=dev
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=H200:8
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=00:40:00
#SBATCH --output=logs/lab5-tune-%A_%a.out
#SBATCH --error=logs/lab5-tune-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load cuda/13.0
rates=(0.0003 0.001 0.002)
export LAB5_LR="${rates[$SLURM_ARRAY_TASK_ID]}"
export LAB5_TUNE=1
export LAB5_PROFILE=0
export LAB5_ALLOCATION_START="$(date +%s)"
export LAB5_ALLOCATION_SECONDS=2400
export LAB5_GIT_COMMIT="$(git rev-parse --short=8 HEAD)"
export LAB5_OUTPUT="/home/nat1andotxyz/lab5/tuning/j$SLURM_ARRAY_JOB_ID/lr$LAB5_LR"
export UV_PROJECT_ENVIRONMENT="$PWD/work/lab4-fa2-env"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=2
export NCCL_SOCKET_IFNAME="=lo"
unset WANDB_SERVICE
mkdir -p "$LAB5_OUTPUT/source"
cp src/labs/lab5/{train_lab5.py,muon_lab5.py,tune_lab5.sbatch} "$LAB5_OUTPUT/source/"
git diff HEAD > "$LAB5_OUTPUT/source/working-tree.patch"
echo "Learning rate $LAB5_LR on $SLURMD_NODENAME"
srun --ntasks=1 --cpus-per-task=24 \
    uv run --env-file "$PWD/.env" --no-sync \
    python -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
    "$LAB5_OUTPUT/source/train_lab5.py"
