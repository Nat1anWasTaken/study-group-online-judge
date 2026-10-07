#!/usr/bin/env bash
#SBATCH --job-name=lab5-profile
#SBATCH --account=ACD115198
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=H200:8
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=00:20:00
#SBATCH --output=logs/lab5-profile-%j.out
#SBATCH --error=logs/lab5-profile-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load cuda/13.0
export LAB5_ALLOCATION_SECONDS=1200
export LAB5_GIT_COMMIT="$(git rev-parse --short=8 HEAD)"
export LAB5_PROFILE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=2
export NCCL_SOCKET_IFNAME="=lo"
export UV_PROJECT_ENVIRONMENT="$PWD/work/lab4-fa2-env"
unset WANDB_SERVICE
profile_dir="$PWD/work/lab5-profiles/j$SLURM_JOB_ID"
mkdir -p "$profile_dir/source"
cp src/labs/lab5/{train_lab5.py,muon_lab5.py} "$profile_dir/source/"
export LAB5_OUTPUT="$profile_dir"
export LAB5_ALLOCATION_START="$(date +%s)"
echo "Profiling on $SLURMD_NODENAME"
srun --ntasks=1 --cpus-per-task=24 uv run --env-file "$PWD/.env" --no-sync python -u \
    -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
    "$LAB5_OUTPUT/source/train_lab5.py"
