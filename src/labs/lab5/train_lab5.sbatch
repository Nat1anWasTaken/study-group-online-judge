#!/usr/bin/env bash
#SBATCH --job-name=lab5-muon-pilot
#SBATCH --account=ACD115198
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=H200:8
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=logs/lab5-train-%j.out
#SBATCH --error=logs/lab5-train-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export LAB5_ALLOCATION_START="$(date +%s)"
export LAB5_ALLOCATION_SECONDS="${LAB5_ALLOCATION_SECONDS:-3600}"
export LAB5_GIT_COMMIT="$(git rev-parse --short=8 HEAD)"
module purge
module load cuda/13.0
REPO=/work/u9216411/acd115198/study-group-online-judge
export LAB5_DATA="${LAB5_DATA:-/home/nat1andotxyz/lab5/dolma-seed42-8192-v1}"
export UV_PROJECT_ENVIRONMENT="$REPO/work/lab4-fa2-env"
export WANDB_PROJECT=lab5-training-llama
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=2
export NCCL_SOCKET_IFNAME="=lo"
unset WANDB_SERVICE
[[ -f "$LAB5_DATA/manifest.json" ]]
export LAB5_OUTPUT="${LAB5_OUTPUT:-/work/u9216411/acd115198/lab5-runs/$SLURM_JOB_NAME-j$SLURM_JOB_ID}"
mkdir -p "$LAB5_OUTPUT/source"
cp src/labs/lab5/{train_lab5.py,muon_lab5.py,train_lab5.sbatch} "$LAB5_OUTPUT/source/"
git diff HEAD > "$LAB5_OUTPUT/source/working-tree.patch"
srun --ntasks=1 --cpus-per-task=24 \
    uv run --env-file "$REPO/.env" --no-sync \
    python -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
    "$LAB5_OUTPUT/source/train_lab5.py"
