#!/usr/bin/env bash
#SBATCH --job-name=lab5-w9-teacher-cache
#SBATCH --account=ACD115198
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=H200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=00:10:00
#SBATCH --exclude=25a-hgpn027,25a-hgpn071
#SBATCH --output=logs/lab5-w9-teacher-cache-%j.out
#SBATCH --error=logs/lab5-w9-teacher-cache-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export LAB5_GIT_COMMIT="$(git rev-parse --short=8 HEAD)"
module purge
module load cuda/13.0
REPO=/work/u9216411/acd115198/study-group-online-judge
export LAB5_DATA="${LAB5_DATA:-/work/u9216411/acd115198/lab5-data/dolma-seed42-8192-trunc8k}"
export UV_PROJECT_ENVIRONMENT="$REPO/work/lab4-fa2-env"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_SOCKET_IFNAME="=lo"
unset WANDB_SERVICE
[[ -f "$LAB5_DATA/manifest.json" ]]
export LAB5_CACHE_BLOCKS="${LAB5_CACHE_BLOCKS:-128}"
export LAB5_OUTPUT="/work/u9216411/acd115198/lab5-runs/$SLURM_JOB_NAME-j$SLURM_JOB_ID"
export LAB5_TEACHER_CACHE="${LAB5_TEACHER_CACHE:-$LAB5_OUTPUT/cache}"
mkdir -p "$LAB5_OUTPUT/source"
cp src/labs/lab5/{prepare_distill_lab5.py,train_lab5.py,muon_lab5.py,prepare_distill_lab5.sbatch} "$LAB5_OUTPUT/source/"
git diff HEAD > "$LAB5_OUTPUT/source/working-tree.patch"
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    uv run --env-file "$REPO/.env" --no-sync \
    python -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$SLURM_GPUS_ON_NODE" \
    "$LAB5_OUTPUT/source/prepare_distill_lab5.py"
