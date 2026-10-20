#!/usr/bin/env bash
#SBATCH --job-name=lab5-w15-teacher3b-3b-cache
#SBATCH --account=ACD115198
#SBATCH --partition=dev
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=H200:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=384G
#SBATCH --time=03:00:00
#SBATCH --exclude=25a-hgpn027,25a-hgpn062,25a-hgpn071,25a-hgpn001,25a-hgpn173,25a-hgpn083
#SBATCH --output=/work/u9216411/acd115198/lab5-runs/tools/wave15-%j.out
#SBATCH --error=/work/u9216411/acd115198/lab5-runs/tools/wave15-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export LAB5_GIT_COMMIT="$(git rev-parse --short=8 HEAD)"
module purge
module load cuda/13.0
export REPO=/work/u9216411/acd115198/study-group-online-judge
export LAB5_DATA="${LAB5_DATA:-/work/u9216411/acd115198/lab5-data/dolma-seed42-8192-trunc8k}"
export UV_PROJECT_ENVIRONMENT="$REPO/work/lab4-fa2-env"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset NCCL_SOCKET_IFNAME
unset WANDB_SERVICE
[[ -f "$LAB5_DATA/manifest.json" ]]
export LAB5_CACHE_BLOCKS=366208
export LAB5_OUTPUT=/work/u9216411/acd115198/lab5-runs/lab5-w15-teacher3b-3b-cache
export LAB5_TEACHER_CACHE="${LAB5_TEACHER_CACHE:-$LAB5_OUTPUT/cache}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 40000))
mkdir -p "$LAB5_OUTPUT/source"
cp src/labs/lab5/{prepare_distill_lab5.py,train_lab5.py,muon_lab5.py,prepare_distill_lab5.sbatch} "$LAB5_OUTPUT/source/"
git diff HEAD > "$LAB5_OUTPUT/source/working-tree.patch"
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    bash -c 'exec uv run --env-file "$REPO/.env" --no-sync \
    python -u -m torch.distributed.run --nnodes="$SLURM_NNODES" --nproc-per-node=8 \
    --node-rank="$SLURM_NODEID" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    "$LAB5_OUTPUT/source/prepare_distill_lab5.py"'
