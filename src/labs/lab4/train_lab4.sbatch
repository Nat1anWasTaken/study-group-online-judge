#!/usr/bin/env bash
#SBATCH --job-name=lab4-train
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/lab4-%j.out
#SBATCH --error=logs/lab4-%j.err
#SBATCH --partition=dev

set -euo pipefail

export LAB4_START_TIME=$(date +%s)

module purge
module load cuda/13.0

LAB4_SOURCE_DIR="$PWD"
export UV_PROJECT_ENVIRONMENT="$LAB4_SOURCE_DIR/work/lab4-fa2-env"

unset WANDB_SERVICE
export NCCL_SOCKET_IFNAME="=lo"
export WANDB_PROJECT=gpt2-training
export TOKENIZERS_PARALLELISM=false
export WANDB_DIR="$LAB4_SOURCE_DIR"
printf 'Allocated CUDA devices: %s\n' "$CUDA_VISIBLE_DEVICES"

uv run --project "$LAB4_SOURCE_DIR" --env-file /work/nat1andotxyz/study-group-online-judge/.env --no-sync python -m accelerate.commands.launch \
    --multi_gpu \
    --gpu_ids all \
    --num_processes 2 \
    --main_process_port "$((20000 + SLURM_JOB_ID % 40000))" \
    --mixed_precision bf16 \
    --num_machines 1 \
    --dynamo_backend no \
    "$LAB4_SOURCE_DIR/src/labs/lab4/train_lab4.py"
