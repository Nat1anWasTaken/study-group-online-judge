#!/usr/bin/env bash
#SBATCH --job-name=lab4-u4-train
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/lab4-%j.out
#SBATCH --error=logs/lab4-%j.err
#SBATCH --partition=dev

set -euo pipefail
unset WANDB_SERVICE

export LAB4_START_TIME=$(date +%s)

module purge
module load cuda/13.0

LAB4_SOURCE_DIR="$PWD"

export WANDB_PROJECT=gpt2-training
export TOKENIZERS_PARALLELISM=false
export WANDB_DIR="$LAB4_SOURCE_DIR"

uv run --project /work/nat1andotxyz/study-group-online-judge --env-file /work/nat1andotxyz/study-group-online-judge/.env --no-sync accelerate launch \
    --multi_gpu \
    --num_processes 2 \
    --main_process_port "$((20000 + SLURM_JOB_ID % 40000))" \
    --mixed_precision bf16 \
    --num_machines 1 \
    --dynamo_backend no \
    "$LAB4_SOURCE_DIR/src/labs/train_lab4.py"
