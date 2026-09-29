#!/usr/bin/env bash
#SBATCH --job-name=lab4-train
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/lab4-%j.out
#SBATCH --error=logs/lab4-%j.err

set -euo pipefail

module purge
module load cuda/13.0

cd "$SLURM_SUBMIT_DIR"

# Before sbatch: mkdir -p logs; uv sync --no-sources --no-dev
# Run train_lab4.py prepare separately, outside this GPU allocation.

export WANDB_PROJECT=gpt2-training
export TOKENIZERS_PARALLELISM=false

uv run --no-sync accelerate launch \
    --multi_gpu \
    --num_processes 2 \
    --mixed_precision bf16 \
    --num_machines 1 \
    --dynamo_backend no \
    src/labs/train_lab4.py train
