#!/usr/bin/env bash
#SBATCH --job-name=lab4-l1-setup
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=logs/lab4-setup-%j.out
#SBATCH --error=logs/lab4-setup-%j.err
#SBATCH --partition=dev

set -euo pipefail

module purge
module load cuda/13.0

LAB4_SOURCE_DIR="$PWD"
LAB4_SHARED_DIR=/work/nat1andotxyz/study-group-online-judge
LAB4_ENV_DIR="$LAB4_SOURCE_DIR/work/lab4-fa2-env"
export MAX_JOBS=8
export FLASH_ATTN_CUDA_ARCHS=90

uv venv --python "$LAB4_SHARED_DIR/.venv/bin/python" "$LAB4_ENV_DIR"
printf '%s\n' "$LAB4_SHARED_DIR/.venv/lib/python3.14/site-packages" > "$LAB4_ENV_DIR/lib/python3.14/site-packages/lab4-shared.pth"
uv pip install --python "$LAB4_ENV_DIR/bin/python" --no-deps ninja packaging psutil setuptools einops
uv pip install --python "$LAB4_ENV_DIR/bin/python" --no-build-isolation --no-deps flash-attn==2.8.3.post1
"$LAB4_ENV_DIR/bin/python" -c 'import torch, flash_attn; print("torch", torch.__version__, "CUDA", torch.version.cuda, "FlashAttention", flash_attn.__version__)'
