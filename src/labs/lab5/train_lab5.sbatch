#!/usr/bin/env bash
#SBATCH --job-name=lab5-w12-decay-start40
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=H200:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=03:00:00
#SBATCH --exclude=25a-hgpn027,25a-hgpn062,25a-hgpn071
#SBATCH --output=/work/u9216411/acd115198/lab5-runs/tools/wave12-%j.out
#SBATCH --error=/work/u9216411/acd115198/lab5-runs/tools/wave12-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ "$SLURM_SUBMIT_DIR" == "/work/u9216411/acd115198/lab5-worktrees/${SLURM_JOB_NAME#lab5-}" ]]
[[ "$SLURM_GPUS_ON_NODE" == 8 ]]
export LAB5_ALLOCATION_START="$(date +%s)"
export LAB5_ALLOCATION_SECONDS=10800
export LAB5_GIT_COMMIT="$(git rev-parse --short=8 HEAD)"
module purge
module load cuda/13.0
REPO=/work/u9216411/acd115198/study-group-online-judge
export LAB5_DATA=/work/u9216411/acd115198/lab5-data/dolma-seed42-8192-trunc8k
export LAB5_TEACHER_CACHE=/work/u9216411/acd115198/lab5-runs/lab5-w10-teacher128-3b-writefix-j559809/cache
export LAB5_STEPS=45776
export LAB5_SAVE_STEPS=27465
export LAB5_RESUME=/work/u9216411/acd115198/lab5-runs/lab5-w12-stable-prefix-j${LAB5_PREFIX_JOB_ID:?}/step-018310
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
[[ -f "$LAB5_TEACHER_CACHE/cache.json" ]]
[[ -s "$LAB5_TEACHER_CACHE/verification.json" && ! -e "$LAB5_TEACHER_CACHE/verification.json.incomplete" ]]
[[ -s "$LAB5_RESUME/optimizer.pt" && -s "$LAB5_RESUME/model/config.json" && -s "$LAB5_RESUME/model/model.safetensors.index.json" ]]
export LAB5_OUTPUT="/work/u9216411/acd115198/lab5-runs/$SLURM_JOB_NAME-j$SLURM_JOB_ID"
mkdir -p "$LAB5_OUTPUT/source"
cp src/labs/lab5/{train_lab5.py,muon_lab5.py,train_lab5.sbatch} "$LAB5_OUTPUT/source/"
cp /work/u9216411/acd115198/lab5-runs/tools/wave10-student-3b-fixed-source/verify_wave10_cache.py "$LAB5_OUTPUT/source/"
cp "$LAB5_TEACHER_CACHE/verification.json" "$LAB5_OUTPUT/source/cache-independent-verification.json"
git rev-parse HEAD > "$LAB5_OUTPUT/source/git-head.txt"
git diff HEAD > "$LAB5_OUTPUT/source/working-tree.patch"
PREFLIGHT=$(cat <<'EOF'
import json, os, socket, torch
from pathlib import Path
assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID")
assert socket.gethostname().split(".")[0] == os.environ["SLURMD_NODENAME"].split(".")[0]
resume, cache = Path(os.environ["LAB5_RESUME"]), Path(os.environ["LAB5_TEACHER_CACHE"])
parent = json.loads((resume.parent / "training-config.json").read_text())
cache_metadata = json.loads((cache / "cache.json").read_text())
distillation = parent["distillation"]
assert parent["seed"] == 42 and parent["effective_batch_tokens"] == 65536 and parent["world_size"] == 8
assert parent["gradient_accumulation_steps"] == 1 and parent["learning_rate"] == 0.001 and parent["schedule"].startswith("warmup_800_")
assert parent["optimizer_recipe"] == "muon_elementwise_gradient_precond_exact_rms_hidden_adamw_rest" and parent["muon_adjust_lr_fn"] == "exact_rms_0.2"
assert distillation["alpha"] == 0.25 and distillation["top_k"] == 128 and distillation["teacher_cache"] == str(cache)
assert distillation["teacher_id"] == cache_metadata["teacher_id"] and distillation["teacher_revision"] == cache_metadata["teacher_revision"]
assert parent["data_manifest"] == cache_metadata["data_manifest"] == json.loads((Path(os.environ["LAB5_DATA"]) / "manifest.json").read_text())
assert parent["data_manifest"]["shuffle_seed"] == 42
assert parent["run_steps"] == 18310 and parent["schedule_steps"] == 45776
assert all((resume.parent / "source" / name).is_file() for name in ("train_lab5.py", "muon_lab5.py", "train_lab5.sbatch", "working-tree.patch"))
state = torch.load(resume / "optimizer.pt", map_location="cpu", weights_only=False)
step = state["step"]
del state
assert step == 18310 <= parent["schedule_steps"] * (1 - float(parent["schedule"].rsplit("_", 1)[1]))
ancestry = {
    "parent_checkpoint": str(resume),
    "parent_config": str(resume.parent / "training-config.json"),
    "initial_winner_checkpoint": "/work/u9216411/acd115198/lab5-runs/lab5-w11-teacher75-j561061/step-004800",
    "initial_winner_alpha": 0.25,
    "shared_prefix_job": os.environ.get("LAB5_PREFIX_JOB_ID"),
    "parent_step": step,
    "target_end_step": 45776,
    "schedule_horizon": 45776,
    "conservative_h200_hour_cap_including_ancestors": 44,
}
assert ancestry["conservative_h200_hour_cap_including_ancestors"] <= 64
(Path(os.environ["LAB5_OUTPUT"]) / "ancestry.json").write_text(json.dumps(ancestry, indent=2))
print(json.dumps(ancestry), flush=True)
EOF
)
srun --ntasks=1 --cpus-per-task=96 \
    uv run --project "$REPO" --env-file "$REPO/.env" --no-sync \
    python "$LAB5_OUTPUT/source/verify_wave10_cache.py" "$LAB5_TEACHER_CACHE"
srun --ntasks=1 --cpus-per-task=96 \
    uv run --project "$REPO" --env-file "$REPO/.env" --no-sync \
    python -c "$PREFLIGHT"
srun --ntasks=1 --cpus-per-task=96 \
    uv run --project "$REPO" --env-file "$REPO/.env" --no-sync \
    python -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
    "$LAB5_OUTPUT/source/train_lab5.py"
