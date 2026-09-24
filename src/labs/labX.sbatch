#!/usr/bin/env bash
set -euo pipefail

# Copy this template to src/labs/labX.sbatch for a GPU task, replacing X with
# the task number. The judge supplies resource limits on the sbatch command line.
module purge
module load cuda/13.0

cd "$JUDGE_SUBMISSION_DIR"
uv sync --no-sources --no-dev
export PYTHONPATH="$JUDGE_TRUSTED_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
uv run --no-sync python -u -m judge.run_task \
  --task "$JUDGE_TASK_ID" \
  --submission "$JUDGE_SUBMISSION_DIR" \
  --output "$JUDGE_OUTPUT_DIR/result.json"
