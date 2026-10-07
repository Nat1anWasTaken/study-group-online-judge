#!/usr/bin/env bash
set -euo pipefail

LAB4_SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$LAB4_SOURCE_DIR"
mkdir -p logs
sbatch_options=(--exclude="${LAB4_EXCLUDE_NODES:-25a-hgpn003}")

training_job=$(sbatch "${sbatch_options[@]}" --parsable --chdir="$LAB4_SOURCE_DIR" src/labs/lab4/train_lab4.sbatch)
training_job=${training_job%%;*}
printf 'Training job: %s\n' "$training_job"

if publish_job=$(sbatch "${sbatch_options[@]}" --parsable --chdir="$LAB4_SOURCE_DIR" \
    --dependency="afterok:$training_job" --kill-on-invalid-dep=yes \
    src/labs/lab4/publish_lab4.sbatch "$training_job"); then
    publish_job=${publish_job%%;*}
    printf 'Evaluation/publish job: %s\n' "$publish_job"
    printf 'Cancel both jobs: scancel %s %s\n' "$training_job" "$publish_job"
else
    printf 'Evaluation/publish submission failed; training job %s is still submitted.\n' "$training_job" >&2
    printf 'From %s, retry with:\n' "$LAB4_SOURCE_DIR" >&2
    printf 'sbatch --dependency=afterok:%s --kill-on-invalid-dep=yes src/labs/lab4/publish_lab4.sbatch %s\n' "$training_job" "$training_job" >&2
    exit 1
fi
