#!/usr/bin/env bash
set -euo pipefail

LAB4_SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$LAB4_SOURCE_DIR"
mkdir -p logs
prepare_job=$(sbatch --parsable --exclude=25a-hgpn003 --chdir="$LAB4_SOURCE_DIR" src/labs/prepare_dedup_lab4.sbatch)
prepare_job=${prepare_job%%;*}
printf 'Preparation job: %s\n' "$prepare_job"
printf '{"preparation_job":"%s"}\n' "$prepare_job" > logs/dedup-jobs.json
mapfile -t active_jobs < <(squeue --noheader --user "$USER" --format='%i')
dependency="afterok:$prepare_job"
for candidate in 475847 475899 475901 475903 475905 475975 475977 475980 475982; do
    for active in "${active_jobs[@]}"; do
        if [[ "$active" == "$candidate" ]]; then
            dependency+=":$candidate"
        fi
    done
done
launch_job=$(sbatch --parsable --exclude=25a-hgpn003 --chdir="$LAB4_SOURCE_DIR" \
    --dependency="$dependency" \
    --kill-on-invalid-dep=yes src/labs/launch_dedup_lab4.sbatch)
launch_job=${launch_job%%;*}
printf 'Selection/launch job: %s\n' "$launch_job"
printf '{"preparation_job":"%s","launch_job":"%s"}\n' "$prepare_job" "$launch_job" > logs/dedup-jobs.json
