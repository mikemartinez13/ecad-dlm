#!/usr/bin/env bash
# One modest pilot allocation; freeze runtime sources before submitting.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
/home/gqs7zr/.conda/envs/ecad-dlm/bin/python "$REPO/scripts/check_dream_scope.py" \
    "${CONFIG:-$REPO/ecad-dlm/experiments/configs/dream7b_reasoning_pilot.yaml}"
RUN="${RUN:-dream_reasoning_v1}"
SNAP="$REPO/run_snapshots/$RUN"
OUT="${OUTPUT:-/scratch/gqs7zr/dream_reasoning/results/$RUN}"
[[ ! -e "$SNAP" ]] || { echo "Snapshot already exists: $SNAP" >&2; exit 1; }
mkdir -p "$SNAP/Dream" "$OUT"
rsync -a --exclude=__pycache__ "$REPO/ecad-dlm" "$SNAP/"
rsync -a --exclude=__pycache__ "$REPO/Dream/eval_instruct" "$REPO/Dream/LICENSE" "$SNAP/Dream/"
cp "$REPO/scripts/slurm/run_dream_reasoning.sh" "$SNAP/"
cp "$REPO/assets/docs/dream_sources.json" "$SNAP/"
cp "${CONFIG:-$REPO/ecad-dlm/experiments/configs/dream7b_reasoning_pilot.yaml}" "$SNAP/config.yaml"
for name in $(compgen -e); do
    if [[ "$name" == SLURM_* || "$name" == SBATCH_* ]]; then
        [[ "$name" == SLURM_CONF ]] || unset "$name"
    fi
done
unset TMPDIR
cd "$SNAP"
job=$(sbatch --parsable --account="${ACCOUNT:-raiselab}" --partition=gpu \
    --constraint="${GPU_TYPE:-a100_80gb}" --gres=gpu:1 --cpus-per-task=8 --mem=96G \
    --time="${TIME_LIMIT:-24:00:00}" --job-name="${JOB_NAME:-dream-reasoning}" \
    --output="$OUT/slurm-%j.out" "$SNAP/run_dream_reasoning.sh" "${STAGE:-all}" \
    --config "$SNAP/config.yaml" --output "$OUT")
printf '%s\n' "$job" > "$SNAP/job.txt"
printf 'job=%s\noutput=%s\nsnapshot=%s\n' "$job" "$OUT" "$SNAP"
