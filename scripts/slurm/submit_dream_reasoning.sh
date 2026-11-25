#!/usr/bin/env bash
# One modest pilot allocation; freeze runtime sources (freeze_source.sh) before submitting.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
CONFIG="${CONFIG:-$REPO/ecad-dlm/experiments/configs/dream7b_reasoning_pilot.yaml}"
/home/gqs7zr/.conda/envs/ecad-dlm/bin/python "$REPO/scripts/check_dream_scope.py" "$CONFIG"
RUN="${RUN:-dream_reasoning_v1}"
OUT="${OUTPUT:-/scratch/gqs7zr/dream_reasoning/results/$RUN}"
# The commit holds ecad-dlm/ and scripts/, Dream's evaluation code (Dream/ is ignored, so it
# is added explicitly), the source list, and the config as config.yaml.
SOURCE="$("$REPO/scripts/slurm/freeze_source.sh" --force Dream/eval_instruct --force Dream/LICENSE \
    --include assets/docs/dream_sources.json --file "config.yaml=$CONFIG" "$RUN")"
mkdir -p "$OUT"
for name in $(compgen -e); do
    if [[ "$name" == SLURM_* || "$name" == SBATCH_* ]]; then
        [[ "$name" == SLURM_CONF ]] || unset "$name"
    fi
done
unset TMPDIR
cd "$REPO"
job=$(sbatch --parsable --account="${ACCOUNT:-raiselab}" --partition=gpu \
    --constraint="${GPU_TYPE:-a100_80gb}" --gres=gpu:1 --cpus-per-task=8 --mem=96G \
    --time="${TIME_LIMIT:-24:00:00}" --job-name="${JOB_NAME:-dream-reasoning}" \
    --output="$OUT/slurm-%j.out" "$REPO/scripts/slurm/run_frozen.sh" "$REPO" "$SOURCE" \
    bash scripts/slurm/run_dream_reasoning.sh "${STAGE:-all}" --config config.yaml --output "$OUT")
printf 'source %s\njob %s\n' "$SOURCE" "$job" > "$OUT/job.txt"
printf 'job=%s\noutput=%s\nsource=%s\n' "$job" "$OUT" "$SOURCE"
