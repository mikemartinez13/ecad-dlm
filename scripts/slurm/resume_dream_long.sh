#!/usr/bin/env bash
# Resume an existing frozen run on the user's regular allocation, on the run's own commit
# (freeze_source.sh --existing). The long-run wrapper is the commit's run_long_v1.py if it has
# one, else its scripts/run_dream_long.py; a commit with neither gets the current wrapper added
# as run_long_v1.py, in a new commit on top of the run's.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
: "${RUN:?Set RUN to the existing frozen run}"
OUT="/scratch/gqs7zr/dream_reasoning/results/$RUN"
[[ -d "$OUT" ]] || { echo "Missing output $OUT" >&2; exit 1; }
SOURCE="$("$REPO/scripts/slurm/freeze_source.sh" --existing "$RUN")"
git -C "$REPO" cat-file -e "$SOURCE:config.yaml" || { echo "Run $RUN has no config.yaml in its source" >&2; exit 1; }
FROZEN_CONFIG="$(mktemp)"
trap 'rm -f "$FROZEN_CONFIG"' EXIT
git -C "$REPO" show "$SOURCE:config.yaml" > "$FROZEN_CONFIG"
/home/gqs7zr/.conda/envs/ecad-dlm/bin/python "$REPO/scripts/check_dream_scope.py" "$FROZEN_CONFIG"
if git -C "$REPO" cat-file -e "$SOURCE:run_long_v1.py" 2>/dev/null; then
    WRAPPER=run_long_v1.py
elif git -C "$REPO" cat-file -e "$SOURCE:scripts/run_dream_long.py" 2>/dev/null; then
    WRAPPER=scripts/run_dream_long.py
else
    SOURCE="$("$REPO/scripts/slurm/freeze_source.sh" --again --base "$SOURCE" \
        --file "run_long_v1.py=$REPO/scripts/run_dream_long.py" "$RUN")"
    WRAPPER=run_long_v1.py
fi
for name in $(compgen -e); do
    if [[ "$name" == SLURM_* || "$name" == SBATCH_* ]]; then
        [[ "$name" == SLURM_CONF ]] || unset "$name"
    fi
done
unset TMPDIR
cd "$REPO"
# The wrapper runs in the unpacked commit; --snapshot . is that copy.
job=$(sbatch --parsable --account=raiselab --partition=gpu --constraint=a100_80gb \
    --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=3-00:00:00 \
    --job-name="${JOB_NAME:-dream-long}" --output="$OUT/long-%j.out" \
    "$REPO/scripts/slurm/run_frozen.sh" "$REPO" "$SOURCE" \
    /home/gqs7zr/.conda/envs/ecad-dlm/bin/python "$WRAPPER" --snapshot . --output "$OUT")
printf '%s\traiselab\t72h\t1GPU\t4CPU\t48G\t%s\n' "$job" "$SOURCE" >> "$OUT/long_jobs.tsv"
printf 'job=%s\noutput=%s\nsource=%s\n' "$job" "$OUT" "$SOURCE"
