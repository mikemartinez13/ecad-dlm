#!/usr/bin/env bash
# Resume an existing frozen run on the user's regular allocation.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
: "${RUN:?Set RUN to the existing frozen run}"
SNAP="$REPO/run_snapshots/$RUN"
OUT="/scratch/gqs7zr/dream_reasoning/results/$RUN"
[[ -f "$SNAP/config.yaml" && -d "$OUT" ]] || { echo 'Missing frozen run' >&2; exit 1; }
/home/gqs7zr/.conda/envs/ecad-dlm/bin/python "$REPO/scripts/check_dream_scope.py" "$SNAP/config.yaml"
WRAPPER="$SNAP/run_long_v1.py"
if [[ ! -f "$WRAPPER" ]]; then
    cp "$REPO/scripts/run_dream_long.py" "$WRAPPER"
else
    cmp "$REPO/scripts/run_dream_long.py" "$WRAPPER"
fi
for name in $(compgen -e); do
    if [[ "$name" == SLURM_* || "$name" == SBATCH_* ]]; then
        [[ "$name" == SLURM_CONF ]] || unset "$name"
    fi
done
unset TMPDIR
cd "$SNAP"
job=$(sbatch --parsable --account=raiselab --partition=gpu --constraint=a100_80gb \
    --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=3-00:00:00 \
    --job-name="${JOB_NAME:-dream-long}" --output="$OUT/long-%j.out" \
    --wrap="exec /home/gqs7zr/.conda/envs/ecad-dlm/bin/python '$WRAPPER' --snapshot '$SNAP' --output '$OUT'")
printf '%s\traiselab\t72h\t1GPU\t4CPU\t48G\n' "$job" >> "$OUT/long_jobs.tsv"
printf 'job=%s\noutput=%s\n' "$job" "$OUT"
