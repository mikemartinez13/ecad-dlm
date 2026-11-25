#!/usr/bin/env bash
# One bounded GSM8K benchmark, no search or automatic successors.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
RUN=dream_gsm8k_sharing_check_v1
SNAP="$REPO/run_snapshots/$RUN"
OUT="/scratch/gqs7zr/dream_reasoning/results/$RUN"
[[ ! -e "$SNAP" && ! -e "$OUT" ]] || { echo 'Benchmark already prepared; refusing duplicate submission' >&2; exit 1; }
mkdir -p "$SNAP/Dream/eval_instruct/lm_eval/tasks/minerva_math" "$OUT"
rsync -a --exclude=__pycache__ "$REPO/ecad-dlm" "$SNAP/"
cp "$REPO/Dream/eval_instruct/lm_eval/tasks/minerva_math/utils.py" "$SNAP/Dream/eval_instruct/lm_eval/tasks/minerva_math/"
cp "$REPO/ecad-dlm/experiments/configs/dream7b_gsm8k_full_96.yaml" "$SNAP/config.yaml"
cp "$REPO/scripts/slurm/run_dream_sharing_check.sh" "$SNAP/"
for name in $(compgen -e); do
    if [[ "$name" == SLURM_* || "$name" == SBATCH_* ]]; then
        [[ "$name" == SLURM_CONF ]] || unset "$name"
    fi
done
unset TMPDIR
cd "$SNAP"
job=$(sbatch --parsable --account=raiselab --partition=gpu --constraint=a100_80gb \
    --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=04:00:00 --job-name=dream-share-check \
    --output="$OUT/slurm-%j.out" "$SNAP/run_dream_sharing_check.sh" \
    --config "$SNAP/config.yaml" --output "$OUT")
printf '%s\n' "$job" > "$SNAP/job.txt"
printf 'job=%s\noutput=%s\nsnapshot=%s\n' "$job" "$OUT" "$SNAP"
