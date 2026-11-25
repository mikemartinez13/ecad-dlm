#!/usr/bin/env bash
# One bounded GSM8K benchmark, no search or automatic successors. Freezes the source
# (freeze_source.sh) and runs that commit through run_frozen.sh.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
RUN=dream_gsm8k_sharing_check_v1
OUT="/scratch/gqs7zr/dream_reasoning/results/$RUN"
[[ ! -e "$OUT" ]] || { echo 'Benchmark already prepared; refusing duplicate submission' >&2; exit 1; }
SOURCE="$("$REPO/scripts/slurm/freeze_source.sh" --force Dream/eval_instruct/lm_eval/tasks/minerva_math/utils.py \
    --file "config.yaml=$REPO/ecad-dlm/experiments/configs/dream7b_gsm8k_full_96.yaml" "$RUN")"
mkdir -p "$OUT"
for name in $(compgen -e); do
    if [[ "$name" == SLURM_* || "$name" == SBATCH_* ]]; then
        [[ "$name" == SLURM_CONF ]] || unset "$name"
    fi
done
unset TMPDIR
cd "$REPO"
job=$(sbatch --parsable --account=raiselab --partition=gpu --constraint=a100_80gb \
    --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=04:00:00 --job-name=dream-share-check \
    --output="$OUT/slurm-%j.out" "$REPO/scripts/slurm/run_frozen.sh" "$REPO" "$SOURCE" \
    bash scripts/slurm/run_dream_sharing_check.sh --config config.yaml --output "$OUT")
printf 'source %s\njob %s\n' "$SOURCE" "$job" > "$OUT/job.txt"
printf 'job=%s\noutput=%s\nsource=%s\n' "$job" "$OUT" "$SOURCE"
