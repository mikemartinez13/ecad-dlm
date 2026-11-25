#!/usr/bin/env bash
# Freeze the source (scripts/slurm/freeze_source.sh) and submit the hand-tuned comparison
# (run_hand_tuned_comparison_rivanna.sh), then EvalPlus's extra tests on its programs once it
# succeeds. Account raiselab by default.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
NAME=v4_hand_tuned_comparison
LOGS=/scratch/gqs7zr/ai4mat/ecad/logs
OUTPUT=/scratch/gqs7zr/ai4mat/ecad/results/dlm_kv_policy/v4_hand_tuned_comparison
RECORD="$LOGS/runs/$NAME.txt"
ACCOUNT="${ACCOUNT:-raiselab}"
SOURCE="$("$REPO/scripts/slurm/freeze_source.sh" "$NAME")"
FROZEN=("$REPO/scripts/slurm/run_frozen.sh" "$REPO" "$SOURCE")
mkdir -p "$LOGS/runs"
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "$name"; done
unset TMPDIR
cd "$REPO"
compare=$(sbatch --parsable --export=ALL --job-name=hand-tuned-cmp --account="$ACCOUNT" --partition=gpu \
    --constraint=a100_80gb --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=16:00:00 \
    --output="$LOGS/slurm-%x-%j.out" "${FROZEN[@]}" bash ecad-dlm/experiments/scripts/slurm/run_hand_tuned_comparison_rivanna.sh)
plus=$(sbatch --parsable --export=ALL --job-name=hand-tuned-plus --account="$ACCOUNT" --partition=standard \
    --cpus-per-task=8 --mem=32G --time=12:00:00 --dependency="afterok:$compare" --output="$LOGS/slurm-%x-%j.out" \
    "${FROZEN[@]}" bash ecad-dlm/experiments/scripts/slurm/run_proxy_cost_prerequisites_rivanna.sh search-plus \
    --config ecad-dlm/experiments/configs/diffucoder7b_audit_96tasks.yaml --output "$OUTPUT")
printf 'source %s\ncompare %s\nplus %s\n' "$SOURCE" "$compare" "$plus" | tee "$RECORD"
echo "Log: $LOGS/slurm-hand-tuned-cmp-$compare.out"
