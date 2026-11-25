#!/usr/bin/env bash
# Freeze the source and submit the hand-tuned comparison (run_hand_tuned_comparison_rivanna.sh),
# then EvalPlus's extra tests on its programs once it succeeds. Account raiselab by default.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SNAP="$REPO/run_snapshots/v4_hand_tuned_comparison"
LOGS=/scratch/gqs7zr/ai4mat/ecad/logs
OUTPUT=/scratch/gqs7zr/ai4mat/ecad/results/dlm_kv_policy/v4_hand_tuned_comparison
ACCOUNT="${ACCOUNT:-raiselab}"
[[ -e "$SNAP/jobs.txt" ]] && { echo "$SNAP/jobs.txt exists; refusing duplicate submission" >&2; exit 1; }
mkdir -p "$SNAP" "$LOGS"
rsync -a --delete --exclude=__pycache__ "$REPO/ecad-dlm" "$SNAP/"
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "$name"; done
unset TMPDIR
cd "$SNAP"
SLURM_DIR="$SNAP/ecad-dlm/experiments/scripts/slurm"
compare=$(sbatch --parsable --export=ALL --job-name=hand-tuned-cmp --account="$ACCOUNT" --partition=gpu \
    --constraint=a100_80gb --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=16:00:00 \
    --output="$LOGS/slurm-%x-%j.out" "$SLURM_DIR/run_hand_tuned_comparison_rivanna.sh")
plus=$(sbatch --parsable --export=ALL --job-name=hand-tuned-plus --account="$ACCOUNT" --partition=standard \
    --cpus-per-task=8 --mem=32G --time=12:00:00 --dependency="afterok:$compare" --output="$LOGS/slurm-%x-%j.out" \
    "$SLURM_DIR/run_proxy_cost_prerequisites_rivanna.sh" search-plus \
    --config "$SNAP/ecad-dlm/experiments/configs/diffucoder7b_audit_96tasks.yaml" --output "$OUTPUT")
printf 'compare %s\nplus %s\n' "$compare" "$plus" | tee -a "$SNAP/jobs.txt"
echo "Log: $LOGS/slurm-hand-tuned-cmp-$compare.out"
