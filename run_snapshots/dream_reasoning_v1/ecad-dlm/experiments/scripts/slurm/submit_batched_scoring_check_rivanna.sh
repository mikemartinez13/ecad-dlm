#!/usr/bin/env bash
# Freeze the source and submit the vectorized scoring check. Test job: raiselab-paid by default.
# Resubmitting resumes from the saved tasks.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SNAP="$REPO/run_snapshots/batched_scoring_check_v1"
ARTIFACTS=/scratch/gqs7zr/ai4mat/ecad
OUTPUT="$ARTIFACTS/results/dlm_kv_policy/batched_teacher_forcing_check_v1"
LOGS="$ARTIFACTS/logs"
ACCOUNT="${ACCOUNT:-raiselab-paid}"
mkdir -p "$SNAP" "$LOGS"
rsync -a --delete --exclude=__pycache__ "$REPO/ecad-dlm" "$SNAP/"
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "$name"; done
unset TMPDIR
cd "$SNAP"
job=$(OUTPUT="$OUTPUT" sbatch --parsable --export=ALL --job-name=batched-scoring-check \
    --account="$ACCOUNT" --partition=gpu --constraint=a100_80gb --gres=gpu:1 --cpus-per-task=8 --mem=96G \
    --time="${TIME:-08:00:00}" --output="$LOGS/slurm-%x-%j.out" \
    "$SNAP/ecad-dlm/experiments/scripts/slurm/run_batched_scoring_check_rivanna.sh")
echo "check $job" | tee -a "$SNAP/jobs.txt"
echo "Log: $LOGS/slurm-batched-scoring-check-$job.out"
