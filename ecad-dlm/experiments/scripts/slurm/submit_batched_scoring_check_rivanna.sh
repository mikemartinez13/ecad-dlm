#!/usr/bin/env bash
# Freeze the source (scripts/slurm/freeze_source.sh) and submit the vectorized scoring check.
# Test job: raiselab-paid by default. Resubmitting resumes from the saved tasks.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
NAME=batched_scoring_check_v1
ARTIFACTS=/scratch/gqs7zr/ai4mat/ecad
OUTPUT="$ARTIFACTS/results/dlm_kv_policy/batched_teacher_forcing_check_v1"
LOGS="$ARTIFACTS/logs"
RECORD="$LOGS/runs/$NAME.txt"
ACCOUNT="${ACCOUNT:-raiselab-paid}"
SOURCE="$("$REPO/scripts/slurm/freeze_source.sh" --again "$NAME")"
FROZEN=("$REPO/scripts/slurm/run_frozen.sh" "$REPO" "$SOURCE")
mkdir -p "$LOGS/runs"
echo "source $SOURCE" >> "$RECORD"
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "$name"; done
unset TMPDIR
cd "$REPO"
job=$(OUTPUT="$OUTPUT" sbatch --parsable --export=ALL --job-name=batched-scoring-check \
    --account="$ACCOUNT" --partition=gpu --constraint=a100_80gb --gres=gpu:1 --cpus-per-task=8 --mem=96G \
    --time="${TIME:-08:00:00}" --output="$LOGS/slurm-%x-%j.out" \
    "${FROZEN[@]}" bash ecad-dlm/experiments/scripts/slurm/run_batched_scoring_check_rivanna.sh)
echo "check $job" | tee -a "$RECORD"
echo "Log: $LOGS/slurm-batched-scoring-check-$job.out"
