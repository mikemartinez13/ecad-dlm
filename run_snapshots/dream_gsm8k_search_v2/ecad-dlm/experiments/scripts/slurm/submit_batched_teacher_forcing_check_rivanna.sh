#!/usr/bin/env bash
# Freeze the source and submit the batched teacher forcing check
# (experiments/BATCHED_TEACHER_FORCING_CHECK.md). Test job: raiselab-paid by default.
#   submit_batched_teacher_forcing_check_rivanna.sh            # RUN=v1, BATCH=16
# Resubmitting the same RUN resumes from the saved tasks.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
RUN="${RUN:-v1}"
SNAP="$REPO/run_snapshots/batched_teacher_forcing_check_$RUN"
ARTIFACTS=/scratch/gqs7zr/ai4mat/ecad
OUTPUT="$ARTIFACTS/results/dlm_kv_policy/batched_teacher_forcing_check_$RUN"
LOGS="$ARTIFACTS/logs"
ACCOUNT="${ACCOUNT:-raiselab-paid}"
mkdir -p "$SNAP" "$LOGS" "$OUTPUT"
rsync -a --delete --exclude=__pycache__ "$REPO/ecad-dlm" "$SNAP/"
# Variables from an interactive allocation must not leak into the job; sbatch needs SLURM_CONF.
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "$name"; done
unset TMPDIR
cd "$SNAP"
job=$(OUTPUT="$OUTPUT" BATCH="${BATCH:-16}" sbatch --parsable --export=ALL --job-name="batched-tf-check-$RUN" \
    --account="$ACCOUNT" --partition=gpu --constraint=a100_80gb --gres=gpu:1 --cpus-per-task=8 --mem=96G \
    --time="${TIME:-12:00:00}" --output="$LOGS/slurm-%x-%j.out" \
    "$SNAP/ecad-dlm/experiments/scripts/slurm/run_batched_teacher_forcing_check_rivanna.sh")
echo "check $job" | tee -a "$SNAP/jobs.txt"
echo "Log: $LOGS/slurm-batched-tf-check-$RUN-$job.out"
