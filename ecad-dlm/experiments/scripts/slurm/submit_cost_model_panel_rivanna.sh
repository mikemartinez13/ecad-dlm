#!/usr/bin/env bash
# Freeze the source, then chain: time the tau panel at batch 8 on the A100 80GB -> report.
# The report ranks candidate search costs against those timings, using the teacher forcing
# records of the finished tau confirmation run.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
VERSION="${VERSION:-v1}"
SNAP="$REPO/run_snapshots/cost_model_panel_rivanna_$VERSION"
ARTIFACTS=/scratch/gqs7zr/ai4mat/ecad
OUTPUT="$ARTIFACTS/results/dlm_kv_policy/cost_model_panel_$VERSION"
TEACHER_FORCING="$ARTIFACTS/results/dlm_kv_policy/tau_confirmation_v1"
LOGS="$ARTIFACTS/logs"
ACCOUNT="${ACCOUNT:-raiselab-paid}"
[[ -e "$SNAP/jobs.txt" ]] && { echo "$SNAP/jobs.txt exists; refusing duplicate submission" >&2; exit 1; }
mkdir -p "$SNAP" "$LOGS" "$OUTPUT"
rsync -a --delete --exclude=__pycache__ "$REPO/ecad-dlm" "$SNAP/"
# Variables from an interactive allocation must not leak into these jobs. SLURM_CONF
# stays: inside an allocation, sbatch needs it to find the cluster.
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "$name"; done
unset TMPDIR
cd "$SNAP"  # SLURM_SUBMIT_DIR, which the launcher puts on the import path
SCRIPT="$SNAP/ecad-dlm/experiments/scripts/slurm/run_proxy_cost_prerequisites_rivanna.sh"
CONFIG="$SNAP/ecad-dlm/experiments/configs/diffucoder7b_tau_confirmation_96tasks.yaml"
SEARCH="$SNAP/ecad-dlm/experiments/configs/diffucoder7b_constrained_kv_search_v2_96tasks.yaml"
PATHS=(--config "$CONFIG" --output "$OUTPUT")
timing=$(sbatch --parsable --job-name=cost-panel-time --account="$ACCOUNT" --partition=gpu --constraint=a100_80gb \
    --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=03:00:00 --output="$LOGS/slurm-%x-%j.out" \
    "$SCRIPT" cost-panel-time "${PATHS[@]}" "$@")
report=$(sbatch --parsable --job-name=cost-panel-report --account="$ACCOUNT" --partition=standard --cpus-per-task=2 --mem=16G \
    --time=00:30:00 --dependency="afterok:$timing" --output="$LOGS/slurm-%x-%j.out" \
    "$SCRIPT" cost-panel-report "${PATHS[@]}" --search-config "$SEARCH" --teacher-forcing "$TEACHER_FORCING")
printf 'time %s\nreport %s\n' "$timing" "$report" | tee "$SNAP/jobs.txt"
