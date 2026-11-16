#!/usr/bin/env bash
# Freeze the source, then chain: one-task pilot -> 16 shards -> report. All GPU jobs
# use the A100 80GB; ear_validity.py refuses any other GPU, because every replay must
# reproduce the proxy recheck exactly. The pilot's task is reused by its shard.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SNAP="$REPO/run_snapshots/ear_validity_rivanna_20261106_v1"
ARTIFACTS=/scratch/gqs7zr/ai4mat/ecad
PROXY="$ARTIFACTS/results/dlm_kv_policy/proxy_cost_20261104_v1/proxy"
OUTPUT="$ARTIFACTS/results/dlm_kv_policy/ear_validity_20261106_v1"
LOGS="$ARTIFACTS/logs"
ACCOUNT="${ACCOUNT:-raiselab-paid}"
PARTS=16
[[ -e "$SNAP/jobs.txt" ]] && { echo "$SNAP/jobs.txt exists; refusing duplicate submission" >&2; exit 1; }
[[ -f "$PROXY/proxy_report.json" ]] || { echo "No proxy recheck results in $PROXY" >&2; exit 1; }
mkdir -p "$SNAP" "$LOGS" "$OUTPUT"
rsync -a --delete --exclude=__pycache__ "$REPO/ecad-dlm" "$SNAP/"
# Variables from an interactive allocation must not leak into these jobs. SLURM_CONF
# stays: inside an allocation, sbatch needs it to find the cluster.
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "$name"; done
unset TMPDIR
cd "$SNAP"  # SLURM_SUBMIT_DIR, which the launcher puts on the import path
SCRIPT="$SNAP/ecad-dlm/experiments/scripts/slurm/run_proxy_cost_prerequisites_rivanna.sh"
CONFIG="$SNAP/ecad-dlm/experiments/configs/diffucoder7b_proxy_recheck_96.yaml"
PATHS=(--config "$CONFIG" --proxy-output "$PROXY" --output "$OUTPUT")
GPU=(--account="$ACCOUNT" --partition=gpu --constraint=a100_80gb --gres=gpu:1 --cpus-per-task=8 --mem=96G)
pilot=$(sbatch --parsable --job-name=ear96-pilot "${GPU[@]}" --time=02:00:00 \
    --output="$LOGS/slurm-%x-%j.out" "$SCRIPT" ear "${PATHS[@]}" --part 0 --parts 96)
shards=$(sbatch --parsable --job-name=ear96-shard "${GPU[@]}" --time=06:00:00 --array=0-$((PARTS - 1)) \
    --dependency="afterok:$pilot" --output="$LOGS/slurm-%x-%A_%a.out" "$SCRIPT" ear "${PATHS[@]}" --parts "$PARTS")
report=$(sbatch --parsable --job-name=ear96-report --account="$ACCOUNT" --partition=standard --cpus-per-task=8 --mem=32G \
    --time=02:00:00 --dependency="afterok:$shards" --output="$LOGS/slurm-%x-%j.out" "$SCRIPT" ear-report "${PATHS[@]}")
printf 'pilot %s\nshards %s\nreport %s\n' "$pilot" "$shards" "$report" | tee "$SNAP/jobs.txt"
