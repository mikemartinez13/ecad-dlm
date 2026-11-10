#!/usr/bin/env bash
# Freeze the source, then chain: one-task pilot -> 16 shards -> scoring report.
# All GPU jobs use the A100 80GB; proxy_recheck.py refuses any other GPU, so the
# 96 tasks cannot mix GPU types. The pilot's task is reused by its shard.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SNAP="$REPO/run_snapshots/proxy_recheck_cs_20261105_v1"
ARTIFACTS=/bigtemp/gqs7zr/ecad-dlm
OUTPUT="$ARTIFACTS/results/dlm_kv_policy/proxy_cost_20261104_v1/proxy"
LOGS="$ARTIFACTS/logs"
PARTS=16
[[ -e "$SNAP/jobs.txt" ]] && { echo "$SNAP/jobs.txt exists; refusing duplicate submission" >&2; exit 1; }
mkdir -p "$SNAP" "$LOGS" "$OUTPUT"
rsync -a --delete --exclude=__pycache__ "$REPO/ecad-dlm" "$SNAP/"
# Slurm variables from an interactive allocation must not leak into these jobs.
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' || true); do unset "$name"; done
cd "$SNAP"  # SLURM_SUBMIT_DIR, which the launcher puts on the import path
SCRIPT="$SNAP/ecad-dlm/experiments/scripts/slurm/run_proxy_cost_prerequisites_cs.sh"
CONFIG="$SNAP/ecad-dlm/experiments/configs/diffucoder7b_proxy_recheck_96_cs.yaml"
GPU=(--partition=gpu --constraint=a100_80gb --gres=gpu:1 --cpus-per-task=8 --mem=96G)
pilot=$(sbatch --parsable --job-name=proxy96-pilot "${GPU[@]}" --time=02:00:00 \
    --output="$LOGS/%x-%j.out" "$SCRIPT" proxy --config "$CONFIG" --output "$OUTPUT" --part 0 --parts 96)
shards=$(sbatch --parsable --job-name=proxy96-shard "${GPU[@]}" --time=08:00:00 --array=0-$((PARTS - 1)) \
    --dependency="afterok:$pilot" --output="$LOGS/%x-%A_%a.out" \
    "$SCRIPT" proxy --config "$CONFIG" --output "$OUTPUT" --parts "$PARTS")
report=$(sbatch --parsable --job-name=proxy96-score --partition=cpu --cpus-per-task=8 --mem=32G --time=04:00:00 \
    --dependency="afterok:$shards" --output="$LOGS/%x-%j.out" "$SCRIPT" report --config "$CONFIG" --output "$OUTPUT")
printf 'pilot %s\nshards %s\nreport %s\n' "$pilot" "$shards" "$report" | tee "$SNAP/jobs.txt"
