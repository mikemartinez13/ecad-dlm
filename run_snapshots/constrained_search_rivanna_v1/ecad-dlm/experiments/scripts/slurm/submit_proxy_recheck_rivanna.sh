#!/usr/bin/env bash
# Freeze the source, then chain: one-task pilot -> 16 shards -> scoring report, with
# the cost study alongside and a second report that adds its projections. Scoring
# does not depend on the cost study, and the second report reuses the cached scores.
# All GPU jobs use the A100 80GB; proxy_recheck.py and batched_cost_model.py refuse
# any other GPU, so the 96 tasks cannot mix GPU types. The pilot's task is reused by
# its shard.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SNAP="$REPO/run_snapshots/proxy_recheck_rivanna_20261106_v1"
ARTIFACTS=/scratch/gqs7zr/ai4mat/ecad
OUTPUT="$ARTIFACTS/results/dlm_kv_policy/proxy_cost_20261104_v1"
PROXY="$OUTPUT/proxy"
COST="$OUTPUT/cost"
LOGS="$ARTIFACTS/logs"
ACCOUNT="${ACCOUNT:-raiselab-paid}"
PARTS=16
[[ -e "$SNAP/jobs.txt" ]] && { echo "$SNAP/jobs.txt exists; refusing duplicate submission" >&2; exit 1; }
mkdir -p "$SNAP" "$LOGS" "$PROXY"
rsync -a --delete --exclude=__pycache__ "$REPO/ecad-dlm" "$SNAP/"
# Variables from an interactive allocation must not leak into these jobs. SLURM_CONF
# stays: inside an allocation, sbatch needs it to find the cluster.
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "$name"; done
unset TMPDIR
cd "$SNAP"  # SLURM_SUBMIT_DIR, which the launcher puts on the import path
SCRIPT="$SNAP/ecad-dlm/experiments/scripts/slurm/run_proxy_cost_prerequisites_rivanna.sh"
CONFIG="$SNAP/ecad-dlm/experiments/configs/diffucoder7b_proxy_recheck_96.yaml"
GPU=(--account="$ACCOUNT" --partition=gpu --constraint=a100_80gb --gres=gpu:1 --cpus-per-task=8 --mem=96G)
CPU=(--account="$ACCOUNT" --partition=standard --cpus-per-task=8 --mem=32G)
pilot=$(sbatch --parsable --job-name=proxy96-pilot "${GPU[@]}" --time=02:00:00 \
    --output="$LOGS/slurm-%x-%j.out" "$SCRIPT" proxy --config "$CONFIG" --output "$PROXY" --part 0 --parts 96)
shards=$(sbatch --parsable --job-name=proxy96-shard "${GPU[@]}" --time=08:00:00 --array=0-$((PARTS - 1)) \
    --dependency="afterok:$pilot" --output="$LOGS/slurm-%x-%A_%a.out" \
    "$SCRIPT" proxy --config "$CONFIG" --output "$PROXY" --parts "$PARTS")
cost=$(sbatch --parsable --job-name=cache-cost "${GPU[@]}" --time=04:00:00 \
    --output="$LOGS/slurm-%x-%j.out" "$SCRIPT" cost --config "$CONFIG" --output "$COST")
report=$(sbatch --parsable --job-name=proxy96-score "${CPU[@]}" --time=04:00:00 \
    --dependency="afterok:$shards" --output="$LOGS/slurm-%x-%j.out" "$SCRIPT" report --config "$CONFIG" --output "$PROXY")
cost_report=$(sbatch --parsable --job-name=proxy96-cost-report "${CPU[@]}" --time=01:00:00 \
    --dependency="afterok:$report:$cost" --output="$LOGS/slurm-%x-%j.out" \
    "$SCRIPT" report --config "$CONFIG" --output "$PROXY" --cost-table "$COST/table.json")
printf 'pilot %s\nshards %s\ncost %s\nreport %s\ncost_report %s\n' "$pilot" "$shards" "$cost" "$report" "$cost_report" \
    | tee "$SNAP/jobs.txt"
