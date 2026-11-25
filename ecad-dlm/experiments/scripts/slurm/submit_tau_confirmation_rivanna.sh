#!/usr/bin/env bash
# Freeze the source (scripts/slurm/freeze_source.sh), then chain: one-task pilot -> 16
# shards -> report, with real batched timing alongside. All GPU jobs use the A100 80GB. The
# pilot's task is reused by its shard.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
NAME=tau_confirmation_rivanna_v1
ARTIFACTS=/scratch/gqs7zr/ai4mat/ecad
OUTPUT="$ARTIFACTS/results/dlm_kv_policy/tau_confirmation_v1"
LOGS="$ARTIFACTS/logs"
RECORD="$LOGS/runs/$NAME.txt"
ACCOUNT="${ACCOUNT:-raiselab-paid}"
PARTS=16
SOURCE="$("$REPO/scripts/slurm/freeze_source.sh" "$NAME")"
FROZEN=("$REPO/scripts/slurm/run_frozen.sh" "$REPO" "$SOURCE")
mkdir -p "$LOGS/runs" "$OUTPUT"
# Variables from an interactive allocation must not leak into these jobs. SLURM_CONF
# stays: inside an allocation, sbatch needs it to find the cluster.
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "$name"; done
unset TMPDIR
cd "$REPO"
# Paths inside the source are relative: each job runs in its own unpacked copy of the commit.
SCRIPT=("${FROZEN[@]}" bash ecad-dlm/experiments/scripts/slurm/run_proxy_cost_prerequisites_rivanna.sh)
CONFIG=ecad-dlm/experiments/configs/diffucoder7b_tau_confirmation_96tasks.yaml
SEARCH=ecad-dlm/experiments/configs/diffucoder7b_constrained_kv_search_96tasks.yaml
PATHS=(--config "$CONFIG" --output "$OUTPUT")
GPU=(--account="$ACCOUNT" --partition=gpu --constraint=a100_80gb --gres=gpu:1 --cpus-per-task=8 --mem=96G)
pilot=$(sbatch --parsable --job-name=tau96-pilot "${GPU[@]}" --time=02:00:00 \
    --output="$LOGS/slurm-%x-%j.out" "${SCRIPT[@]}" tau "${PATHS[@]}" --part 0 --parts 96)
shards=$(sbatch --parsable --job-name=tau96-shard "${GPU[@]}" --time=06:00:00 --array=0-$((PARTS - 1)) \
    --dependency="afterok:$pilot" --output="$LOGS/slurm-%x-%A_%a.out" "${SCRIPT[@]}" tau "${PATHS[@]}" --parts "$PARTS")
timing=$(sbatch --parsable --job-name=tau96-time "${GPU[@]}" --time=06:00:00 --dependency="afterok:$pilot" \
    --output="$LOGS/slurm-%x-%j.out" "${SCRIPT[@]}" tau-time "${PATHS[@]}")
report=$(sbatch --parsable --job-name=tau96-report --account="$ACCOUNT" --partition=standard --cpus-per-task=8 --mem=32G \
    --time=04:00:00 --dependency="afterok:$shards:$timing" --output="$LOGS/slurm-%x-%j.out" \
    "${SCRIPT[@]}" tau-report "${PATHS[@]}" --search-config "$SEARCH")
printf 'source %s\npilot %s\nshards %s\ntime %s\nreport %s\n' "$SOURCE" "$pilot" "$shards" "$timing" "$report" | tee "$RECORD"
