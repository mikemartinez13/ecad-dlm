#!/usr/bin/env bash
# Freeze the source, then chain every generation: advance (CPU) -> 16 replay shards
# (A100 80GB) -> ... -> final advance -> audit. GENERATIONS must match the config.
# A shard job that fails stops the chain; finished tasks are kept, so resubmitting
# the rest of the chain by hand resumes it. AUDIT_TOP sets the audited policies.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SNAP="$REPO/run_snapshots/constrained_search_rivanna_v1"
ARTIFACTS=/scratch/gqs7zr/ai4mat/ecad
OUTPUT="$ARTIFACTS/results/dlm_kv_policy/constrained_kv_search_v1"
LOGS="$ARTIFACTS/logs"
ACCOUNT="${ACCOUNT:-raiselab-paid}"
GENERATIONS="${GENERATIONS:-8}"
AUDIT_TOP="${AUDIT_TOP:-4}"
PARTS=16
[[ -e "$SNAP/jobs.txt" ]] && { echo "$SNAP/jobs.txt exists; refusing duplicate submission" >&2; exit 1; }
mkdir -p "$SNAP" "$LOGS" "$OUTPUT"
rsync -a --delete --exclude=__pycache__ "$REPO/ecad-dlm" "$SNAP/"
# Variables from an interactive allocation must not leak into these jobs. SLURM_CONF
# stays: inside an allocation, sbatch needs it to find the cluster.
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "$name"; done
unset TMPDIR
cd "$SNAP"  # SLURM_SUBMIT_DIR, which the launcher puts on the import path
SCRIPT="$SNAP/ecad-dlm/experiments/scripts/slurm/run_proxy_cost_prerequisites_rivanna.sh"
CONFIG="$SNAP/ecad-dlm/experiments/configs/diffucoder7b_constrained_kv_search_96tasks.yaml"
PATHS=(--config "$CONFIG" --output "$OUTPUT")
GPU=(--account="$ACCOUNT" --partition=gpu --constraint=a100_80gb --gres=gpu:1 --cpus-per-task=8 --mem=96G)
CPU=(--account="$ACCOUNT" --partition=standard --cpus-per-task=2 --mem=16G --time=00:30:00)
dependency=()
: > "$SNAP/jobs.tmp"
for generation in $(seq 0 $((GENERATIONS - 1))); do
    advance=$(sbatch --parsable --job-name="search-advance-$generation" "${CPU[@]}" "${dependency[@]}" \
        --output="$LOGS/slurm-%x-%j.out" "$SCRIPT" search-advance "${PATHS[@]}")
    shards=$(sbatch --parsable --job-name="search-eval-$generation" "${GPU[@]}" --time=04:00:00 --array=0-$((PARTS - 1)) \
        --dependency="afterok:$advance" --output="$LOGS/slurm-%x-%A_%a.out" \
        "$SCRIPT" search-evaluate "${PATHS[@]}" --generation "$generation" --parts "$PARTS")
    dependency=(--dependency="afterok:$shards")
    printf 'generation %s advance %s shards %s\n' "$generation" "$advance" "$shards" >> "$SNAP/jobs.tmp"
done
final=$(sbatch --parsable --job-name=search-advance-final "${CPU[@]}" "${dependency[@]}" \
    --output="$LOGS/slurm-%x-%j.out" "$SCRIPT" search-advance "${PATHS[@]}")
audit=$(sbatch --parsable --job-name=search-audit "${GPU[@]}" --time=08:00:00 --dependency="afterok:$final" \
    --output="$LOGS/slurm-%x-%j.out" "$SCRIPT" search-audit "${PATHS[@]}" --top "$AUDIT_TOP")
printf 'final %s\naudit %s\n' "$final" "$audit" >> "$SNAP/jobs.tmp"
mv "$SNAP/jobs.tmp" "$SNAP/jobs.txt"
cat "$SNAP/jobs.txt"
