#!/usr/bin/env bash
# EvalPlus extra tests on the programs the constrained search's audit generated. CPU
# only, on the submitting user's default account. Reads and writes
# constrained_kv_search_v1/audit. Freezes the source first (scripts/slurm/freeze_source.sh).
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
NAME=audit_plus_tests_rivanna_v1
ARTIFACTS=/scratch/gqs7zr/ai4mat/ecad
OUTPUT="$ARTIFACTS/results/dlm_kv_policy/constrained_kv_search_v1"
LOGS="$ARTIFACTS/logs"
RECORD="$LOGS/runs/$NAME.txt"
[[ -f "$OUTPUT/audit/audit.json" ]] || { echo "No audit at $OUTPUT/audit" >&2; exit 1; }
SOURCE="$("$REPO/scripts/slurm/freeze_source.sh" "$NAME")"
FROZEN=("$REPO/scripts/slurm/run_frozen.sh" "$REPO" "$SOURCE")
mkdir -p "$LOGS/runs"
echo "source $SOURCE" > "$RECORD"
# Variables from an interactive allocation must not leak into the job. SLURM_CONF
# stays: inside an allocation, sbatch needs it to find the cluster.
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "$name"; done
unset TMPDIR
cd "$REPO"
# Paths inside the source are relative: the job runs in its unpacked copy of the commit.
job=$(sbatch --parsable --job-name=audit-plus --partition=standard --cpus-per-task=8 --mem=32G --time=12:00:00 \
    --output="$LOGS/slurm-%x-%j.out" "${FROZEN[@]}" bash ecad-dlm/experiments/scripts/slurm/run_proxy_cost_prerequisites_rivanna.sh \
    search-plus --config ecad-dlm/experiments/configs/diffucoder7b_audit_96tasks.yaml --output "$OUTPUT")
printf 'plus %s\n' "$job" >> "$RECORD"
cat "$RECORD"
