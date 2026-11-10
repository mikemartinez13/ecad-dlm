#!/usr/bin/env bash
# EvalPlus extra tests on the programs the constrained search's audit generated. CPU
# only, on the submitting user's default account. Reads and writes
# constrained_kv_search_v1/audit.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SNAP="$REPO/run_snapshots/audit_plus_tests_rivanna_v1"
ARTIFACTS=/scratch/gqs7zr/ai4mat/ecad
OUTPUT="$ARTIFACTS/results/dlm_kv_policy/constrained_kv_search_v1"
LOGS="$ARTIFACTS/logs"
[[ -e "$SNAP/jobs.txt" ]] && { echo "$SNAP/jobs.txt exists; refusing duplicate submission" >&2; exit 1; }
[[ -f "$OUTPUT/audit/audit.json" ]] || { echo "No audit at $OUTPUT/audit" >&2; exit 1; }
mkdir -p "$SNAP" "$LOGS"
rsync -a --delete --exclude=__pycache__ "$REPO/ecad-dlm" "$SNAP/"
# Variables from an interactive allocation must not leak into the job. SLURM_CONF
# stays: inside an allocation, sbatch needs it to find the cluster.
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "$name"; done
unset TMPDIR
cd "$SNAP"  # SLURM_SUBMIT_DIR, which the launcher puts on the import path
job=$(sbatch --parsable --job-name=audit-plus --partition=standard --cpus-per-task=8 --mem=32G --time=12:00:00 \
    --output="$LOGS/slurm-%x-%j.out" "$SNAP/ecad-dlm/experiments/scripts/slurm/run_proxy_cost_prerequisites_rivanna.sh" \
    search-plus --config "$SNAP/ecad-dlm/experiments/configs/diffucoder7b_audit_96tasks.yaml" --output "$OUTPUT")
printf 'plus %s\n' "$job" > "$SNAP/jobs.txt"
cat "$SNAP/jobs.txt"
