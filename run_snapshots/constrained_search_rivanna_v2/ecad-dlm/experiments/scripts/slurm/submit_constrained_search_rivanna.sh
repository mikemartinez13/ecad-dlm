#!/usr/bin/env bash
# Freeze the source, then chain: dense reference -> every generation: advance (CPU)
# -> 16 replay shards (A100 80GB) -> ... -> final advance -> audit on the audit set
# -> EvalPlus extra tests on the audited programs (CPU). A batched-backend check of
# generation 0 runs alongside; nothing waits on it.
# RUN names the snapshot and output (v1, v2, ...); CONFIG is the search config under
# experiments/configs, and GENERATIONS is read from it. Run only after the tau
# confirmation report says LAUNCH: True, or with OVERRIDE_TAU_GATE=1.
# A shard job that fails stops the chain; finished tasks are kept, so resubmitting
# the rest of the chain by hand resumes it. AUDIT_TOP sets the audited policies.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
RUN="${RUN:-v1}"
CONFIG_NAME="${CONFIG:-diffucoder7b_constrained_kv_search_96tasks.yaml}"
SNAP="$REPO/run_snapshots/constrained_search_rivanna_$RUN"
ARTIFACTS=/scratch/gqs7zr/ai4mat/ecad
OUTPUT="$ARTIFACTS/results/dlm_kv_policy/constrained_kv_search_$RUN"
LOGS="$ARTIFACTS/logs"
ACCOUNT="${ACCOUNT:-raiselab-paid}"
GENERATIONS="$(awk '$1 == "generations:" {print $2}' "$REPO/ecad-dlm/experiments/configs/$CONFIG_NAME")"
[[ -n "$GENERATIONS" ]] || { echo "No generations in $CONFIG_NAME" >&2; exit 1; }
AUDIT_TOP="${AUDIT_TOP:-4}"
PARTS=16
CONFIRMATION="$ARTIFACTS/results/dlm_kv_policy/tau_confirmation_v1/tau_confirmation_report.json"
[[ -e "$SNAP/jobs.txt" ]] && { echo "$SNAP/jobs.txt exists; refusing duplicate submission" >&2; exit 1; }
# Tau failed confirmation; OVERRIDE_TAU_GATE=1 records Michael's 2026-11-08 decision (CONSTRAINED_SEARCH.md).
[[ "${OVERRIDE_TAU_GATE:-0}" == 1 ]] || grep -q '"launch": true' "$CONFIRMATION" || { echo "$CONFIRMATION does not approve a launch" >&2; exit 1; }
mkdir -p "$SNAP" "$LOGS" "$OUTPUT"
rsync -a --delete --exclude=__pycache__ "$REPO/ecad-dlm" "$SNAP/"
# Variables from an interactive allocation must not leak into these jobs. SLURM_CONF
# stays: inside an allocation, sbatch needs it to find the cluster.
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "$name"; done
unset TMPDIR
cd "$SNAP"  # SLURM_SUBMIT_DIR, which the launcher puts on the import path
SCRIPT="$SNAP/ecad-dlm/experiments/scripts/slurm/run_proxy_cost_prerequisites_rivanna.sh"
CONFIG="$SNAP/ecad-dlm/experiments/configs/$CONFIG_NAME"
AUDIT_CONFIG="$SNAP/ecad-dlm/experiments/configs/diffucoder7b_audit_96tasks.yaml"
PATHS=(--config "$CONFIG" --output "$OUTPUT")
GPU=(--account="$ACCOUNT" --partition=gpu --constraint=a100_80gb --gres=gpu:1 --cpus-per-task=8 --mem=96G)
CPU=(--account="$ACCOUNT" --partition=standard --cpus-per-task=2 --mem=16G --time=00:30:00)
dense=$(sbatch --parsable --job-name=search-dense "${GPU[@]}" --time=02:00:00 \
    --output="$LOGS/slurm-%x-%j.out" "$SCRIPT" search-dense "${PATHS[@]}")
dependency=(--dependency="afterok:$dense")
printf 'dense %s\n' "$dense" > "$SNAP/jobs.tmp"
for generation in $(seq 0 $((GENERATIONS - 1))); do
    advance=$(sbatch --parsable --job-name="search-advance-$generation" "${CPU[@]}" "${dependency[@]}" \
        --output="$LOGS/slurm-%x-%j.out" "$SCRIPT" search-advance "${PATHS[@]}")
    shards=$(sbatch --parsable --job-name="search-eval-$generation" "${GPU[@]}" --time=04:00:00 --array=0-$((PARTS - 1)) \
        --dependency="afterok:$advance" --output="$LOGS/slurm-%x-%A_%a.out" \
        "$SCRIPT" search-evaluate "${PATHS[@]}" --generation "$generation" --parts "$PARTS")
    dependency=(--dependency="afterok:$shards")
    printf 'generation %s advance %s shards %s\n' "$generation" "$advance" "$shards" >> "$SNAP/jobs.tmp"
    if [[ "$generation" == 0 ]]; then
        check=$(sbatch --parsable --job-name=search-check "${GPU[@]}" --time=02:00:00 --dependency="afterok:$advance" \
            --output="$LOGS/slurm-%x-%j.out" "$SCRIPT" search-check "${PATHS[@]}" --generation 0)
        printf 'check %s\n' "$check" >> "$SNAP/jobs.tmp"
    fi
done
final=$(sbatch --parsable --job-name=search-advance-final "${CPU[@]}" "${dependency[@]}" \
    --output="$LOGS/slurm-%x-%j.out" "$SCRIPT" search-advance "${PATHS[@]}")
audit=$(sbatch --parsable --job-name=search-audit "${GPU[@]}" --time=12:00:00 --dependency="afterok:$final" \
    --output="$LOGS/slurm-%x-%j.out" "$SCRIPT" search-audit --config "$AUDIT_CONFIG" --output "$OUTPUT" --top "$AUDIT_TOP")
plus=$(sbatch --parsable --job-name=search-plus --account="$ACCOUNT" --partition=standard --cpus-per-task=8 --mem=32G --time=12:00:00 \
    --dependency="afterok:$audit" --output="$LOGS/slurm-%x-%j.out" "$SCRIPT" search-plus --config "$AUDIT_CONFIG" --output "$OUTPUT")
printf 'final %s\naudit %s\nplus %s\n' "$final" "$audit" "$plus" >> "$SNAP/jobs.tmp"
mv "$SNAP/jobs.tmp" "$SNAP/jobs.txt"
cat "$SNAP/jobs.txt"
