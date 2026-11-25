#!/usr/bin/env bash
# One constrained search on one A100 80GB. Each generation's teacher forcing runs as PROCS
# processes (default 1; batched teacher forcing fills the GPU on its own), and breeding runs
# in between, in the same allocation.
# Finished tasks are saved, so near the time limit (Slurm sends USR1) the job stops its
# processes and resubmits itself, and the new job resumes. After the last generation
# it submits the audit and the EvalPlus extra tests as their own jobs.
# Started by submit_search_single_gpu_rivanna.sh through scripts/slurm/run_frozen.sh, which
# runs it in an unpacked copy of the run's commit. The environment carries CONFIG, OUTPUT,
# AUDIT_CONFIG, AUDIT_TOP, PROCS, ACCOUNT, TIME, JOB_NAME, LOGS, RECORD, ECAD_REPO and
# ECAD_SOURCE_COMMIT; every job it submits runs the same commit.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Start through scripts/slurm/run_frozen.sh}"
export PYTHONPATH="$PWD/ecad-dlm:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON_BIN=/home/gqs7zr/.conda/envs/ecad-dlm/bin/python
SEARCH=(-u -m evolution.constrained_kv_policy_search)
PATHS=(--config "$CONFIG" --output "$OUTPUT")
STEPS=("$PWD/scripts/slurm/run_frozen.sh" "$ECAD_REPO" "$ECAD_SOURCE_COMMIT"
       bash ecad-dlm/experiments/scripts/slurm/run_proxy_cost_prerequisites_rivanna.sh)
children=()

resubmit() {
    echo "Time limit near: stopping and resubmitting" >&2
    kill "${children[@]}" 2>/dev/null || true
    wait || true
    bash "$PWD/ecad-dlm/experiments/scripts/slurm/sbatch_search_single_gpu_rivanna.sh"
    exit 0
}
trap resubmit USR1

nvidia-smi
"$PYTHON_BIN" "${SEARCH[@]}" dense "${PATHS[@]}"
while :; do
    "$PYTHON_BIN" "${SEARCH[@]}" advance "${PATHS[@]}"
    latest=$(find "$OUTPUT" -maxdepth 1 -name 'generation_*' -printf '%f\n' | sort | tail -1)
    [[ -e "$OUTPUT/$latest/summary.json" ]] && break
    generation=$((10#${latest#generation_}))
    children=()
    for part in $(seq 0 $((PROCS - 1))); do
        "$PYTHON_BIN" "${SEARCH[@]}" evaluate "${PATHS[@]}" --generation "$generation" --part "$part" --parts "$PROCS" &
        children+=($!)
    done
    failed=0
    for pid in "${children[@]}"; do wait "$pid" || failed=1; done
    children=()
    [[ "$failed" == 0 ]] || { echo "Generation $generation: a teacher_forcing process failed" >&2; exit 1; }
done
echo "Search finished: $latest"
# Variables from this job must not leak into the next ones. SLURM_CONF stays: sbatch needs it.
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "$name"; done
unset TMPDIR PYTHONPATH
cd "$ECAD_REPO"  # their working directory; this job's unpacked copy is removed when it ends
audit=$(sbatch --parsable --job-name=search-audit --account="$ACCOUNT" --partition=gpu --constraint=a100_80gb \
    --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=12:00:00 --output="$LOGS/slurm-%x-%j.out" \
    "${STEPS[@]}" search-audit --config "$AUDIT_CONFIG" --output "$OUTPUT" --top "$AUDIT_TOP")
plus=$(sbatch --parsable --job-name=search-plus --account="$ACCOUNT" --partition=standard --cpus-per-task=8 \
    --mem=32G --time=12:00:00 --dependency="afterok:$audit" --output="$LOGS/slurm-%x-%j.out" \
    "${STEPS[@]}" search-plus --config "$AUDIT_CONFIG" --output "$OUTPUT")
printf 'audit %s\nplus %s\n' "$audit" "$plus" | tee -a "$RECORD"
