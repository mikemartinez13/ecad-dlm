#!/usr/bin/env bash
# Submit from the frozen source snapshot. Resource requests are explicit in sbatch.
set -euo pipefail
MODE="${1:?proxy, cost or report}"
shift
cd "${SLURM_SUBMIT_DIR:?Submit from the source snapshot}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export APPTAINER_CACHEDIR=/scratch/gqs7zr/ai4mat/ecad/apptainer/cache
export APPTAINER_TMPDIR=/scratch/gqs7zr/ai4mat/ecad/apptainer/tmp
PYTHON_BIN=/home/gqs7zr/.conda/envs/ecad-dlm/bin/python
if ! command -v apptainer >/dev/null 2>&1 && type module >/dev/null 2>&1; then
    module load apptainer
fi
case "$MODE" in
    proxy) nvidia-smi; srun "$PYTHON_BIN" -u -m dlm_kv_policy.proxy_recheck run "$@" ;;
    cost) nvidia-smi; srun "$PYTHON_BIN" -u -m dlm_kv_policy.batched_cost_model "$@" ;;
    report) "$PYTHON_BIN" -u -m dlm_kv_policy.proxy_recheck report "$@" ;;
    *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
