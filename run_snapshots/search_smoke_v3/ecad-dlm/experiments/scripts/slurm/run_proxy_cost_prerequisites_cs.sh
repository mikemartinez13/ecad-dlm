#!/usr/bin/env bash
# CS cluster paths. Submit from the frozen source snapshot; resources are explicit in sbatch.
set -euo pipefail
MODE="${1:?proxy, cost or report}"
shift
cd "${SLURM_SUBMIT_DIR:?Submit from the source snapshot}"
export PYTHONPATH="$PWD/ecad-dlm:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export APPTAINER_CACHEDIR=/bigtemp/gqs7zr/ecad-dlm/apptainer/cache
export APPTAINER_TMPDIR=/bigtemp/gqs7zr/ecad-dlm/apptainer/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
PYTHON_BIN=/u/gqs7zr/.conda/envs/ecad-dlm/bin/python
# Array tasks take their shard from Slurm; single jobs pass --part themselves.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    set -- "$@" --part "$SLURM_ARRAY_TASK_ID"
fi
case "$MODE" in
    proxy) nvidia-smi; srun "$PYTHON_BIN" -u -m experiments.proxy_recheck run "$@" ;;
    cost) nvidia-smi; srun "$PYTHON_BIN" -u -m experiments.batched_cost_model "$@" ;;
    report)
        if ! command -v apptainer >/dev/null 2>&1; then
            set +u; source /etc/profile.d/modules.sh; module load apptainer; set -u
        fi
        "$PYTHON_BIN" -u -m experiments.proxy_recheck report "$@" ;;
    *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
