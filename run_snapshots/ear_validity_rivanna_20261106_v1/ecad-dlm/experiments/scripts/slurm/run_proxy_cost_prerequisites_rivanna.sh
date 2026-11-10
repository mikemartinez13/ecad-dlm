#!/usr/bin/env bash
# Rivanna paths. Submit from the frozen source snapshot; resources are explicit in sbatch.
set -euo pipefail
MODE="${1:?proxy, cost, report, ear or ear-report}"
shift
cd "${SLURM_SUBMIT_DIR:?Submit from the source snapshot}"
export PYTHONPATH="$PWD/ecad-dlm:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export APPTAINER_CACHEDIR=/scratch/gqs7zr/ai4mat/ecad/apptainer/cache
export APPTAINER_TMPDIR=/scratch/gqs7zr/ai4mat/ecad/apptainer/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
PYTHON_BIN=/home/gqs7zr/.conda/envs/ecad-dlm/bin/python
# Array tasks take their shard from Slurm; single jobs pass --part themselves.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    set -- "$@" --part "$SLURM_ARRAY_TASK_ID"
fi
case "$MODE" in
    proxy) nvidia-smi; srun "$PYTHON_BIN" -u -m experiments.proxy_recheck run "$@" ;;
    cost) nvidia-smi; srun "$PYTHON_BIN" -u -m experiments.batched_cost_model "$@" ;;
    ear) nvidia-smi; srun "$PYTHON_BIN" -u -m experiments.ear_validity run "$@" ;;
    report|ear-report)
        if ! command -v apptainer >/dev/null 2>&1; then
            set +u; source /etc/profile.d/modules.sh; module load apptainer; set -u
        fi
        if [[ "$MODE" == report ]]; then
            "$PYTHON_BIN" -u -m experiments.proxy_recheck report "$@"
        else
            "$PYTHON_BIN" -u -m experiments.ear_validity report "$@"
        fi ;;
    *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
