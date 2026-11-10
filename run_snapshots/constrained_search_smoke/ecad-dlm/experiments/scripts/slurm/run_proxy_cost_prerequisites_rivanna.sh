#!/usr/bin/env bash
# Rivanna paths. Submit from the frozen source snapshot; resources are explicit in sbatch.
set -euo pipefail
MODE="${1:?proxy, cost, report, ear, ear-report, search-advance, search-evaluate, search-check or search-audit}"
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
    search-advance) "$PYTHON_BIN" -u -m evolution.constrained_kv_policy_search advance "$@" ;;
    search-evaluate) nvidia-smi; srun "$PYTHON_BIN" -u -m evolution.constrained_kv_policy_search evaluate "$@" ;;
    search-check) nvidia-smi; srun "$PYTHON_BIN" -u -m evolution.constrained_kv_policy_search check "$@" ;;
    report|ear-report|search-audit)
        if ! command -v apptainer >/dev/null 2>&1; then
            set +u; source /etc/profile.d/modules.sh; module load apptainer; set -u
        fi
        case "$MODE" in
            report) "$PYTHON_BIN" -u -m experiments.proxy_recheck report "$@" ;;
            ear-report) "$PYTHON_BIN" -u -m experiments.ear_validity report "$@" ;;
            search-audit) nvidia-smi; srun "$PYTHON_BIN" -u -m evolution.constrained_kv_policy_search audit "$@" ;;
        esac ;;
    *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
