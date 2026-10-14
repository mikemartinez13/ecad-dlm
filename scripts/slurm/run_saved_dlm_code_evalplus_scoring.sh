#!/usr/bin/env bash
# Evaluate any saved DLM coding artifact with the cached EvalPlus image.
#SBATCH --job-name=dlm-code-evalplus-score
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="b200|a100|a40|a6000|rtxpro6000"
#SBATCH --output=/scratch/%u/slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}}"
cd "${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi

PYTHON_BIN="$(command -v "${PYTHON_BIN:-python}")" || { echo "Activate the DLM Python environment before submitting." >&2; exit 2; }
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
ARTIFACT_ROOT="$("${PYTHON_BIN}" -c 'from ecad.language_models.experiment_artifact_paths import DEFAULT_ARTIFACT_ROOT; print(DEFAULT_ARTIFACT_ROOT)')"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
USE_SAVED_TASK_OUTPUTS="${USE_SAVED_TASK_OUTPUTS:-0}"

if ! command -v "${APPTAINER_BIN}" >/dev/null 2>&1 && type module >/dev/null 2>&1; then
    module load apptainer || true
fi

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${ARTIFACT_ROOT}/apptainer/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${ARTIFACT_ROOT}/apptainer/tmp}"
for sandbox_dir in "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"; do
    if [[ -L "${sandbox_dir}" ]]; then
        echo "Refusing symlinked Apptainer directory: ${sandbox_dir}" >&2
        exit 2
    fi
done
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
chmod 700 "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
export PYTHONUNBUFFERED=1

COMMAND=(srun "${PYTHON_BIN}" -u "${REPO_ROOT}/scripts/evaluate_saved_dlm_code_generations.py")
if (( $# > 0 )); then
    COMMAND+=("$@")
else
    # Keep historical environment-based submissions working.
    RUN_DIR="${RUN_DIR:-${ARTIFACT_ROOT}/results/dlm_ecad/diffucoder7b_code_evalplus_64_len256}"
    ARTIFACT_PATH="${ARTIFACT_PATH:-${RUN_DIR}/baseline.json}"
    OUTPUT_PATH="${OUTPUT_PATH:-${ARTIFACT_PATH%.json}.code_eval.json}"
    COMMAND+=(--run-dir "${RUN_DIR}" --artifact "${ARTIFACT_PATH}"
        --image "${APPTAINER_EVALPLUS_IMAGE:-/scratch/${USER}/apptainer/evalplus.sif}"
        --humaneval-data "${EVALPLUS_HUMANEVAL_DATA:-/scratch/${USER}/evalplus_data/HumanEvalPlus.jsonl}"
        --mbpp-data "${EVALPLUS_MBPP_DATA:-/scratch/${USER}/evalplus_data/MbppPlus.jsonl}"
        --parallel "${SLURM_CPUS_PER_TASK:-8}" --output "${OUTPUT_PATH}"
        --apptainer-bin "${APPTAINER_BIN}")
fi
if [[ "${USE_SAVED_TASK_OUTPUTS}" == "1" ]]; then
    COMMAND+=(--use-saved-task-outputs)
fi

echo "Python: ${PYTHON_BIN}"
"${COMMAND[@]}"
