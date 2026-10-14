#!/usr/bin/env bash
# Search attention/MLP residual-output-cache schedules for the configured DLM.
#SBATCH --job-name=dlm-residual-cache-search
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
# PyTorch 2.4.1+cu124 in ecad-dlm supports CUDA architectures through sm_90;
# exclude Blackwell GPUs until this environment is upgraded for them.
#SBATCH --constraint="a100|a40|a6000"
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
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/dlm/diffucoder7b_residual_cache_text_agreement_search_8prompts_32tokens.yaml}"
# The YAML model path is the default for either family. Override it explicitly
# rather than forwarding a DiffuCoder checkpoint to a LLaDA configuration.
MODEL_PATH="${MODEL_PATH:-}"
MAX_GENERATIONS="${MAX_GENERATIONS:-}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
EVALPLUS_IMAGE="${APPTAINER_EVALPLUS_IMAGE:-}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
ARTIFACT_ROOT="$("${PYTHON_BIN}" -c 'from ecad.language_models.experiment_artifact_paths import DEFAULT_ARTIFACT_ROOT; print(DEFAULT_ARTIFACT_ROOT)')"

# Correctness-aware code runs invoke EvalPlus inside a no-network Apptainer
# sandbox. Token-only runs do not require it, so module discovery is best-effort
# here and the Python evaluator fails closed if a configured run cannot find it.
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COMMAND=(srun "${PYTHON_BIN}" -u -m ecad.genetic.search_dlm_residual_cache_schedules --config "${CONFIG_PATH}")
COMMAND+=(--apptainer-bin "${APPTAINER_BIN}")
if [[ -n "${MODEL_PATH}" ]]; then
    COMMAND+=(--model-path "${MODEL_PATH}")
fi
if [[ -n "${EVALPLUS_IMAGE}" ]]; then
    COMMAND+=(--evalplus-image "${EVALPLUS_IMAGE}")
fi
if [[ -n "${MAX_GENERATIONS}" ]]; then
    COMMAND+=(--max-generations "${MAX_GENERATIONS}")
fi
# Explicit script arguments override defaults and legacy environment settings.
COMMAND+=("$@")

echo "Config: ${CONFIG_PATH}"
echo "Python: ${PYTHON_BIN}"
nvidia-smi
"${COMMAND[@]}"
