#!/usr/bin/env bash
#SBATCH --job-name=diffucoder-kv-experiments
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40|a6000"
#SBATCH --mail-user=gqs7zr@virginia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/%u/slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}}"
cd "${REPO_ROOT}"
PYTHON_BIN="$(command -v "${PYTHON_BIN:-python}")" || { echo "Activate the DLM Python environment before submitting." >&2; exit 2; }
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/dlm_kv_policy/configs/diffucoder7b_kv_fixed_policy_screen_64tasks_256tokens.yaml}"
RUN_NAME="${RUN_NAME:-}"
FIXED_POLICY_NAMES="${FIXED_POLICY_NAMES:-}"
DKV_DECODE_REFERENCE_EVALUATION="${DKV_DECODE_REFERENCE_EVALUATION:-0}"
DKV_DECODE_ABLATION_EVALUATION="${DKV_DECODE_ABLATION_EVALUATION:-0}"
KV_POLICY_NSGA2_SEARCH="${KV_POLICY_NSGA2_SEARCH:-0}"
EVOLUTION_POPULATIONS_PER_JOB="${EVOLUTION_POPULATIONS_PER_JOB:-1}"
ABLATION_POLICY_NAMES="${ABLATION_POLICY_NAMES:-}"
REUSE_DENSE_BASELINE_DIR="${REUSE_DENSE_BASELINE_DIR:-}"
SKIP_CODE_EXECUTION="${SKIP_CODE_EXECUTION:-0}"
HELDOUT_EVALPLUS_POLICY="${HELDOUT_EVALPLUS_POLICY:-}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
ARTIFACT_ROOT="$("${PYTHON_BIN}" -c 'from ecad.language_models.experiment_artifact_paths import DEFAULT_ARTIFACT_ROOT; print(DEFAULT_ARTIFACT_ROOT)')"
if ! command -v apptainer >/dev/null 2>&1 && type module >/dev/null 2>&1; then
    module load apptainer || true
fi
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${ARTIFACT_ROOT}/apptainer/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${ARTIFACT_ROOT}/apptainer/tmp}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

ARGS=(--config "${CONFIG_PATH}")
[[ -n "${RUN_NAME}" ]] && ARGS+=(--run-name "${RUN_NAME}")
[[ -n "${FIXED_POLICY_NAMES}" ]] && ARGS+=(--fixed-policy-names "${FIXED_POLICY_NAMES}")
EXPERIMENT_MODE_COUNT=0
if [[ "${DKV_DECODE_REFERENCE_EVALUATION}" == "1" ]]; then
    ((EXPERIMENT_MODE_COUNT += 1))
fi
if [[ "${DKV_DECODE_ABLATION_EVALUATION}" == "1" ]]; then
    ((EXPERIMENT_MODE_COUNT += 1))
fi
if [[ "${KV_POLICY_NSGA2_SEARCH}" == "1" ]]; then
    ((EXPERIMENT_MODE_COUNT += 1))
fi
if (( EXPERIMENT_MODE_COUNT > 1 )); then
    echo "Choose one experiment: dKV reference evaluation, dKV ablation evaluation, or KV policy NSGA-II search" >&2
    exit 2
fi
[[ "${DKV_DECODE_REFERENCE_EVALUATION}" == "1" ]] && ARGS+=(--dkv-decode-reference-evaluation)
[[ "${DKV_DECODE_ABLATION_EVALUATION}" == "1" ]] && ARGS+=(--dkv-decode-ablation-evaluation)
if [[ "${KV_POLICY_NSGA2_SEARCH}" == "1" ]]; then
    ARGS+=(--kv-policy-nsga2-search --populations-per-job "${EVOLUTION_POPULATIONS_PER_JOB}")
fi
[[ -n "${ABLATION_POLICY_NAMES}" ]] && ARGS+=(--ablation-policy-names "${ABLATION_POLICY_NAMES}")
[[ -n "${REUSE_DENSE_BASELINE_DIR}" ]] && ARGS+=(--reuse-dense-baseline-dir "${REUSE_DENSE_BASELINE_DIR}")
[[ "${SKIP_CODE_EXECUTION}" == "1" ]] && ARGS+=(--skip-code-execution)
[[ -n "${HELDOUT_EVALPLUS_POLICY}" ]] && ARGS+=(--heldout-evalplus-policy "${HELDOUT_EVALPLUS_POLICY}")
ARGS+=("$@")

echo "Config: ${CONFIG_PATH}"
echo "Python: ${PYTHON_BIN}"
nvidia-smi
srun "${PYTHON_BIN}" -u -m dlm_kv_policy.run_diffucoder_kv_experiments "${ARGS[@]}"
