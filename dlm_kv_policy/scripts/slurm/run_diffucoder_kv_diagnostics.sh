#!/usr/bin/env bash
#SBATCH --job-name=diffucoder-kv-diagnostics
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a6000|a40"
#SBATCH --mail-user=gqs7zr@virginia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/%u/slurm-%x-%j.out

# Runs one diagnostic program. The first argument names it; the rest is
# passed through. Results are stored per GPU type, so any allowed GPU is safe;
# pin with `sbatch --constraint=a40 ...` when two jobs must be compared.
# B200 and RTX PRO 6000 need a CUDA 12.8 PyTorch build; ecad-dlm has 12.4.
# Submit from the repository root with ecad-dlm active:
#
#   sbatch dlm_kv_policy/scripts/slurm/run_diffucoder_kv_diagnostics.sh \
#     output_variance_study --config dlm_kv_policy/configs/<config>.yaml
#
#   sbatch dlm_kv_policy/scripts/slurm/run_diffucoder_kv_diagnostics.sh \
#     forced_trajectory_replay --config dlm_kv_policy/configs/<config>.yaml \
#     --policies full_refresh_every_step,dkv_reference,depth_22,delay_0
#
# Timing comparisons (decoding_oracles, autoregressive_reference) must run on
# the same GPU type; pin it with --constraint.

set -euo pipefail

PROGRAM="${1:-}"
shift || true
case "${PROGRAM}" in
    output_variance_study|forced_trajectory_replay|decoding_oracles|autoregressive_reference) ;;
    *)
        echo "First argument must be output_variance_study, forced_trajectory_replay," \
             "decoding_oracles or autoregressive_reference" >&2
        exit 2
        ;;
esac

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
cd "${REPO_ROOT}"
PYTHON_BIN="$(command -v "${PYTHON_BIN:-python}")" || { echo "Activate the DLM Python environment before submitting." >&2; exit 2; }

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

echo "Program: ${PROGRAM} $*"
echo "Python: ${PYTHON_BIN}"
nvidia-smi
srun "${PYTHON_BIN}" -u -m "dlm_kv_policy.${PROGRAM}" "$@"
