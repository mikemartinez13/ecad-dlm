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
#SBATCH --output=/scratch/%u/ai4mat/ecad/logs/slurm-%x-%j.out

# Runs one diagnostic program. The first argument names it; the rest is
# passed through. Results are stored per GPU type, so any allowed GPU is safe;
# pin with `sbatch --constraint=a40 ...` when two jobs must be compared.
# B200 and RTX PRO 6000 need a CUDA 12.8 PyTorch build; ecad-dlm has 12.4.
# Submit from the repository root with ecad-dlm active:
#
#   sbatch ecad-dlm/experiments/scripts/slurm/run_diffucoder_kv_diagnostics.sh \
#     output_variance_study --config ecad-dlm/experiments/configs/<config>.yaml
#
#   sbatch ecad-dlm/experiments/scripts/slurm/run_diffucoder_kv_diagnostics.sh \
#     forced_trajectory_replay --config ecad-dlm/experiments/configs/<config>.yaml \
#     --policies full_refresh_every_step,dkv_reference,depth_22,delay_0
#
# Timing comparisons (decoding_oracles, autoregressive_reference) must run on
# the same GPU type; pin it with --constraint. So must collect and evaluate
# of stale_kv_error_study, which read the forced-trajectory replay of that
# type. Its rounding step runs on any A100 (--constraint="a100_80gb|a100_40gb")
# and stale_kv_correction_pilot and stale_kv_correction_diagnostic on any
# allowed GPU; all read the A100 80GB study's data.
#
#   sbatch --constraint=a100_80gb ecad-dlm/experiments/scripts/slurm/run_diffucoder_kv_diagnostics.sh \
#     masked_row_smoothness_study run --arm cadd_on
# The arms (cadd_on, cadd_off, diffucoder_base, diffucoder) must run on one GPU type.
#
#   sbatch --constraint=a100_80gb ecad-dlm/experiments/scripts/slurm/run_diffucoder_kv_diagnostics.sh \
#     commit_position_study run --config ecad-dlm/experiments/configs/<greedy config>.yaml
#   (same for batched_row_timing --config ...). Both read the A100 80GB replay's
#   dense outputs, so run them on that GPU type. Submit from a shell with no SLURM_*
#   variables if you are inside an interactive allocation.
#
#   sbatch ecad-dlm/experiments/scripts/slurm/run_diffucoder_kv_diagnostics.sh \
#     masked_row_reuse_replay run --config ecad-dlm/experiments/configs/<greedy config>.yaml --part 0
#   (parts 0-3). Any allowed GPU; each task is compared with refresh_256 in the same run.
#   Then `masked_row_reuse_replay generate --config ...` (one job, any GPU) for
#   end-to-end pass@1 of dense, refresh_256 and the reuse policies. Add
#   `--base depth_22` to run, report and generate for Step 3.

set -euo pipefail

PROGRAM="${1:-}"
shift || true
case "${PROGRAM}" in
    output_variance_study|forced_trajectory_replay|decoding_oracles|autoregressive_reference|stale_kv_error_study|stale_kv_correction_pilot|stale_kv_correction_diagnostic|masked_row_smoothness_study|commit_position_study|batched_row_timing|masked_row_reuse_replay|batched_decoding_study) ;;
    *)
        echo "First argument must be output_variance_study, forced_trajectory_replay," \
             "decoding_oracles, autoregressive_reference, stale_kv_error_study," \
             "stale_kv_correction_pilot, stale_kv_correction_diagnostic or" \
             "masked_row_smoothness_study, commit_position_study, batched_row_timing" \
             "or masked_row_reuse_replay" >&2
        exit 2
        ;;
esac

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
cd "${REPO_ROOT}"
PYTHON_BIN="$(command -v "${PYTHON_BIN:-python}")" || { echo "Activate the DLM Python environment before submitting." >&2; exit 2; }

export PYTHONPATH="${REPO_ROOT}/ecad-dlm:${REPO_ROOT}:${PYTHONPATH:-}"
ARTIFACT_ROOT="$("${PYTHON_BIN}" -c 'from language_models.experiment_artifact_paths import DEFAULT_ARTIFACT_ROOT; print(DEFAULT_ARTIFACT_ROOT)')"
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
# Proxy code lives in the core package; every other program is an experiment.
MODULE="experiments.${PROGRAM}"
[[ -f "${REPO_ROOT}/ecad-dlm/dlm_kv_policy/${PROGRAM}.py" ]] && MODULE="dlm_kv_policy.${PROGRAM}"
srun "${PYTHON_BIN}" -u -m "${MODULE}" "$@"
