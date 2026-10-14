#!/usr/bin/env bash
#SBATCH --job-name=llada-dkv-repro
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000
#SBATCH --output=/scratch/%u/slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}}"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/dkv_repro/.venv/bin/python}"
LLADA_PATH="${LLADA_PATH:-/scratch/gqs7zr/model_checkpoints/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07}"
CASE="${CASE:-lily_128tokens_refresh4}"
REPEATS="${REPEATS:-3}"
WARMUP="${WARMUP:-1}"
LABEL="${LABEL:-${CASE}}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/${USER}/ai4mat/ecad/results/dkv_reproduction}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing reproduction Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -d "${LLADA_PATH}" ]] || { echo "Missing local LLaDA checkpoint: ${LLADA_PATH}" >&2; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}/dkv_cache_official:${REPO_ROOT}/dkv_repro:${PYTHONPATH:-}"
# The generation is GPU-bound. Avoid a large NumPy/OpenBLAS thread pool on the
# login-like nodes used by this cluster's Slurm jobs.
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

echo "Python: ${PYTHON_BIN}"
echo "LLaDA checkpoint: ${LLADA_PATH}"
echo "Official dKV source: ${REPO_ROOT}/dkv_cache_official"
nvidia-smi
srun "${PYTHON_BIN}" -u "${REPO_ROOT}/dkv_repro/benchmark_official_llada_dkv_decode.py" \
  --case "${CASE}" --model-path "${LLADA_PATH}" --repeats "${REPEATS}" --warmup "${WARMUP}" \
  --output-dir "${OUTPUT_DIR}" --label "${LABEL}"
