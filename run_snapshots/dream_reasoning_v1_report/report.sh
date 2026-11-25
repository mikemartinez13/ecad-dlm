#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/sfs/gpfs/tardis/home/gqs7zr/ecad/run_snapshots/dream_reasoning_v1_report
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec /home/gqs7zr/.conda/envs/ecad-dlm/bin/python -m experiments.dream_reasoning_report /scratch/gqs7zr/dream_reasoning/results/dream_reasoning_v1
