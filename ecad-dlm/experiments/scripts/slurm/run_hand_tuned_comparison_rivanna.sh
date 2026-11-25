#!/usr/bin/env bash
# Run 4's latest population, run 3's best and the hand-tuned policies on the audit set, in
# one job (experiments/hand_tuned_comparison.py). Base tests and measured batch-8 speed.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the source snapshot}"
export PYTHONPATH="$PWD/ecad-dlm:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if ! command -v apptainer >/dev/null 2>&1; then
    set +u; source /etc/profile.d/modules.sh; module load apptainer; set -u
fi
RESULTS=/scratch/gqs7zr/ai4mat/ecad/results/dlm_kv_policy
nvidia-smi
/home/gqs7zr/.conda/envs/ecad-dlm/bin/python -u -m experiments.hand_tuned_comparison \
    --config "$PWD/ecad-dlm/experiments/configs/diffucoder7b_audit_96tasks.yaml" \
    --search "$RESULTS/constrained_kv_search_v4" --output "$RESULTS/v4_hand_tuned_comparison"
