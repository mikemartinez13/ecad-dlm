#!/usr/bin/env bash
set -euo pipefail
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "$name"; done
unset TMPDIR
cd "/sfs/gpfs/tardis/home/gqs7zr/ecad/run_snapshots/constrained_search_rivanna_v4"
job=$(CONFIG="/sfs/gpfs/tardis/home/gqs7zr/ecad/run_snapshots/constrained_search_rivanna_v4/ecad-dlm/experiments/configs/diffucoder7b_constrained_kv_search_v4_96tasks.yaml" OUTPUT="/scratch/gqs7zr/ai4mat/ecad/results/dlm_kv_policy/constrained_kv_search_v4" LOGS="/scratch/gqs7zr/ai4mat/ecad/logs" ACCOUNT="raiselab" \
    PROCS="1" AUDIT_TOP="8" AUDIT_CONFIG="/sfs/gpfs/tardis/home/gqs7zr/ecad/run_snapshots/constrained_search_rivanna_v4/ecad-dlm/experiments/configs/diffucoder7b_audit_96tasks.yaml" \
    sbatch --parsable --export=ALL --job-name="search-v4" --account="raiselab" --partition=gpu \
    --constraint=a100_80gb --gres=gpu:1 --cpus-per-task=8 --mem=96G --time="3-00:00:00" --signal=B:USR1@1200 \
    --output="/scratch/gqs7zr/ai4mat/ecad/logs/slurm-%x-%j.out" "/sfs/gpfs/tardis/home/gqs7zr/ecad/run_snapshots/constrained_search_rivanna_v4/ecad-dlm/experiments/scripts/slurm/run_search_single_gpu_rivanna.sh")
echo "search $job" | tee -a "/sfs/gpfs/tardis/home/gqs7zr/ecad/run_snapshots/constrained_search_rivanna_v4/jobs.txt"
