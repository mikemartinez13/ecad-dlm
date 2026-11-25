#!/usr/bin/env bash
# Submit one single-GPU search job (run_search_single_gpu_rivanna.sh) on the frozen source
# ECAD_SOURCE_COMMIT of the repository ECAD_REPO. submit_search_single_gpu_rivanna.sh runs it
# for the first job, and the job runs it to resubmit itself at its time limit, so both use the
# same commit and settings. The settings come from the environment: JOB_NAME, ACCOUNT, TIME,
# LOGS, RECORD, and CONFIG, OUTPUT, AUDIT_CONFIG, AUDIT_TOP and PROCS for the job itself.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# Variables from an interactive allocation or from this job must not leak into the next.
# SLURM_CONF stays: inside an allocation, sbatch needs it to find the cluster.
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "$name"; done
unset TMPDIR PYTHONPATH
cd "$ECAD_REPO"
job=$(sbatch --parsable --export=ALL --job-name="$JOB_NAME" --account="$ACCOUNT" --partition=gpu \
    --constraint=a100_80gb --gres=gpu:1 --cpus-per-task=8 --mem=96G --time="$TIME" --signal=B:USR1@1200 \
    --output="$LOGS/slurm-%x-%j.out" "$ROOT/scripts/slurm/run_frozen.sh" "$ECAD_REPO" "$ECAD_SOURCE_COMMIT" \
    bash ecad-dlm/experiments/scripts/slurm/run_search_single_gpu_rivanna.sh)
echo "search $job" | tee -a "$RECORD"
