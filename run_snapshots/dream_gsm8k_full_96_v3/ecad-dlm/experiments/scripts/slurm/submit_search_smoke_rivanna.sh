#!/usr/bin/env bash
# Freeze the source and submit the 30-minute search smoke test on raiselab-paid.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SNAP="$REPO/run_snapshots/search_smoke_v3"
LOGS=/scratch/gqs7zr/ai4mat/ecad/logs
mkdir -p "$SNAP" "$LOGS"
rsync -a --delete --exclude=__pycache__ "$REPO/ecad-dlm" "$SNAP/"
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "$name"; done
unset TMPDIR
cd "$SNAP"
job=$(sbatch --parsable --export=ALL --job-name=search-smoke-v3 --account="${ACCOUNT:-raiselab-paid}" --partition=gpu \
    --constraint=a100_80gb --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=00:30:00 \
    --output="$LOGS/slurm-%x-%j.out" "$SNAP/ecad-dlm/experiments/scripts/slurm/run_search_smoke_rivanna.sh")
echo "smoke $job" | tee -a "$SNAP/jobs.txt"
echo "Log: $LOGS/slurm-search-smoke-v3-$job.out"
