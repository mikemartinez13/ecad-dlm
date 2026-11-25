#!/usr/bin/env bash
# Freeze the source and submit run 4's smoke test (run_search_smoke_v4_rivanna.sh) on raiselab.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SNAP="$REPO/run_snapshots/search_smoke_v4"
LOGS=/scratch/gqs7zr/ai4mat/ecad/logs
[[ -e "$SNAP/jobs.txt" ]] && { echo "$SNAP/jobs.txt exists; refusing duplicate submission" >&2; exit 1; }
mkdir -p "$SNAP" "$LOGS"
rsync -a --delete --exclude=__pycache__ "$REPO/ecad-dlm" "$SNAP/"
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "$name"; done
unset TMPDIR
cd "$SNAP"
job=$(sbatch --parsable --export=ALL --job-name=search-smoke-v4 --account="${ACCOUNT:-raiselab}" --partition=gpu \
    --constraint=a100_80gb --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=00:45:00 \
    --output="$LOGS/slurm-%x-%j.out" "$SNAP/ecad-dlm/experiments/scripts/slurm/run_search_smoke_v4_rivanna.sh")
echo "smoke $job" | tee -a "$SNAP/jobs.txt"
echo "Log: $LOGS/slurm-search-smoke-v4-$job.out"
