#!/usr/bin/env bash
# Freeze the source (scripts/slurm/freeze_source.sh) and submit run 4's smoke test
# (run_search_smoke_v4_rivanna.sh) on raiselab.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
NAME=search_smoke_v4
LOGS=/scratch/gqs7zr/ai4mat/ecad/logs
RECORD="$LOGS/runs/$NAME.txt"
SOURCE="$("$REPO/scripts/slurm/freeze_source.sh" "$NAME")"
FROZEN=("$REPO/scripts/slurm/run_frozen.sh" "$REPO" "$SOURCE")
mkdir -p "$LOGS/runs"
echo "source $SOURCE" > "$RECORD"
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "$name"; done
unset TMPDIR
cd "$REPO"
job=$(sbatch --parsable --export=ALL --job-name=search-smoke-v4 --account="${ACCOUNT:-raiselab}" --partition=gpu \
    --constraint=a100_80gb --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=00:45:00 \
    --output="$LOGS/slurm-%x-%j.out" "${FROZEN[@]}" bash ecad-dlm/experiments/scripts/slurm/run_search_smoke_v4_rivanna.sh)
echo "smoke $job" | tee -a "$RECORD"
echo "Log: $LOGS/slurm-search-smoke-v4-$job.out"
