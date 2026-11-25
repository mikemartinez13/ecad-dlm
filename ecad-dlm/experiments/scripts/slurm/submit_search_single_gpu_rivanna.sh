#!/usr/bin/env bash
# Freeze the source (scripts/slurm/freeze_source.sh) and start one single-GPU search job
# (run_search_single_gpu_rivanna.sh). The job runs every generation, resubmits itself on the
# same commit at its time limit, and submits the audit and EvalPlus extra tests at the end.
# Fairshare is charged for GPU time, not queue time, so one busy GPU costs no more than many
# shard jobs and waits once.
#   RUN=v3 CONFIG=diffucoder7b_constrained_kv_search_v3_96tasks.yaml submit_search_single_gpu_rivanna.sh
# PROCS teacher forcing processes share the GPU. Default 1: with batched teacher forcing
# (evolution.teacher_forcing_batch_size) one process keeps the GPU busy. At batch one, 2 processes
# gave 1.08x one (bench_search_concurrency_rivanna.sh, 2026-11-12). ACCOUNT, TIME, AUDIT_TOP optional.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
RUN="${RUN:?RUN names the run and output, e.g. v3}"
CONFIG_NAME="${CONFIG:?CONFIG is a search config under experiments/configs}"
NAME="constrained_search_rivanna_$RUN"
ARTIFACTS=/scratch/gqs7zr/ai4mat/ecad
export OUTPUT="$ARTIFACTS/results/dlm_kv_policy/constrained_kv_search_$RUN"
export LOGS="$ARTIFACTS/logs"
export RECORD="$LOGS/runs/$NAME.txt"
export ACCOUNT="${ACCOUNT:-raiselab-paid}" PROCS="${PROCS:-1}" TIME="${TIME:-3-00:00:00}" AUDIT_TOP="${AUDIT_TOP:-8}"
export JOB_NAME="search-$RUN"
# Paths inside the source are relative: each job runs in its own unpacked copy of the commit.
export CONFIG="ecad-dlm/experiments/configs/$CONFIG_NAME"
export AUDIT_CONFIG=ecad-dlm/experiments/configs/diffucoder7b_audit_96tasks.yaml
git -C "$REPO" cat-file -e "HEAD:$CONFIG" 2>/dev/null || [[ -f "$REPO/$CONFIG" ]] || { echo "No config $CONFIG" >&2; exit 1; }
ECAD_SOURCE_COMMIT="$("$REPO/scripts/slurm/freeze_source.sh" "$NAME")"
export ECAD_REPO="$REPO" ECAD_SOURCE_COMMIT
mkdir -p "$LOGS/runs" "$OUTPUT"
echo "source $ECAD_SOURCE_COMMIT" > "$RECORD"
bash "$REPO/ecad-dlm/experiments/scripts/slurm/sbatch_search_single_gpu_rivanna.sh"
