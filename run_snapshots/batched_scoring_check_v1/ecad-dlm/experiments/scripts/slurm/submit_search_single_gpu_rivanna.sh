#!/usr/bin/env bash
# Freeze the source and start one single-GPU search job (run_search_single_gpu_rivanna.sh).
# The job runs every generation, resubmits itself at its time limit, and submits the
# audit and EvalPlus extra tests at the end. Fairshare is charged for GPU time, not
# queue time, so one busy GPU costs no more than many shard jobs and waits once.
#   RUN=v3 CONFIG=diffucoder7b_constrained_kv_search_v3_96tasks.yaml submit_search_single_gpu_rivanna.sh
# PROCS teacher forcing processes share the GPU. Measured 2026-11-12 (bench_search_concurrency_rivanna.sh):
# 2 give 1.08x one, 3 give 1.12x, 4 run out of memory (20 GB each). ACCOUNT, TIME, AUDIT_TOP optional.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
RUN="${RUN:?RUN names the snapshot and output, e.g. v3}"
CONFIG_NAME="${CONFIG:?CONFIG is a search config under experiments/configs}"
SNAP="$REPO/run_snapshots/constrained_search_rivanna_$RUN"
ARTIFACTS=/scratch/gqs7zr/ai4mat/ecad
OUTPUT="$ARTIFACTS/results/dlm_kv_policy/constrained_kv_search_$RUN"
LOGS="$ARTIFACTS/logs"
ACCOUNT="${ACCOUNT:-raiselab-paid}"
PROCS="${PROCS:-2}"
TIME="${TIME:-3-00:00:00}"
AUDIT_TOP="${AUDIT_TOP:-8}"
[[ -e "$SNAP/jobs.txt" ]] && { echo "$SNAP/jobs.txt exists; refusing duplicate submission" >&2; exit 1; }
mkdir -p "$SNAP" "$LOGS" "$OUTPUT"
rsync -a --delete --exclude=__pycache__ "$REPO/ecad-dlm" "$SNAP/"
# Variables from an interactive allocation must not leak into the job. SLURM_CONF
# stays: inside an allocation, sbatch needs it to find the cluster.
for name in $(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "$name"; done
unset TMPDIR
# The job runs this to resubmit itself, from the snapshot, with the same settings.
cat > "$SNAP/resubmit.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
for name in \$(compgen -e | grep -E '^(SLURM|SBATCH)_' | grep -vx SLURM_CONF || true); do unset "\$name"; done
unset TMPDIR
cd "$SNAP"
job=\$(CONFIG="$SNAP/ecad-dlm/experiments/configs/$CONFIG_NAME" OUTPUT="$OUTPUT" LOGS="$LOGS" ACCOUNT="$ACCOUNT" \\
    PROCS="$PROCS" AUDIT_TOP="$AUDIT_TOP" AUDIT_CONFIG="$SNAP/ecad-dlm/experiments/configs/diffucoder7b_audit_96tasks.yaml" \\
    sbatch --parsable --export=ALL --job-name="search-$RUN" --account="$ACCOUNT" --partition=gpu \\
    --constraint=a100_80gb --gres=gpu:1 --cpus-per-task=8 --mem=96G --time="$TIME" --signal=B:USR1@1200 \\
    --output="$LOGS/slurm-%x-%j.out" "$SNAP/ecad-dlm/experiments/scripts/slurm/run_search_single_gpu_rivanna.sh")
echo "search \$job" | tee -a "$SNAP/jobs.txt"
EOF
bash "$SNAP/resubmit.sh"
