#!/usr/bin/env bash
# Freeze one full run; chain calibration, every generation, and the final audit.
# RESUME_SUBMISSION=1 continues a partially submitted chain without duplicates.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
: "${RUN:?Set RUN to a unique run name}"
: "${CONFIG:?Set CONFIG to the full-search YAML}"
PY=/home/gqs7zr/.conda/envs/ecad-dlm/bin/python
SNAP="$REPO/run_snapshots/$RUN"
OUT="${OUTPUT:-/scratch/gqs7zr/dream_reasoning/results/$RUN}"
if [[ -e "$SNAP" ]]; then
    [[ "${RESUME_SUBMISSION:-0}" == 1 ]] || { echo "Snapshot already exists: $SNAP" >&2; exit 1; }
else
    mkdir -p "$SNAP/Dream/eval_instruct/lm_eval/tasks/minerva_math" "$OUT"
    rsync -a --exclude=__pycache__ "$REPO/ecad-dlm" "$SNAP/"
    cp "$REPO/Dream/eval_instruct/lm_eval/tasks/minerva_math/utils.py" "$SNAP/Dream/eval_instruct/lm_eval/tasks/minerva_math/"
    cp "$REPO/Dream/LICENSE" "$SNAP/Dream/"
    cp "$REPO/scripts/slurm/run_dream_reasoning.sh" "$SNAP/"
    cp "$REPO/assets/docs/dream_sources.json" "$SNAP/"
    cp "$CONFIG" "$SNAP/config.yaml"
fi
GENERATIONS=$("$PY" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["search"]["generations"])' "$SNAP/config.yaml")
"$PY" -c 'import sys,yaml,json; from pathlib import Path; Path(sys.argv[2]).write_text(json.dumps(yaml.safe_load(open(sys.argv[1])),indent=2))' "$SNAP/config.yaml" "$OUT/run_config.json"
for name in $(compgen -e); do
    if [[ "$name" == SLURM_* || "$name" == SBATCH_* ]]; then
        [[ "$name" == SLURM_CONF ]] || unset "$name"
    fi
done
unset TMPDIR
cd "$SNAP"
JOBS="$OUT/jobs.tsv"
touch "$JOBS"
cp "$REPO/scripts/slurm/submit_dream_full_search.sh" "$SNAP/submit_chain.sh"
submit() {
    local key="$1" dependency="$2"
    shift 2
    local existing
    existing=$(awk -F '\t' -v key="$key" '$1==key {print $2}' "$JOBS")
    if [[ -n "$existing" ]]; then
        LAST_JOB="$existing"
        return
    fi
    local dependency_args=()
    if [[ -n "$dependency" ]]; then
        dependency_args=(--dependency="afterok:$dependency")
    fi
    LAST_JOB=$(sbatch --parsable --account="${ACCOUNT:-raiselab-paid}" --partition=gpu \
        --constraint="${GPU_TYPE:-a100_80gb}" --gres=gpu:1 --cpus-per-task=8 --mem=96G \
        --time="${TIME_LIMIT:-24:00:00}" --job-name="${JOB_PREFIX:-dream}-$key" \
        "${dependency_args[@]}" --output="$OUT/${key}-%j.out" \
        "$SNAP/run_dream_reasoning.sh" "$@" --config "$SNAP/config.yaml" --output "$OUT")
    LAST_JOB="${LAST_JOB%%;*}"
    printf '%s\t%s\t%s\n' "$key" "$LAST_JOB" "$dependency" >> "$JOBS"
}
submit calibration '' calibration
FIRST_JOB="$LAST_JOB"
for ((generation=0; generation<GENERATIONS; generation++)); do
    printf -v key 'g%03d' "$generation"
    submit "$key" "$LAST_JOB" search --generation "$generation"
done
submit audit "$LAST_JOB" finish
cp "$JOBS" "$SNAP/jobs.tsv"
PYTHONPATH="$SNAP/ecad-dlm" "$PY" -c 'import sys; from pathlib import Path; from experiments.dream_search_progress import publish; publish(Path(sys.argv[1]))' "$OUT"
printf 'run=%s\ncalibration_job=%s\nfinal_audit_job=%s\noutput=%s\nsnapshot=%s\n' "$RUN" "$FIRST_JOB" "$LAST_JOB" "$OUT" "$SNAP"
