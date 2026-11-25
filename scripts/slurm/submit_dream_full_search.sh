#!/usr/bin/env bash
# Freeze one full run (freeze_source.sh); chain calibration, every generation, and the final
# audit, each run on that commit through run_frozen.sh.
# RESUME_SUBMISSION=1 continues a partially submitted chain on the same commit without duplicates.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
: "${RUN:?Set RUN to a unique run name}"
: "${CONFIG:?Set CONFIG to the full-search YAML}"
PY=/home/gqs7zr/.conda/envs/ecad-dlm/bin/python
"$PY" "$REPO/scripts/check_dream_scope.py" "$CONFIG"
OUT="${OUTPUT:-/scratch/gqs7zr/dream_reasoning/results/$RUN}"
if [[ "${RESUME_SUBMISSION:-0}" == 1 ]]; then
    SOURCE="$("$REPO/scripts/slurm/freeze_source.sh" --existing "$RUN")"
else
    # The commit holds ecad-dlm/ and scripts/, the Dream files the scorer needs (Dream/ is
    # ignored, so it is added explicitly), the source list, and the config as config.yaml.
    SOURCE="$("$REPO/scripts/slurm/freeze_source.sh" --force Dream/eval_instruct/lm_eval/tasks/minerva_math/utils.py \
        --force Dream/LICENSE --include assets/docs/dream_sources.json --file "config.yaml=$CONFIG" "$RUN")"
fi
mkdir -p "$OUT"
FROZEN_CONFIG="$(mktemp)"
trap 'rm -f "$FROZEN_CONFIG"' EXIT
git -C "$REPO" show "$SOURCE:config.yaml" > "$FROZEN_CONFIG"
GENERATIONS=$("$PY" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["search"]["generations"])' "$FROZEN_CONFIG")
"$PY" "$REPO/scripts/check_dream_scope.py" "$FROZEN_CONFIG"
"$PY" -c 'import sys,yaml,json; from pathlib import Path; Path(sys.argv[2]).write_text(json.dumps(yaml.safe_load(open(sys.argv[1])),indent=2))' "$FROZEN_CONFIG" "$OUT/run_config.json"
for name in $(compgen -e); do
    if [[ "$name" == SLURM_* || "$name" == SBATCH_* ]]; then
        [[ "$name" == SLURM_CONF ]] || unset "$name"
    fi
done
unset TMPDIR
cd "$REPO"
FROZEN=("$REPO/scripts/slurm/run_frozen.sh" "$REPO" "$SOURCE")
JOBS="$OUT/jobs.tsv"
touch "$JOBS"
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
    # Paths inside the source are relative: each job runs in its own unpacked copy of the commit.
    LAST_JOB=$(sbatch --parsable --account="${ACCOUNT:-raiselab}" --partition=gpu \
        --constraint="${GPU_TYPE:-a100_80gb}" --gres=gpu:1 --cpus-per-task=8 --mem=96G \
        --time="${TIME_LIMIT:-24:00:00}" --job-name="${JOB_PREFIX:-dream}-$key" \
        "${dependency_args[@]}" --output="$OUT/${key}-%j.out" \
        "${FROZEN[@]}" bash scripts/slurm/run_dream_reasoning.sh "$@" --config config.yaml --output "$OUT")
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
PYTHONPATH="$REPO/ecad-dlm" "$PY" -c 'import sys; from pathlib import Path; from experiments.dream_search_progress import publish; publish(Path(sys.argv[1]))' "$OUT"
printf 'run=%s\ncalibration_job=%s\nfinal_audit_job=%s\noutput=%s\nsource=%s\n' "$RUN" "$FIRST_JOB" "$LAST_JOB" "$OUT" "$SOURCE"
