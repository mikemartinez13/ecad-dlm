#!/usr/bin/env bash
# Vectorized scoring check (experiments/BATCHED_TEACHER_FORCING_CHECK.md) on one A100 80GB.
# Reuses the first check's trajectories and per-task-scored records in $OUTPUT. Started by
# submit_batched_scoring_check_rivanna.sh from a source snapshot; finished tasks are skipped.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the source snapshot}"
export PYTHONPATH="$PWD/ecad-dlm:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON_BIN=/home/gqs7zr/.conda/envs/ecad-dlm/bin/python
CHECK=(-u -m experiments.batched_teacher_forcing_check)
ARGS=(--config "$PWD/ecad-dlm/experiments/configs/diffucoder7b_constrained_kv_search_v3_96tasks.yaml" --output "$OUTPUT")
[[ -d "$OUTPUT/trajectories" ]] || { echo "No trajectories in $OUTPUT" >&2; exit 1; }
nvidia-smi
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 10 > "$OUTPUT/gpu_scoring.csv" &
monitor=$!
trap 'kill $monitor 2>/dev/null || true' EXIT
# 1. Where the time goes: one batch of 16, each scoring path, timed per phase.
for scoring in per_task vectorized; do
    "$PYTHON_BIN" "${CHECK[@]}" force "${ARGS[@]}" --generation v2_generation_001 --batch-size 16 --max-batches 1 \
        --scoring "$scoring" --profile --folder "profile_$scoring"
done
# 2. Both panel generations with vectorized scoring, unsynchronized, for the comparison and speed.
for generation in v1_generation_007 v2_generation_001; do
    "$PYTHON_BIN" "${CHECK[@]}" force "${ARGS[@]}" --generation "$generation" --batch-size 16 \
        --scoring vectorized --folder "vectorized_$generation"
done
# 3. Vectorized throughput at B=8 and B=32, two batches each; running out of memory is reported, not fatal.
for size in 8 32; do
    "$PYTHON_BIN" "${CHECK[@]}" force "${ARGS[@]}" --generation v2_generation_001 --batch-size "$size" --max-batches 2 \
        --scoring vectorized --folder "vectorized_timing_b$size" || echo "Batch size $size failed (see above)"
done
"$PYTHON_BIN" "${CHECK[@]}" compare-scoring "${ARGS[@]}"
