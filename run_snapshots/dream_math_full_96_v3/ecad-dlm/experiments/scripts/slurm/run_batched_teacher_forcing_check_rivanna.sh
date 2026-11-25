#!/usr/bin/env bash
# Batched teacher forcing check (experiments/BATCHED_TEACHER_FORCING_CHECK.md) on one A100 80GB.
# Started by submit_batched_teacher_forcing_check_rivanna.sh from a source snapshot. Every
# stage skips finished tasks, so resubmitting the same snapshot resumes.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the source snapshot}"
export PYTHONPATH="$PWD/ecad-dlm:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON_BIN=/home/gqs7zr/.conda/envs/ecad-dlm/bin/python
CHECK=(-u -m experiments.batched_teacher_forcing_check)
ARGS=(--config "$PWD/ecad-dlm/experiments/configs/diffucoder7b_constrained_kv_search_v3_96tasks.yaml" --output "$OUTPUT")
BATCH="${BATCH:-16}"
mkdir -p "$OUTPUT"
nvidia-smi
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 10 > "$OUTPUT/gpu.csv" &
monitor=$!
trap 'kill $monitor 2>/dev/null || true' EXIT
"$PYTHON_BIN" "${CHECK[@]}" trajectories "${ARGS[@]}"
"$PYTHON_BIN" "${CHECK[@]}" control "${ARGS[@]}"
for generation in v1_generation_007 v2_generation_001; do
    "$PYTHON_BIN" "${CHECK[@]}" force "${ARGS[@]}" --generation "$generation" --batch-size "$BATCH"
done
# Throughput at other batch sizes, two batches each; a size that runs out of memory is reported, not fatal.
for size in 8 32; do
    "$PYTHON_BIN" "${CHECK[@]}" force "${ARGS[@]}" --generation v2_generation_001 --batch-size "$size" \
        --max-batches 2 --folder "timing_b$size" || echo "Batch size $size failed (see above)"
done
"$PYTHON_BIN" "${CHECK[@]}" report "${ARGS[@]}"
