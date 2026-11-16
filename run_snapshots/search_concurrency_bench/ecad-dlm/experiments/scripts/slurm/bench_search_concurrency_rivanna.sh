#!/usr/bin/env bash
# Replay throughput with 1-4 search processes sharing one A100 80GB.
# Every process replays the same 3 tasks (part 0 of 32) for one generation's genomes
# into its own folder, so the counts are comparable. Task file times give seconds per
# task without model loading. Submit from a source snapshot:
#   sbatch --account=raiselab-paid --partition=gpu --constraint=a100_80gb --gres=gpu:1 \
#       --cpus-per-task=8 --mem=96G --time=02:30:00 bench_search_concurrency_rivanna.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the source snapshot}"
export PYTHONPATH="$PWD/ecad-dlm:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON_BIN=/home/gqs7zr/.conda/envs/ecad-dlm/bin/python
RESULTS=/scratch/gqs7zr/ai4mat/ecad/results/dlm_kv_policy
CONFIG="$PWD/ecad-dlm/experiments/configs/diffucoder7b_constrained_kv_search_v3_96tasks.yaml"
GENOMES="${GENOMES:-$RESULTS/constrained_kv_search_v2/generation_002/genomes.json}"
OUT="$RESULTS/search_concurrency_bench"
mkdir -p "$OUT"
nvidia-smi
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 5 > "$OUT/gpu.csv" &
monitor=$!
for count in ${COUNTS:-1 2 3 4}; do
    pids=()
    for process in $(seq 1 "$count"); do
        run="$OUT/k${count}_p${process}"
        mkdir -p "$run/generation_000"
        cp "$GENOMES" "$run/generation_000/genomes.json"
        "$PYTHON_BIN" -u -m evolution.constrained_kv_policy_search evaluate --config "$CONFIG" --output "$run" \
            --generation 0 --part 0 --parts 32 > "$run.log" 2>&1 &
        pids+=($!)
    done
    started=$(date +%s)
    for pid in "${pids[@]}"; do wait "$pid" || echo "k=$count: a process failed (see $OUT/k${count}_p*.log)"; done
    echo "k=$count: $(( $(date +%s) - started )) s wall" | tee -a "$OUT/wall.txt"
done
kill "$monitor"
"$PYTHON_BIN" - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); report = {}
for count in sorted({int(p.name.split("_")[0][1:]) for p in out.glob("k*_p*") if p.is_dir()}):
    per_task = []
    for run in sorted(out.glob(f"k{count}_p*")):
        if not run.is_dir():
            continue
        times = sorted(p.stat().st_mtime for p in (run / "generation_000" / "tasks").glob("task_*.json"))
        if len(times) == 3:
            per_task.append((times[-1] - times[0]) / 2)
    if per_task:
        seconds = sum(per_task) / len(per_task)
        report[count] = {"processes_finished": len(per_task), "seconds_per_task_per_process": seconds,
                         "tasks_per_gpu_hour": 3600 * len(per_task) / seconds}
base = report.get(1, {}).get("tasks_per_gpu_hour")
for count, row in report.items():
    row["throughput_vs_one"] = row["tasks_per_gpu_hour"] / base if base else None
    print(f"k={count}: {row['seconds_per_task_per_process']:.0f} s per task per process, "
          f"{row['tasks_per_gpu_hour']:.1f} tasks per GPU-hour, x{row['throughput_vs_one'] or 0:.2f}")
(out / "report.json").write_text(json.dumps(report, indent=2))
EOF
