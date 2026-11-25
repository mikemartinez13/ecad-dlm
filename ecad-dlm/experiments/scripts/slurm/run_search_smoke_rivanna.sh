#!/usr/bin/env bash
# Smoke test of the single-GPU search path with batched teacher forcing: the v3 config cut
# to its first 16 tasks and one generation. dense -> advance -> evaluate -> advance, then each
# policy's EAR against run 2's batch-one records of the same tasks. Submits nothing else.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Start through scripts/slurm/run_frozen.sh}"
export PYTHONPATH="$PWD/ecad-dlm:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON_BIN=/home/gqs7zr/.conda/envs/ecad-dlm/bin/python
RESULTS=/scratch/gqs7zr/ai4mat/ecad/results/dlm_kv_policy
OUTPUT="$RESULTS/constrained_kv_search_v3_smoke"
CONFIG="$OUTPUT/config.yaml"
mkdir -p "$OUTPUT"
"$PYTHON_BIN" - "$PWD/ecad-dlm/experiments/configs/diffucoder7b_constrained_kv_search_v3_96tasks.yaml" "$CONFIG" <<'PY'
import sys, yaml
raw = yaml.safe_load(open(sys.argv[1]))
raw["run"]["name"] = "diffucoder_constrained_kv_search_v3_smoke"
raw["evaluation"]["max_prompts"] = 16
raw["evolution"]["generations"] = 1
yaml.safe_dump(raw, open(sys.argv[2], "w"), sort_keys=False)
PY
SEARCH=(-u -m evolution.constrained_kv_policy_search)
PATHS=(--config "$CONFIG" --output "$OUTPUT")
nvidia-smi
started=$(date +%s)
"$PYTHON_BIN" "${SEARCH[@]}" dense "${PATHS[@]}"
"$PYTHON_BIN" "${SEARCH[@]}" advance "${PATHS[@]}"
"$PYTHON_BIN" "${SEARCH[@]}" evaluate "${PATHS[@]}" --generation 0 --part 0 --parts 1
"$PYTHON_BIN" "${SEARCH[@]}" advance "${PATHS[@]}"
echo "Smoke stages took $(( $(date +%s) - started )) s"
"$PYTHON_BIN" - "$OUTPUT" "$RESULTS/constrained_kv_search_v2" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
out, v2 = Path(sys.argv[1]), Path(sys.argv[2])
records = [json.loads(p.read_text()) for p in sorted((out / "generation_000" / "tasks").glob("task_*.json"))]
worst, compared = 0.0, 0
for record in records:
    for name, row in record["policies"].items():
        for generation in ("generation_000", "generation_001"):
            path = v2 / generation / "tasks" / f"task_{record['index']:03d}.json"
            saved = json.loads(path.read_text())["policies"].get(name)
            if saved is not None:
                if not (np.isnan(row["ear"]) and np.isnan(saved["ear"])):
                    worst = max(worst, abs(row["ear"] - saved["ear"]))
                compared += 1
                break
summary = json.loads((out / "generation_000" / "summary.json").read_text())
print(f"{len(records)} task records; {compared} task-policy EARs compared with run 2's batch-one records; "
      f"max |dEAR| {worst:.2e}; {summary['feasible']} of {len(summary['population'])} feasible")
PY
