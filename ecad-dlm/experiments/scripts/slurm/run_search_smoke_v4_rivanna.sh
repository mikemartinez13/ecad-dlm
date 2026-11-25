#!/usr/bin/env bash
# Smoke test for run 4's wider genes: the v4 config cut to its first 16 tasks and one
# generation, whose 16 genomes are run 3's best policy and variants of it using the values
# run 4 adds. Batched teacher forcing on all 16 tasks, then batch one on tasks 0 and 1 for
# comparison. Reports each policy's numbers and the batched minus batch-one differences.
# Submits nothing else.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Start through scripts/slurm/run_frozen.sh}"
export PYTHONPATH="$PWD/ecad-dlm:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON_BIN=/home/gqs7zr/.conda/envs/ecad-dlm/bin/python
RESULTS=/scratch/gqs7zr/ai4mat/ecad/results/dlm_kv_policy
OUTPUT="$RESULTS/constrained_kv_search_v4_smoke"
BATCH_ONE="$OUTPUT/batch_one"
mkdir -p "$OUTPUT" "$BATCH_ONE"
"$PYTHON_BIN" - "$PWD/ecad-dlm/experiments/configs/diffucoder7b_constrained_kv_search_v4_96tasks.yaml" "$OUTPUT" <<'PY'
import json, sys, yaml
from dataclasses import replace
from pathlib import Path
from evolution.nsga2_kv_policy_search import KVRefreshPolicyGenome
source, out = sys.argv[1], Path(sys.argv[2])
raw = yaml.safe_load(open(source))
raw["run"]["name"] = "diffucoder_constrained_kv_search_v4_smoke"
raw["evaluation"]["max_prompts"] = 16
raw["evolution"]["generations"] = 1
yaml.safe_dump(raw, open(out / "config.yaml", "w"), sort_keys=False)
raw["run"]["name"] += "_batch_one"
raw["evaluation"]["max_prompts"] = 2
raw["evolution"]["teacher_forcing_batch_size"] = 1
yaml.safe_dump(raw, open(out / "batch_one" / "config.yaml", "w"), sort_keys=False)
# Run 3's best policy, then one variant per added value, then combinations of them.
best = KVRefreshPolicyGenome(delay=2, refresh_interval=16, refresh_depth=1, terminal_refresh=0, masked_window=8,
                             protected_size=16, protected_rule="split", prompt_refresh_interval=128,
                             local_refresh_radius=4, padding_window=64, entropy_threshold=0.5, entropy_scope="full")
variants = [{}, {"protected_size": 8}, {"protected_size": 12}, {"protected_size": 20}, {"refresh_interval": 64},
            {"prompt_refresh_interval": 256}, {"local_refresh_radius": 6}, {"padding_window": 256},
            {"entropy_threshold": 0.25}, {"entropy_threshold": 1.0},
            {"entropy_threshold": 0.25, "protected_size": 8}, {"local_refresh_radius": 6, "refresh_interval": 64},
            {"prompt_refresh_interval": 256, "padding_window": 256},
            {"masked_window": 16, "padding_window": 256, "protected_size": 12},
            {"refresh_interval": 64, "prompt_refresh_interval": 256, "padding_window": 256, "protected_size": 8,
             "entropy_threshold": 0.25, "local_refresh_radius": 6, "masked_window": 16},
            {"entropy_threshold": 1.0, "entropy_scope": "local", "protected_size": 20, "refresh_interval": 64}]
genomes = [replace(best, **change).canonical() for change in variants]
assert len({genome.name for genome in genomes}) == len(genomes) == 16
for folder in (out, out / "batch_one"):
    (folder / "generation_000").mkdir(parents=True, exist_ok=True)
    (folder / "generation_000" / "genomes.json").write_text(json.dumps([g.to_dict() for g in genomes], indent=1))
print("\n".join(genome.name for genome in genomes))
PY
SEARCH=(-u -m evolution.constrained_kv_policy_search)
nvidia-smi
started=$(date +%s)
"$PYTHON_BIN" "${SEARCH[@]}" dense --config "$OUTPUT/config.yaml" --output "$OUTPUT"
"$PYTHON_BIN" "${SEARCH[@]}" evaluate --config "$OUTPUT/config.yaml" --output "$OUTPUT" --generation 0 --part 0 --parts 1
"$PYTHON_BIN" "${SEARCH[@]}" advance --config "$OUTPUT/config.yaml" --output "$OUTPUT"
echo "Batched stages took $(( $(date +%s) - started )) s"
started=$(date +%s)
"$PYTHON_BIN" "${SEARCH[@]}" evaluate --config "$BATCH_ONE/config.yaml" --output "$BATCH_ONE" --generation 0 --part 0 --parts 1
echo "Batch-one stage took $(( $(date +%s) - started )) s"
"$PYTHON_BIN" - "$OUTPUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
summary = json.loads((out / "generation_000" / "summary.json").read_text())
print(f"{summary['feasible']} of {len(summary['evaluated'])} feasible on 16 tasks")
for item in sorted(summary["evaluated"], key=lambda item: -item["constraints"]["predicted_speedup"]):
    c = item["constraints"]
    print(f"  {item['policy_name']}: padded {c['predicted_speedup']:.2f}x, flops {c['flops_speedup']:.2f}x, EAR {c['ear']:.4f}, lower {c['ear_lower_95']:.4f}, "
          f"tail {c['tail_ear']:.4f}, coverage {c['answer_coverage']:.4f}, violation {item['violation']:.4f}")
print("Batched minus batch one, tasks 0 and 1:")
for index in (0, 1):
    batched = json.loads((out / "generation_000" / "tasks" / f"task_{index:03d}.json").read_text())["policies"]
    single = json.loads((out / "batch_one" / "generation_000" / "tasks" / f"task_{index:03d}.json").read_text())["policies"]
    ear = max(abs(batched[name]["ear"] - single[name]["ear"]) for name in single)
    flops = max(abs(batched[name]["flops"] / single[name]["flops"] - 1) for name in single)
    print(f"  task {index}: max |dEAR| {ear:.2e}, max relative dFLOPs {flops:.2e} over {len(single)} policies")
PY
