"""Which cost should the search minimise? Rank four candidates against real batched decoding.

The search needs a cost it can compute from replay alone. This times the tau
confirmation panel (21 policies) with the current batched backend at batch 8, then
checks how well each candidate cost orders the policies:

  flops              operations of each task's own rows (the ECAD paper's TMACs,
                     counted in FLOPs; ``compute_fraction`` in the search)
  padded_flops       the same, with every request charged the batch's largest row
                     count, which is what a batch actually computes
  padded_flops_layer padded_flops plus a fixed cost per executed layer, fitted here
  table              the timed lookup table the search uses today

Row counts come from the saved replay records of ``tau_confirmation run``; no
replay is rerun. Replay follows the dense trajectory, so they are the counts the
search would see, not the counts of free-running decoding.

  time     GPU, A100 80GB. Dense and every panel policy, all tasks, one job.
  report   No model. Writes cost_model_panel_report.json and prints the table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from dlm_kv_policy.compute_cost import ModelShape, dense_forward_flops, layer_flops, lm_head_flops
from dlm_kv_policy.kv_policy_experiment_config import load_kv_policy_experiment_config

from .batched_cost_model import predicted_seconds
from .batched_decoding_study import source_provenance
from .proxy_recheck import rank_score, save

BATCH = 8
GEN_LENGTH = 256


def time_policies(config_path: Path, output: Path, compile_layers: bool) -> None:
    import torch

    from dlm_kv_policy.dense_diffucoder_baseline import BatchedDenseDiffuCoderBaseline

    from .proxy_recheck import backend, generation_settings
    from .tau_confirmation import load, measured_batched_seconds, panel, stop_ids

    config, evaluator = load(config_path)
    model, tokenizer = evaluator.model, evaluator.tokenizer
    stops = stop_ids(tokenizer)
    settings = generation_settings(config)
    path = output / "measured_batch8.json"
    seconds: dict[str, float] = json.loads(path.read_text()) if path.exists() else {}
    save(output / "time_provenance.json", {**source_provenance(), "gpu": torch.cuda.get_device_name(),
                                           "compile_layers": compile_layers, "batch": BATCH})
    makers = {"dense": lambda: BatchedDenseDiffuCoderBaseline(model)}
    for name in panel(stops):
        makers[name] = lambda _name=name: backend(model, panel(stops)[_name], batched=True, compile_layers=compile_layers)
    for name, make in makers.items():
        if name not in seconds:
            seconds[name] = measured_batched_seconds(model, tokenizer, evaluator.tasks, make, settings, config.model.device,
                                                     BATCH, warm_every_batch=compile_layers and name != "dense")
            save(path, seconds)
        print(f"{name}: {seconds[name]:.1f} s, {seconds['dense'] / seconds[name]:.2f}x", flush=True)


def model_shape(model_path: str) -> ModelShape:
    data = json.loads((Path(model_path) / "config.json").read_text())
    return ModelShape(hidden_size=data["hidden_size"], intermediate_size=data["intermediate_size"],
                      num_heads=data["num_attention_heads"], num_kv_heads=data["num_key_value_heads"],
                      num_layers=data["num_hidden_layers"], vocab_size=data["vocab_size"])


def padded_costs(shape: ModelShape, records: list[dict[str, Any]], name: str, depth: int) -> dict[str, float]:
    """FLOPs and executed layers of one policy when tasks decode in the timing harness's batches.

    Batches are consecutive groups of tasks sorted by prompt length, as fixed_batches
    builds them. Every request is charged the batch's longest sequence and largest
    row counts. The head is charged for every masked row; replay does not record
    which of them the backend's lean head would skip.
    """

    ordered = sorted(records, key=lambda record: (record["prompt_tokens"], record["task_id"]))
    depth = min(depth, shape.num_layers)
    policy = dense = layers = 0.0
    for offset in range(0, len(ordered), BATCH):
        group = ordered[offset:offset + BATCH]
        length = max(record["prompt_tokens"] for record in group) + GEN_LENGTH
        rows = [record["metrics"][name]["rows"] for record in group]
        for step in range(len(rows[0])):
            full = max(task[step]["full_depth"] for task in rows)
            active = max(task[step]["active"] for task in rows)
            masked = max(task[step]["masked"] for task in rows)
            policy += len(group) * ((depth - 1) * layer_flops(shape, full, length)
                                    + (shape.num_layers - depth + 1) * layer_flops(shape, active, length)
                                    + lm_head_flops(shape, masked))
            layers += (depth - 1) * (full > 0) + (shape.num_layers - depth + 1) * (active > 0)
            dense += len(group) * (shape.num_layers * layer_flops(shape, length, length) + lm_head_flops(shape, masked))
    return {"flops": policy, "dense_flops": dense, "layers": layers}


def fit_layer_cost(flops: np.ndarray, layers: np.ndarray, seconds: np.ndarray) -> tuple[float, float]:
    """Non-negative seconds per FLOP and per executed layer, by least squares on relative error."""

    design = np.stack([flops, layers], axis=1) / seconds[:, None]
    best = None
    # Two coefficients: try the unconstrained fit, then each one alone, and keep the best valid one.
    for columns in ((0, 1), (0,), (1,)):
        solution, *_ = np.linalg.lstsq(design[:, columns], np.ones(len(seconds)), rcond=None)
        if (solution < 0).any():
            continue
        coefficients = np.zeros(2)
        coefficients[list(columns)] = solution
        error = float(np.square(design @ coefficients - 1).sum())
        if best is None or error < best[0]:
            best = (error, coefficients)
    return float(best[1][0]), float(best[1][1])


def report(config_path: Path, search_config: Path, replay: Path, output: Path) -> dict[str, Any]:
    from .tau_confirmation import DEPTHS

    config = load_kv_policy_experiment_config(config_path)
    shape = model_shape(config.model.path)
    records = [json.loads(path.read_text()) for path in sorted((replay / "tasks").glob("task_*.json"))]
    if len(records) % BATCH:
        raise SystemExit(f"{len(records)} replay records do not fill batches of {BATCH}")
    measured = json.loads((output / "measured_batch8.json").read_text())
    names = [name for name in DEPTHS if name in measured and all(record["metrics"][name].get("rows") for record in records)]
    observed = {name: measured["dense"] / measured[name] for name in names}
    dense_own = sum(record["dense_flops"] for record in records)
    table = json.loads(Path(load_kv_policy_experiment_config(search_config).evolution.cost_table).read_text())

    padded = {name: padded_costs(shape, records, name, DEPTHS[name]) for name in names}
    dense_layers = shape.num_layers * GEN_LENGTH * len(records) / BATCH
    dense_padded = next(iter(padded.values()))["dense_flops"]
    candidates: dict[str, dict[str, float]] = {
        "flops": {name: dense_own / sum(record["metrics"][name]["flops"] for record in records) for name in names},
        "padded_flops": {name: padded[name]["dense_flops"] / padded[name]["flops"] for name in names},
    }
    # The fit sees dense too, so the fixed cost is pinned by a run where it hardly matters.
    fit_names = ["dense", *names]
    flops = np.array([dense_padded, *(padded[name]["flops"] for name in names)])
    layers = np.array([dense_layers, *(padded[name]["layers"] for name in names)])
    seconds = np.array([measured[name] for name in fit_names])
    per_flop, per_layer = fit_layer_cost(flops, layers, seconds)
    fitted = per_flop * flops + per_layer * layers
    candidates["padded_flops_layer"] = {name: float(fitted[0] / fitted[index + 1]) for index, name in enumerate(names)}
    # Leave one policy out: its prediction never saw its own timing.
    held_out = {}
    for index, name in enumerate(names):
        keep = np.arange(len(fit_names)) != index + 1
        a, b = fit_layer_cost(flops[keep], layers[keep], seconds[keep])
        held_out[name] = float((a * flops[0] + b * layers[0]) / (a * flops[index + 1] + b * layers[index + 1]))
    candidates["padded_flops_layer_held_out"] = held_out
    candidates["table"] = {}
    for name in names:
        tasks = [{"task_id": record["task_id"], "prompt_tokens": record["prompt_tokens"], "rows": record["metrics"][name]["rows"]}
                 for record in records]
        try:
            policy_seconds, dense_seconds = predicted_seconds(table, tasks, BATCH, DEPTHS[name])
            candidates["table"][name] = dense_seconds / policy_seconds
        except ValueError:
            pass  # a depth or row count the table was never timed for

    summary: dict[str, Any] = {"policies": names, "measured_speedup": observed, "candidates": {},
                               "layer_fit": {"seconds_per_flop": per_flop, "seconds_per_executed_layer": per_layer,
                                             "fixed_share_of_dense": per_layer * dense_layers / float(fitted[0])}}
    for label, predicted in candidates.items():
        shared = [name for name in names if name in predicted]
        ratio = np.array([predicted[name] / observed[name] for name in shared])
        summary["candidates"][label] = {
            "spearman": rank_score([predicted[name] for name in shared], [observed[name] for name in shared]),
            "policies": len(shared), "median_predicted_over_measured": float(np.median(ratio)),
            "worst_predicted_over_measured": float(ratio[np.argmax(np.abs(np.log(ratio)))]), "predicted_speedup": predicted}
    save(output / "cost_model_panel_report.json", summary)

    labels = list(candidates)
    print(f"{'policy':<22}{'measured':>9}" + "".join(f"{label:>29}" for label in labels))
    for name in sorted(names, key=lambda item: -observed[item]):
        print(f"{name:<22}{observed[name]:>9.2f}" + "".join(f"{candidates[label].get(name, float('nan')):>29.2f}" for label in labels))
    for label in labels:
        row = summary["candidates"][label]
        print(f"{label}: Spearman {row['spearman']:.3f} over {row['policies']} policies, "
              f"median predicted/measured {row['median_predicted_over_measured']:.2f}")
    print("layer fit:", summary["layer_fit"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["time", "report"])
    parser.add_argument("--config", type=Path, required=True, help="the tau confirmation config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-config", type=Path, help="report: names the cost table to compare against")
    parser.add_argument("--replay", type=Path, help="report: output folder of tau_confirmation run")
    parser.add_argument("--compile", action="store_true", help="time: use torch.compile'd layers")
    args = parser.parse_args()
    if args.mode == "time":
        time_policies(args.config, args.output, args.compile)
    else:
        if not args.search_config or not args.replay:
            parser.error("report needs --search-config and --replay")
        report(args.config, args.search_config, args.replay, args.output)


if __name__ == "__main__":
    main()
