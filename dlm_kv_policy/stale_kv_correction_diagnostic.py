"""Diagnostic: why the fitted correction did not reduce divergence events.

The correction pilot added the M4-predicted stale error to stale K/V rows and
divergence did not fall. Three explanations are told apart here (criteria in
STALE_KV_ERROR_STUDY.md):

  compounding        the corrected K/V are not closer to dense, because
                     corrections in earlier layers and steps change what the
                     cache holds, and the fit no longer applies;
  not_decision_relevant
                     the corrected K/V are closer to dense, but each layer's
                     attention output is not;
  residual           both are closer, but decisions still turn on the error
                     that cannot be predicted.

Every task is replayed with each policy at strength 0 (the uncorrected run,
which the pilot showed is exactly the plain policy), 0.5 and 1.0. Per run and
layer, summed over steps:

  - stale rows' K/V error at read time against the same step's dense K/V,
    before and after the correction (before RoPE, which preserves norms);
  - the attention output (o_proj input) of the rows computed that step,
    against dense.

  run --fold K --part P   GPU. Replays half of fold K's tasks.
  report                  Ratios to the uncorrected run, with intervals. No model.

The layer-4 test adds --skip-layers 4 to both. The corrections then leave
layer 4 alone, and refresh_32 and refresh_256 also run the full correction
("<policy>@<strength>_full") with an attention probe. At every corrected layer
the probe recomputes attention with the cached and with the corrected K/V,
for the same queries. It records how concentrated attention is, where the
change lands, and how much of the output change comes from keys or values.

Usage (from the repository root; run inside a GPU allocation):

    python -m dlm_kv_policy.stale_kv_correction_diagnostic run \\
        --config dlm_kv_policy/configs/diffucoder7b_output_variance_greedy_64tasks.yaml --fold 0 --part 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .code_correctness_and_latency_evaluator import gpu_folder_name
from .dense_diffucoder_baseline import DenseDiffuCoderBaseline
from .diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend
from .diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
from .forced_trajectory_replay import load_top_p_logits, replay_task
from .kv_policy_experiment_config import load_kv_policy_experiment_config
from .stale_kv_correction_pilot import (
    INTERVAL,
    PRIMARY_POLICY,
    STRENGTHS,
    StaleRowCorrection,
    paired_interval,
    pilot_dir,
    task_metrics,
)
from .stale_kv_error_study import (
    BOOTSTRAP_RESAMPLES,
    FOLDS,
    PARTS,
    POLICIES,
    SCORED_FIRST_LAYER,
    STUDY_GPU,
    DenseWithPreRopeKV,
    Study,
    same_steps,
)

# Strength 0 is the uncorrected run every ratio is taken against.
DIAGNOSED_STRENGTHS = (0.0, *STRENGTHS)
SUM_KEYS = ("kv_before", "kv_after", "kv_dense", "attention_error", "attention_dense", "attention_rows")
# In the layer-4 test these policies also run the full correction, probed.
PROBED_POLICIES = ("refresh_32", "refresh_256")
# Per layer, summed over steps, heads and query rows. "Before" attention uses
# the cached K/V and "after" the corrected K/V; "top" is the key row with the
# most attention before, and "first" is position 0. Output energies: key_part
# changes keys only, value_part then changes values too.
PROBE_KEYS = (
    "queries", "top_share", "first_share", "first_is_top", "change", "change_at_top", "change_at_first",
    "key_part", "value_part", "total_change", "output",
)


def diagnostic_dir(config: Any, skipped_layers: tuple[int, ...] = ()) -> Path:
    name = "stale_kv_correction_diagnostic"
    if skipped_layers:
        name += "_skip" + "_".join(str(layer) for layer in skipped_layers)
    return config.run.run_dir / STUDY_GPU / name


def diagnosed_runs(policy_names: list[str], skipped_layers: tuple[int, ...] = ()) -> list[tuple[str, str, float, tuple[int, ...], bool]]:
    """(label, policy, strength, skipped layers, probed) for each run.

    Each policy runs at strengths 0, 0.5 and 1.0, its corrections skipping
    skipped_layers. If any are skipped, the probed policies also run the full
    correction, probed, labelled '<policy>@<strength>_full'.
    """

    runs = [(f"{name}@{strength}", name, strength, skipped_layers, False) for name in policy_names for strength in DIAGNOSED_STRENGTHS]
    if skipped_layers:
        runs += [
            (f"{name}@{strength}_full", name, strength, (), True)
            for name in policy_names if name in PROBED_POLICIES for strength in STRENGTHS
        ]
    return runs


class AttentionOutputs:
    """Records each layer's attention output (the o_proj input, [rows, hidden]) in a backend's last forward."""

    def __init__(self, model: Any, backend: Any) -> None:
        self.layers = model.model.layers
        self.outputs: list[Any] = [None] * len(self.layers)
        self.forward = backend.forward_mask_logits
        backend.forward_mask_logits = self.forward_mask_logits

    def forward_mask_logits(self, **inputs: Any) -> Any:
        self.outputs = [None] * len(self.layers)

        def recorder(layer_id: int):
            def hook(module, args) -> None:
                self.outputs[layer_id] = args[0][0]

            return hook

        handles = [layer.self_attn.o_proj.register_forward_pre_hook(recorder(i)) for i, layer in enumerate(self.layers)]
        try:
            return self.forward(**inputs)
        finally:
            for handle in handles:
                handle.remove()


def empty_sums(num_layers: int, device: Any) -> dict[str, torch.Tensor]:
    """Per layer: K/V sums are [layers, kinds], attention sums are [layers]."""

    shapes = {"kv_before": (num_layers, 2), "kv_after": (num_layers, 2), "kv_dense": (num_layers, 2)}
    return {key: torch.zeros(shapes.get(key, (num_layers,)), dtype=torch.float64, device=device) for key in SUM_KEYS}


def energy(x: torch.Tensor) -> torch.Tensor:
    return x.double().square().sum()


def attention_effect(
    query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, cached_keys: torch.Tensor, cached_values: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """How replacing the cached K/V by the corrected K/V changes one layer's attention (PROBE_KEYS).

    query is [1, heads, rows, head_dim] and each K/V is [1, kv_heads, length, head_dim].
    """

    kv_heads = keys.shape[1]
    groups = query.shape[1] // kv_heads
    scale = query.shape[-1] ** -0.5
    sums = {key: torch.zeros((), dtype=torch.float64, device=query.device) for key in PROBE_KEYS}
    for head in range(kv_heads):
        q = query[0, head * groups : (head + 1) * groups].float()
        before = torch.softmax(q @ cached_keys[0, head].float().T * scale, dim=-1)  # [groups, rows, length]
        after = torch.softmax(q @ keys[0, head].float().T * scale, dim=-1)
        change = (after - before).abs()
        top = before.argmax(dim=-1, keepdim=True)
        output_before = before @ cached_values[0, head].float()
        output_keys = after @ cached_values[0, head].float()
        output_after = after @ values[0, head].float()
        sums["queries"] += before.shape[0] * before.shape[1]
        sums["top_share"] += before.amax(dim=-1).double().sum()
        sums["first_share"] += before[..., 0].double().sum()
        sums["first_is_top"] += top.eq(0).sum()
        sums["change"] += change.double().sum()
        sums["change_at_top"] += change.gather(-1, top).double().sum()
        sums["change_at_first"] += change[..., 0].double().sum()
        sums["key_part"] += energy(output_keys - output_before)
        sums["value_part"] += energy(output_after - output_keys)
        sums["total_change"] += energy(output_after - output_before)
        sums["output"] += energy(output_before)
    return sums


class AttentionProbe:
    """Sums attention_effect per layer, at every layer where the backend's K/V were corrected."""

    def __init__(self, backend: DiffuCoderActiveTokenKVBackend, num_layers: int, device: Any) -> None:
        self.backend = backend
        self.sums = {key: torch.zeros(num_layers, dtype=torch.float64, device=device) for key in PROBE_KEYS}
        backend.attention_observer = self.observe

    def observe(self, layer: int, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: Any) -> None:
        cached_keys, cached_values = self.backend.cache.kv(layer)
        if keys is cached_keys and values is cached_values:
            return  # nothing was corrected at this layer and step
        if mask is not None:
            raise ValueError("The attention probe assumes no attention mask (batch size one, no padding)")
        for key, value in attention_effect(query, keys, values, cached_keys, cached_values).items():
            self.sums[key][layer] += value


class DiagnosedReplay:
    """Forced-trajectory replay of every policy at each strength, summing errors against dense."""

    def __init__(
        self, model: Any, dense_backend: Any, policies: dict[str, Any], fits: dict[str, dict[str, torch.Tensor]],
        stop_ids: set[int], generation: Any, top_p_logits: Any, runs: list[tuple] | None = None,
    ) -> None:
        attention = model.model.layers[0].self_attn
        num_layers = len(model.model.layers)
        device = next(model.parameters()).device
        self.model = model
        self.generation = generation
        self.top_p_logits = top_p_logits
        self.dense = DenseWithPreRopeKV(dense_backend, attention.num_key_value_heads, attention.head_dim)
        self.dense_outputs = AttentionOutputs(model, self.dense)
        self.sampler = DiffuCoderDenoisingSampler(self.dense)
        self.runs = runs or diagnosed_runs(list(policies))
        self.backends, self.outputs, self.sums, self.probes, self.corrections = {}, {}, {}, {}, []
        for label, name, strength, skipped_layers, probed in self.runs:
            backend = DiffuCoderActiveTokenKVBackend(model, policies[name])
            correction = StaleRowCorrection(backend, fits[name], strength, stop_ids, skipped_layers)
            self.sums[label] = empty_sums(num_layers, device)
            correction.observer = self.kv_observer(self.sums[label])
            self.corrections.append(correction)
            self.backends[label] = backend
            self.outputs[label] = AttentionOutputs(model, backend)
            if probed:
                self.probes[label] = AttentionProbe(backend, num_layers, device)

    def kv_observer(self, sums: dict[str, torch.Tensor]):
        def observe(layer: int, kind: int, positions: torch.Tensor, c: torch.Tensor, shift: torch.Tensor) -> None:
            dense = (self.dense.keys if kind == 0 else self.dense.values)[layer][:, positions].float()
            before = dense - c
            sums["kv_before"][layer, kind] += energy(before)
            sums["kv_after"][layer, kind] += energy(before - shift)
            sums["kv_dense"][layer, kind] += energy(dense)

        return observe

    def on_step(self, step: int, x: torch.Tensor, state: Any) -> None:
        for label, backend in self.backends.items():
            sums = self.sums[label]
            for layer in range(backend.num_layers):
                rows = torch.where(backend.cache.layer_last_refresh[layer, 0].eq(step))[0]
                if rows.numel() == 0:
                    continue
                output = self.outputs[label].outputs[layer]
                if output is None or output.shape[0] != rows.numel():
                    raise RuntimeError(f"{label}: layer {layer} attention output does not match the rows computed")
                dense = self.dense_outputs.outputs[layer][rows].float()
                sums["attention_error"][layer] += energy(output.float() - dense)
                sums["attention_dense"][layer] += energy(dense)
                sums["attention_rows"][layer] += rows.numel()

    def run(self, prompt: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, Any]:
        for sums in [*self.sums.values(), *(probe.sums for probe in self.probes.values())]:
            for tensor in sums.values():
                tensor.zero_()
        for correction in self.corrections:
            correction.start(prompt.shape[1], prompt.shape[1] + self.generation.gen_length)
        return replay_task(
            self.model, self.sampler, self.backends, prompt, attention_mask, self.generation, self.top_p_logits,
            on_step=self.on_step,
        )


def run(config: Any, fold: int, part: int, skipped_layers: tuple[int, ...] = ()) -> None:
    study = Study(config, STUDY_GPU)
    gpu = gpu_folder_name()
    output_dir = diagnostic_dir(config, skipped_layers)
    study.write_job_info(output_dir.name, fold, part)
    device = config.model.device
    fits = {}
    for name in POLICIES:
        with np.load(study.output_dir / "fit" / name / f"fold{fold}.npz") as saved:
            fits[name] = {key: torch.from_numpy(saved[key]).to(device) for key in ("m4_matrix", "m4_intercept")}
    replay = DiagnosedReplay(
        study.model, DenseDiffuCoderBaseline(study.model), {name: config.dkv_policies[name] for name in POLICIES},
        fits, study.stop_ids, config.generation, load_top_p_logits(study.model),
        diagnosed_runs(list(POLICIES), skipped_layers),
    )
    # Runs that correct every layer, or nothing, are the pilot's runs under another label.
    pilot_labels = {
        label: f"{name}@{strength}" for label, name, strength, skipped, _ in replay.runs if not skipped or strength == 0
    }

    indices = study.part_tasks(fold, part)
    for count, index in enumerate(indices):
        prompt, attention_mask, _, saved = study.task_inputs(index)
        pilot = json.loads((pilot_dir(config, STUDY_GPU) / "tasks" / f"task_{index:03d}.json").read_text())
        result = replay.run(prompt, attention_mask)
        # The recorders only read, so on the pilot task's GPU type those runs must equal the pilot's.
        matches_pilot = None
        if gpu == pilot["gpu"]:
            matches_pilot = all(
                same_steps(result["policies"][label], pilot["steps"][pilot_label]) for label, pilot_label in pilot_labels.items()
            )
        record = {
            "task_id": study.tasks[index].task_id,
            "fold": fold,
            "gpu": gpu,
            "pilot_gpu": pilot["gpu"],
            "matches_pilot": matches_pilot,
            "skipped_layers": list(skipped_layers),
            "metrics": {label: task_metrics(rows) for label, rows in result["policies"].items()},
            "sums": {label: {key: tensor.cpu().tolist() for key, tensor in sums.items()} for label, sums in replay.sums.items()},
            "probe": {
                label: {key: tensor.cpu().tolist() for key, tensor in probe.sums.items()} for label, probe in replay.probes.items()
            },
        }
        path = output_dir / "tasks" / f"task_{index:03d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")

        def scored(label: str, key: str) -> float:
            return float(replay.sums[label][key][SCORED_FIRST_LAYER:].sum())

        base = f"{PRIMARY_POLICY}@0.0"
        events = record["metrics"]
        ratios = f"divergence events {events[base]['divergence_events']:.1f} uncorrected; " + " ".join(
            f"@{strength}: K/V {scored(f'{PRIMARY_POLICY}@{strength}', 'kv_after') / scored(base, 'kv_after'):.3f}, "
            f"attention {scored(f'{PRIMARY_POLICY}@{strength}', 'attention_error') / scored(base, 'attention_error'):.3f}, "
            f"events {events[f'{PRIMARY_POLICY}@{strength}']['divergence_events']:.1f};"
            for strength in STRENGTHS
        )
        print(
            f"fold {fold} part {part} on {gpu}: task {count + 1}/{len(indices)} {record['task_id']}: "
            f"{PRIMARY_POLICY} error vs uncorrected {ratios} matches pilot {matches_pilot}",
            flush=True,
        )


# Report -----------------------------------------------------------------------


def ratio_interval(numerator: np.ndarray, denominator: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Bootstrap over tasks of sum(numerator) / sum(denominator)."""

    samples = rng.integers(0, len(numerator), size=(BOOTSTRAP_RESAMPLES, len(numerator)))
    ratios = numerator[samples].sum(axis=1) / denominator[samples].sum(axis=1)
    tail = 100 * (1 - INTERVAL) / 2
    return float(np.percentile(ratios, tail)), float(np.percentile(ratios, 100 - tail))


def report(output_dir: Path, task_count: int) -> dict[str, Any]:
    records = [json.loads(path.read_text()) for path in sorted((output_dir / "tasks").glob("task_*.json"))]
    if len(records) != task_count:
        raise SystemExit(f"{len(records)} diagnosed tasks, expected {task_count}. Finish run first.")
    rng = np.random.default_rng(0)

    def per_task(label: str, key: str, kind: int | None = None) -> np.ndarray:
        """Each task's sum over layers 1-27; kind picks keys (0) or values (1) of a K/V sum."""

        sums = [np.asarray(record["sums"][label][key])[SCORED_FIRST_LAYER:] for record in records]
        return np.array([part.sum() if kind is None else part[:, kind].sum() for part in sums])

    rows, comparisons = [], {}
    for name in POLICIES:
        base = f"{name}@0.0"
        for strength in STRENGTHS:
            label = f"{name}@{strength}"
            # Each ratio: (name, numerator, denominator). Below 1 means closer to dense.
            ratios = {
                "kv": (per_task(label, "kv_after"), per_task(base, "kv_after")),
                "key": (per_task(label, "kv_after", 0), per_task(base, "kv_after", 0)),
                "value": (per_task(label, "kv_after", 1), per_task(base, "kv_after", 1)),
                "attention": (per_task(label, "attention_error"), per_task(base, "attention_error")),
                # kv = fit x drift: how well the correction works on the cache it reads, and
                # how much the corrected run's cache has drifted from the uncorrected one's.
                "fit": (per_task(label, "kv_after"), per_task(label, "kv_before")),
                "drift": (per_task(label, "kv_before"), per_task(base, "kv_before")),
            }
            entry = {}
            for quantity, (numerator, denominator) in ratios.items():
                low, high = ratio_interval(numerator, denominator, rng)
                entry[quantity] = {"ratio": float(numerator.sum() / denominator.sum()), "interval_low": low, "interval_high": high}
                rows.append({"policy": name, "strength": strength, "quantity": quantity, **entry[quantity]})
            kv_closer = entry["kv"]["interval_high"] < 1
            attention_closer = entry["attention"]["interval_high"] < 1
            entry["explanation"] = (
                "compounding" if not kv_closer else "not_decision_relevant" if not attention_closer else "residual"
            )
            comparisons[label] = entry

    layer_rows = []
    for name in POLICIES:
        for strength in STRENGTHS:
            for layer in range(SCORED_FIRST_LAYER, len(records[0]["sums"][f"{name}@0.0"]["attention_error"])):
                total = {
                    label: {key: sum(np.asarray(record["sums"][label][key])[layer] for record in records) for key in SUM_KEYS}
                    for label in (f"{name}@0.0", f"{name}@{strength}")
                }
                corrected, base = total[f"{name}@{strength}"], total[f"{name}@0.0"]
                layer_rows.append({
                    "policy": name, "strength": strength, "layer": layer,
                    "key_ratio": float(corrected["kv_after"][0] / base["kv_after"][0]),
                    "value_ratio": float(corrected["kv_after"][1] / base["kv_after"][1]),
                    "attention_ratio": float(corrected["attention_error"] / base["attention_error"]),
                    "uncorrected_kv_relative_error": float(base["kv_after"].sum() / base["kv_dense"].sum()),
                    "uncorrected_attention_relative_error": float(base["attention_error"] / base["attention_dense"]),
                })

    # Paired change in divergence events per task, as in the pilot (recorded since the layer-4 test).
    divergence = {}
    if "metrics" in records[0]:
        for name in POLICIES:
            base = np.array([record["metrics"][f"{name}@0.0"]["divergence_events"] for record in records])
            for strength in STRENGTHS:
                change = np.array([record["metrics"][f"{name}@{strength}"]["divergence_events"] for record in records]) - base
                low, high = paired_interval(change, rng, INTERVAL)
                divergence[f"{name}@{strength}"] = {
                    "uncorrected_mean": float(base.mean()), "change_mean": float(change.mean()),
                    "interval_low": low, "interval_high": high,
                }
        comparisons_primary = [divergence[f"{PRIMARY_POLICY}@{strength}"] for strength in STRENGTHS]

    # Attention probe, per probed run and layer, as means per query row and head.
    probe_rows = []
    for label in records[0].get("probe", {}):
        totals = {key: sum(np.asarray(record["probe"][label][key]) for record in records) for key in PROBE_KEYS}
        for layer in range(SCORED_FIRST_LAYER, len(totals["queries"])):
            t = {key: float(value[layer]) for key, value in totals.items()}
            if t["queries"] == 0:
                continue
            probe_rows.append({
                "run": label, "layer": layer,
                "top_share": t["top_share"] / t["queries"],
                "first_share": t["first_share"] / t["queries"],
                "first_is_top": t["first_is_top"] / t["queries"],
                "change": t["change"] / t["queries"],
                "change_at_top_fraction": t["change_at_top"] / t["change"],
                "change_at_first_fraction": t["change_at_first"] / t["change"],
                "key_part_fraction": t["key_part"] / (t["key_part"] + t["value_part"]),
                "relative_output_change": t["total_change"] / t["output"],
            })

    checked = [record["matches_pilot"] for record in records if record["matches_pilot"] is not None]
    summary = {
        "primary_policy": PRIMARY_POLICY,
        "interval": INTERVAL,
        "skipped_layers": records[0].get("skipped_layers", []),
        "explanation": {strength: comparisons[f"{PRIMARY_POLICY}@{strength}"]["explanation"] for strength in STRENGTHS},
        "pipeline_passed": bool(checked) and all(checked),
        "tasks_checked_against_pilot": len(checked),
        "comparisons": comparisons,
        "divergence_events": divergence,
    }
    if divergence:
        summary["success"] = any(entry["interval_high"] < 0 for entry in comparisons_primary)
    if probe_rows:
        path = output_dir / "attention_probe_by_layer.csv"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(",".join(probe_rows[0]) + "\n")
            handle.writelines(",".join(str(value) for value in row.values()) + "\n" for row in probe_rows)
    (output_dir / "diagnostic_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for path, table in ((output_dir / "diagnostic_results.csv", rows), (output_dir / "diagnostic_by_layer.csv", layer_rows)):
        with path.open("w", encoding="utf-8") as handle:
            handle.write(",".join(table[0]) + "\n")
            handle.writelines(",".join(str(value) for value in row.values()) + "\n" for row in table)

    print(f"{'policy':<12} {'strength':>8} {'quantity':<10} {'ratio':>7} {f'{INTERVAL:.1%} interval':>18}")
    for row in rows:
        print(
            f"{row['policy']:<12} {row['strength']:>8} {row['quantity']:<10} {row['ratio']:>7.3f} "
            f"[{row['interval_low']:>7.3f}, {row['interval_high']:>7.3f}]"
        )
    for label, entry in divergence.items():
        print(
            f"divergence events {label:<16} uncorrected {entry['uncorrected_mean']:6.2f} change {entry['change_mean']:+6.2f} "
            f"[{entry['interval_low']:+6.2f}, {entry['interval_high']:+6.2f}]"
        )
    for row in probe_rows:
        if row["run"] == f"{PRIMARY_POLICY}@1.0_full":
            print(
                f"probe {row['run']} layer {row['layer']:2d}: top share {row['top_share']:.2f}, first is top {row['first_is_top']:.2f}, "
                f"change {row['change']:.3f} ({row['change_at_top_fraction']:.0%} at top row), key part {row['key_part_fraction']:.0%}, "
                f"output change {row['relative_output_change']:.3f}"
            )
    print(f"Runs match the pilot exactly on {len(checked)} tasks run on the pilot's GPU type: {summary['pipeline_passed']}")
    print(f"Explanation at {PRIMARY_POLICY}: {summary['explanation']}")
    if divergence:
        print(f"Success ({PRIMARY_POLICY} divergence events fall at some strength): {summary['success']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("run", "report"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(FOLDS), help="Required for run")
    parser.add_argument("--part", type=int, choices=range(PARTS), help="Required for run")
    parser.add_argument("--skip-layers", type=int, nargs="+", default=[], help="Layers left uncorrected (the layer-4 test: 4)")
    args = parser.parse_args()
    config = load_kv_policy_experiment_config(args.config)
    skipped_layers = tuple(args.skip_layers)
    if args.step == "run":
        if args.fold is None or args.part is None:
            parser.error("run needs --fold and --part")
        run(config, args.fold, args.part, skipped_layers)
    else:
        report(diagnostic_dir(config, skipped_layers), config.evaluation.max_prompts)


if __name__ == "__main__":
    main()
