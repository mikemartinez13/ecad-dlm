"""Small resumable NSGA-II search over structured DiffuCoder KV policies.

Objectives are (quality_loss, compute_fraction), both minimized. The quality
loss is chosen by ``evolution.quality_objective``:

- ``base_regression_rate``: generate every task end to end and run paired
  EvalPlus base tests (coarse: one bit per task).
- ``replay_divergence``: forced-trajectory replay; divergence events per task
  (see forced_trajectory_replay.py). The whole generation shares one dense
  pass, and no code is executed. It failed its recheck (EAR_VALIDITY_STUDY.md),
  so it is kept only to read old runs.

``constrained_kv_policy_search.py`` reuses the genome and selection here with one
objective, predicted batched time, under Deb's constrained domination.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from dlm_kv_policy.kv_policy_experiment_config import KVPolicyNSGA2SearchConfig, KVPolicyExperimentConfig
from dlm_kv_policy.dkv_decode_refresh_policy import DKVDecodeRefreshPolicy
from dlm_kv_policy.masked_row_reuse_policy import MaskedRowReusePolicy
from dlm_kv_policy.code_correctness_and_latency_evaluator import DiffuCoderKVPolicyEvaluator
from dlm_kv_policy.forced_trajectory_replay import replay_policies, summarize


@dataclass(frozen=True)
class KVRefreshPolicyGenome:
    delay: int
    refresh_interval: int
    refresh_depth: int
    terminal_refresh: int
    masked_window: int = 1
    protected_size: int = 32
    protected_rule: str = "split+recent"
    prompt_refresh_interval: int = 0
    local_refresh_radius: int = 0
    padding_window: int = 0
    entropy_threshold: float = 0.0
    entropy_scope: str = "local"

    @property
    def wrapped(self) -> bool:
        """Whether the policy needs MaskedRowReusePolicy around the base."""

        return self.masked_window > 1 or self.padding_window > 0 or self.entropy_threshold > 0

    @property
    def name(self) -> str:
        name = f"ga_d{self.delay}_r{self.refresh_interval}_l{self.refresh_depth}_t{self.terminal_refresh}"
        if self.prompt_refresh_interval:
            name += f"_p{self.prompt_refresh_interval}"
        if self.local_refresh_radius:
            name += f"_n{self.local_refresh_radius}"
        if self.masked_window > 1 or self.padding_window:
            name += f"_w{self.masked_window}_m{self.protected_size}_{self.protected_rule}"
        if self.padding_window:
            name += f"_pad{self.padding_window}"
        if self.entropy_threshold:
            name += f"_h{self.entropy_threshold:g}{self.entropy_scope}"
        return name

    def canonical(self) -> "KVRefreshPolicyGenome":
        """Reset genes that have no effect, so one policy has one genome."""

        values = self.to_dict()
        if self.masked_window == 1 and not self.padding_window:
            values["protected_size"], values["protected_rule"] = 32, "split+recent"
        if not self.entropy_threshold:
            values["entropy_scope"] = "local"
        return KVRefreshPolicyGenome(**values)

    def policy(self, stop_ids: set[int] | None = None) -> Any:
        base = DKVDecodeRefreshPolicy(
            name=self.name,
            additional_decoded_forwards=self.delay,
            cache_reloading_step=self.refresh_interval,
            refresh_depth=self.refresh_depth,
            terminal_full_refresh_steps=self.terminal_refresh,
            prompt_refresh_interval=self.prompt_refresh_interval or None,
            local_refresh_radius=self.local_refresh_radius,
        )
        if not self.wrapped:
            return base
        if stop_ids is None:
            raise ValueError("Masked reuse and the entropy trigger need the tokenizer's stop ids")
        policy = MaskedRowReusePolicy(base, self.masked_window, self.protected_size, self.protected_rule, stop_ids,
                                      self.padding_window, self.entropy_threshold, self.entropy_scope)
        policy.name = self.name
        return policy

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "KVRefreshPolicyGenome":
        kinds = {"int": int, "float": float, "str": str}
        return cls(**{item.name: kinds[item.type](raw.get(item.name, item.default)) for item in fields(cls)})


@dataclass(frozen=True)
class EvaluatedKVPolicy:
    genome: KVRefreshPolicyGenome
    quality_loss: float  # base regression rate, or replay divergence events per task
    compute_fraction: float  # policy FLOPs / dense FLOPs; the speed objective
    # Recorded only with the EvalPlus objective, for information.
    wall_time_seconds: float | None = None
    candidate_pass_at_1: float | None = None
    syntax_invalid_count: int | None = None
    output_hash: str = ""
    same_output_as: str | None = None
    # Constrained search only: its one objective, and the total constraint shortfall.
    predicted_seconds: float | None = None
    violation: float = 0.0
    constraints: dict[str, Any] | None = None

    @property
    def objectives(self) -> tuple[float, ...]:
        if self.predicted_seconds is not None:
            return (self.predicted_seconds,)
        return (self.quality_loss, self.compute_fraction)

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome": self.genome.to_dict(),
            "policy_name": self.genome.name,
            "objectives": {
                "quality_loss": self.quality_loss,
                "compute_fraction": self.compute_fraction,
                "predicted_seconds": self.predicted_seconds,
            },
            "violation": self.violation,
            "constraints": self.constraints,
            "median_generation_wall_time": self.wall_time_seconds,
            "candidate_base_pass_at_1": self.candidate_pass_at_1,
            "syntax_invalid_count": self.syntax_invalid_count,
            "output_hash": self.output_hash,
            "same_output_as": self.same_output_as,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvaluatedKVPolicy":
        objectives = raw["objectives"]
        return cls(
            genome=KVRefreshPolicyGenome.from_dict(raw["genome"]),
            quality_loss=float(objectives["quality_loss"]),
            compute_fraction=float(objectives["compute_fraction"]),
            wall_time_seconds=raw.get("median_generation_wall_time"),
            candidate_pass_at_1=raw.get("candidate_base_pass_at_1"),
            syntax_invalid_count=raw.get("syntax_invalid_count"),
            output_hash=raw.get("output_hash", ""),
            same_output_as=raw.get("same_output_as"),
            predicted_seconds=objectives.get("predicted_seconds"),
            violation=float(raw.get("violation", 0.0)),
            constraints=raw.get("constraints"),
        )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _dominates(left: EvaluatedKVPolicy, right: EvaluatedKVPolicy) -> bool:
    """Deb's constrained domination: smaller violation first, then Pareto on the objectives."""

    if left.violation != right.violation:
        return left.violation < right.violation
    pairs = list(zip(left.objectives, right.objectives))
    return all(a <= b for a, b in pairs) and any(a < b for a, b in pairs)


def non_dominated_fronts(items: list[EvaluatedKVPolicy]) -> list[list[int]]:
    """Return NSGA-II fronts; the population is intentionally tiny."""

    remaining = list(range(len(items)))
    fronts: list[list[int]] = []
    while remaining:
        front = [
            index
            for index in remaining
            if not any(_dominates(items[other], items[index]) for other in remaining if other != index)
        ]
        fronts.append(front)
        remaining = [index for index in remaining if index not in front]
    return fronts


def crowding_distance(items: list[EvaluatedKVPolicy], front: list[int]) -> dict[int, float]:
    distances = {index: 0.0 for index in front}
    if len(front) <= 2:
        return {index: float("inf") for index in front}
    for position in range(len(items[front[0]].objectives)):
        def objective(item: EvaluatedKVPolicy) -> float:
            return item.objectives[position]

        ordered = sorted(front, key=lambda index: objective(items[index]))
        distances[ordered[0]] = distances[ordered[-1]] = float("inf")
        low, high = objective(items[ordered[0]]), objective(items[ordered[-1]])
        if high == low:
            continue
        for previous, current, following in zip(ordered, ordered[1:-1], ordered[2:]):
            if distances[current] != float("inf"):
                distances[current] += (objective(items[following]) - objective(items[previous])) / (high - low)
    return distances


def _rank_and_crowding(items: list[EvaluatedKVPolicy]) -> tuple[dict[int, int], dict[int, float]]:
    ranks: dict[int, int] = {}
    distances: dict[int, float] = {}
    for rank, front in enumerate(non_dominated_fronts(items)):
        ranks.update({index: rank for index in front})
        distances.update(crowding_distance(items, front))
    return ranks, distances


def environmental_selection(items: list[EvaluatedKVPolicy], size: int) -> list[EvaluatedKVPolicy]:
    selected: list[EvaluatedKVPolicy] = []
    for front in non_dominated_fronts(items):
        if len(selected) + len(front) <= size:
            selected.extend(items[index] for index in front)
            continue
        distances = crowding_distance(items, front)
        ordered = sorted(
            front,
            key=lambda index: (-distances[index], tuple(map(str, items[index].genome.to_dict().values()))),
        )
        selected.extend(items[index] for index in ordered[: size - len(selected)])
        break
    return selected


# Known controls, which seed every search.
SEED_GENOMES = (
    KVRefreshPolicyGenome(1, 8, 1, 0),      # dKV-Decode reference
    KVRefreshPolicyGenome(1, 16, 1, 0),     # speed-oriented fixed control
    KVRefreshPolicyGenome(1, 8, 22, 0),     # depth-22 fixed control
    KVRefreshPolicyGenome(1, 16, 22, 0),    # observed interaction control
    KVRefreshPolicyGenome(1, 8, 22, 0, 2),  # masked reuse on depth_22, checked end to end
    KVRefreshPolicyGenome(1, 8, 22, 0, 4),
    KVRefreshPolicyGenome(1, 8, 22, 0, 8),
    KVRefreshPolicyGenome(1, 8, 24, 0, 4),
)


def _random_genome(config: KVPolicyNSGA2SearchConfig, rng: random.Random) -> KVRefreshPolicyGenome:
    return KVRefreshPolicyGenome(*[rng.choice(values) for values in config.choices()]).canonical()


def initial_population(config: KVPolicyNSGA2SearchConfig) -> list[KVRefreshPolicyGenome]:
    """Seed known controls, or a finished run's survivors, then fill with reproducible diversity."""

    allowed = [set(values) for values in config.choices()]

    def fits(genome: KVRefreshPolicyGenome) -> bool:
        return all(value in choices for value, choices in zip(genome.to_dict().values(), allowed))

    if config.initial_population:
        summary = json.loads(Path(config.initial_population).read_text())
        population = [KVRefreshPolicyGenome.from_dict(raw["genome"]) for raw in summary["population"]]
        outside = [genome.name for genome in population if not fits(genome)]
        if outside:
            raise ValueError(f"Starting genomes outside this config's choices: {outside}")
    else:
        population = [genome for genome in SEED_GENOMES if fits(genome)]
    rng = random.Random(config.seed)
    while len(population) < config.population_size:
        candidate = _random_genome(config, rng)
        if candidate not in population:
            population.append(candidate)
    return population


def _tournament(
    population: list[EvaluatedKVPolicy], ranks: dict[int, int], distances: dict[int, float], rng: random.Random
) -> EvaluatedKVPolicy:
    left, right = rng.sample(range(len(population)), 2)
    left_key = (ranks[left], -distances[left])
    right_key = (ranks[right], -distances[right])
    return population[left] if left_key < right_key else population[right]


def _crossover(left: KVRefreshPolicyGenome, right: KVRefreshPolicyGenome, config: KVPolicyNSGA2SearchConfig, rng: random.Random) -> KVRefreshPolicyGenome:
    if rng.random() > config.crossover_probability:
        return left
    return KVRefreshPolicyGenome(*[
        rng.choice((first, second)) for first, second in zip(left.to_dict().values(), right.to_dict().values())
    ]).canonical()


def _mutate(genome: KVRefreshPolicyGenome, config: KVPolicyNSGA2SearchConfig, rng: random.Random) -> KVRefreshPolicyGenome:
    values = list(genome.to_dict().values())
    choices = config.choices()
    mutable = [index for index, options in enumerate(choices) if len(set(options)) > 1]
    count = 2 if rng.random() < config.two_gene_mutation_probability else 1
    for index in rng.sample(mutable, min(count, len(mutable))):
        options = [value for value in choices[index] if value != values[index]]
        values[index] = rng.choice(options)
    return KVRefreshPolicyGenome(*values).canonical()


def offspring_population(
    population: list[EvaluatedKVPolicy],
    config: KVPolicyNSGA2SearchConfig,
    generation: int,
    *,
    forbidden: set[KVRefreshPolicyGenome] | None = None,
) -> list[KVRefreshPolicyGenome]:
    rng = random.Random(config.seed + generation)
    ranks, distances = _rank_and_crowding(population)
    blocked = {item.genome for item in population} | (forbidden or set())
    children: list[KVRefreshPolicyGenome] = []
    attempts = 0
    while len(children) < config.population_size:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("Evolution search space exhausted before forming a unique population")
        child = _mutate(
            _crossover(
                _tournament(population, ranks, distances, rng).genome,
                _tournament(population, ranks, distances, rng).genome,
                config,
                rng,
            ),
            config,
            rng,
        )
        if child not in blocked and child not in children:
            children.append(child)
    return children


def _signature(config: KVPolicyExperimentConfig, evaluator: DiffuCoderKVPolicyEvaluator) -> str:
    payload = {
        "evaluator_signature": evaluator.signature,
        "evolution": asdict(config.evolution),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _evaluate_with_evalplus(
    evaluator: DiffuCoderKVPolicyEvaluator,
    genomes: list[KVRefreshPolicyGenome],
    *,
    baseline: Any,
    baseline_execution: dict[str, Any],
) -> tuple[dict[str, Any], list[EvaluatedKVPolicy]]:
    records = evaluator.evaluate_evolution_policies(
        [genome.policy() for genome in genomes],
        baseline=baseline,
        baseline_execution=baseline_execution,
    )
    individuals = [
        EvaluatedKVPolicy(
            genome=genome,
            quality_loss=float(record["quality_loss"]),
            compute_fraction=float(record["compute_fraction"]),
            wall_time_seconds=float(record["median_elapsed_seconds"]),
            candidate_pass_at_1=float(record["candidate_base_pass_at_1"]),
            syntax_invalid_count=int(record["syntax_invalid_count"]),
            output_hash=record["output_hash"],
            same_output_as=record.get("same_output_as"),
        )
        for genome, record in zip(genomes, records)
    ]
    return baseline.record, individuals


def _evaluate_with_replay(
    evaluator: DiffuCoderKVPolicyEvaluator, genomes: list[KVRefreshPolicyGenome], replay_dir: Path
) -> list[EvaluatedKVPolicy]:
    """Score a whole population with one forced-trajectory replay."""

    policies = {genome.policy().name: genome.policy() for genome in genomes}
    config = evaluator.config
    tasks = replay_policies(
        evaluator.model,
        evaluator.tokenizer,
        evaluator.tasks,
        policies,
        config.generation,
        config.model.device,
        replay_dir / "tasks",
    )
    results = summarize(tasks, list(policies))
    return [
        EvaluatedKVPolicy(
            genome=genome,
            quality_loss=results[genome.policy().name]["divergence_events_per_task"],
            compute_fraction=results[genome.policy().name]["compute_fraction"],
        )
        for genome in genomes
    ]


def run_kv_policy_nsga2_search(
    config: KVPolicyExperimentConfig,
    evaluator: DiffuCoderKVPolicyEvaluator,
    *,
    reuse_baseline: Path | None,
    generations_this_job: int,
) -> dict[str, Any]:
    """Evaluate up to ``generations_this_job`` populations and checkpoint each."""

    if config.evolution is None:
        raise ValueError("The selected config has no evolution section")
    if generations_this_job < 1:
        raise ValueError("generations_this_job must be positive")
    settings = config.evolution
    if settings.quality_objective == "constrained_batched_time":
        raise ValueError("Run constrained_batched_time with evolution/constrained_kv_policy_search.py")
    root = config.run.run_dir / "evolution"
    state_path = root / "state.json"
    signature = _signature(config, evaluator)
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature") != signature:
            raise RuntimeError("Evolution checkpoint does not match this config or evaluator")
        population = [EvaluatedKVPolicy.from_dict(item) for item in state["population"]]
        seen = {KVRefreshPolicyGenome.from_dict(item) for item in state.get("seen_genomes", [])}
        completed = int(state["completed_generations"])
    else:
        population = []
        seen = set()
        completed = 0

    last_summary: dict[str, Any] | None = None
    baseline = None
    baseline_execution = None
    baseline_summary = None
    use_replay = settings.quality_objective == "replay_divergence"
    if completed < settings.generations and not use_replay:
        baseline, baseline_execution, baseline_summary = evaluator.prepare_evolution_baseline(
            reuse_baseline
        )
    while completed < settings.generations and (last_summary is None or last_summary["jobs_run"] < generations_this_job):
        generation = completed
        genomes = (
            initial_population(settings)
            if generation == 0
            else offspring_population(population, settings, generation, forbidden=seen)
        )
        if use_replay:
            evaluated = _evaluate_with_replay(evaluator, genomes, root / "replay" / f"generation_{generation:03d}")
        else:
            _, evaluated = _evaluate_with_evalplus(
                evaluator,
                genomes,
                baseline=baseline,
                baseline_execution=baseline_execution,
            )
        seen.update(genomes)
        population = evaluated if generation == 0 else environmental_selection(population + evaluated, settings.population_size)
        pareto = [population[index] for index in non_dominated_fronts(population)[0]]
        summary = {
            "generation": generation,
            "quality_objective": settings.quality_objective,
            "jobs_run": 1 if last_summary is None else last_summary["jobs_run"] + 1,
            "baseline": baseline_summary,
            "evaluated": [item.to_dict() for item in evaluated],
            "population": [item.to_dict() for item in population],
            "pareto": [item.to_dict() for item in pareto],
        }
        _write_json(root / "generations" / f"generation_{generation:03d}.json", summary)
        completed += 1
        state = {
            "signature": signature,
            "completed_generations": completed,
            "population": [item.to_dict() for item in population],
            "seen_genomes": [genome.to_dict() for genome in sorted(seen, key=lambda item: item.name)],
        }
        _write_json(state_path, state)
        last_summary = summary

    result = {
        "completed_generations": completed,
        "target_generations": settings.generations,
        "complete": completed == settings.generations,
        "last_generation": last_summary,
    }
    _write_json(root / "summary.json", result)
    return result
