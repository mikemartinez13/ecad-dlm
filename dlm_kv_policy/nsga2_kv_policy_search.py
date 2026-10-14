"""Small resumable NSGA-II search over structured DiffuCoder KV policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from .kv_policy_experiment_config import KVPolicyNSGA2SearchConfig, KVPolicyExperimentConfig
from .dkv_decode_refresh_policy import DKVDecodeRefreshPolicy, IMPLEMENTATION_REVISION as POLICY_IMPLEMENTATION_REVISION
from .code_correctness_and_latency_evaluator import DiffuCoderKVPolicyEvaluator


IMPLEMENTATION_REVISION = "categorical_nsga2_fast_selection_v2"


@dataclass(frozen=True)
class KVRefreshPolicyGenome:
    delay: int
    refresh_interval: int
    refresh_depth: int
    terminal_refresh: int

    def policy(self) -> DKVDecodeRefreshPolicy:
        return DKVDecodeRefreshPolicy(
            name=f"ga_d{self.delay}_r{self.refresh_interval}_l{self.refresh_depth}_t{self.terminal_refresh}",
            additional_decoded_forwards=self.delay,
            cache_reloading_step=self.refresh_interval,
            refresh_depth=self.refresh_depth,
            terminal_full_refresh_steps=self.terminal_refresh,
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "KVRefreshPolicyGenome":
        return cls(**{name: int(raw[name]) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class EvaluatedKVPolicy:
    genome: KVRefreshPolicyGenome
    regression: float
    wall_time_seconds: float
    candidate_pass_at_1: float
    syntax_invalid_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome": self.genome.to_dict(),
            "policy_name": self.genome.policy().name,
            "objectives": {
                "base_regression_rate": self.regression,
                "median_generation_wall_time": self.wall_time_seconds,
            },
            "candidate_base_pass_at_1": self.candidate_pass_at_1,
            "syntax_invalid_count": self.syntax_invalid_count,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvaluatedKVPolicy":
        objectives = raw["objectives"]
        return cls(
            genome=KVRefreshPolicyGenome.from_dict(raw["genome"]),
            regression=float(objectives["base_regression_rate"]),
            wall_time_seconds=float(objectives["median_generation_wall_time"]),
            candidate_pass_at_1=float(raw["candidate_base_pass_at_1"]),
            syntax_invalid_count=int(raw["syntax_invalid_count"]),
        )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _dominates(left: EvaluatedKVPolicy, right: EvaluatedKVPolicy) -> bool:
    no_worse = left.regression <= right.regression and left.wall_time_seconds <= right.wall_time_seconds
    return no_worse and (
        left.regression < right.regression or left.wall_time_seconds < right.wall_time_seconds
    )


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
    for objective in (lambda item: item.regression, lambda item: item.wall_time_seconds):
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
            key=lambda index: (-distances[index], tuple(items[index].genome.to_dict().values())),
        )
        selected.extend(items[index] for index in ordered[: size - len(selected)])
        break
    return selected


def _choices(config: KVPolicyNSGA2SearchConfig) -> tuple[list[int], list[int], list[int], list[int]]:
    return (
        config.delays,
        config.refresh_intervals,
        config.refresh_depths,
        config.terminal_refresh_steps,
    )


def _random_genome(config: KVPolicyNSGA2SearchConfig, rng: random.Random) -> KVRefreshPolicyGenome:
    return KVRefreshPolicyGenome(*[rng.choice(values) for values in _choices(config)])


def initial_population(config: KVPolicyNSGA2SearchConfig) -> list[KVRefreshPolicyGenome]:
    """Seed known controls, then fill the population with reproducible diversity."""

    allowed = [set(values) for values in _choices(config)]
    seeds = [
        KVRefreshPolicyGenome(1, 8, 1, 0),      # dKV-Decode reference
        KVRefreshPolicyGenome(1, 16, 1, 0),     # speed-oriented fixed control
        KVRefreshPolicyGenome(1, 8, 22, 0),     # zero-regression fixed control
        KVRefreshPolicyGenome(1, 16, 22, 0),    # observed interaction control
    ]
    population = [
        genome
        for genome in seeds
        if all(value in choices for value, choices in zip(genome.to_dict().values(), allowed))
    ]
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
    ])


def _mutate(genome: KVRefreshPolicyGenome, config: KVPolicyNSGA2SearchConfig, rng: random.Random) -> KVRefreshPolicyGenome:
    values = list(genome.to_dict().values())
    mutable = [index for index, options in enumerate(_choices(config)) if len(set(options)) > 1]
    count = 2 if rng.random() < config.two_gene_mutation_probability else 1
    for index in rng.sample(mutable, min(count, len(mutable))):
        options = [value for value in _choices(config)[index] if value != values[index]]
        values[index] = rng.choice(options)
    return KVRefreshPolicyGenome(*values)


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
        "revision": IMPLEMENTATION_REVISION,
        "policy_revision": POLICY_IMPLEMENTATION_REVISION,
        "evaluator_signature": evaluator.signature,
        "evolution": asdict(config.evolution),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _evaluate(
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
            regression=float(record["quality_loss"]),
            wall_time_seconds=float(record["median_elapsed_seconds"]),
            candidate_pass_at_1=float(record["candidate_base_pass_at_1"]),
            syntax_invalid_count=int(record["syntax_invalid_count"]),
        )
        for genome, record in zip(genomes, records)
    ]
    return baseline.record, individuals


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
    if completed < settings.generations:
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
        _, evaluated = _evaluate(
            evaluator,
            genomes,
            baseline=baseline,
            baseline_execution=baseline_execution,
        )
        seen.update(genomes)
        population = evaluated if generation == 0 else environmental_selection(population + evaluated, settings.population_size)
        pareto = [population[index] for index in non_dominated_fronts(population)[0]]
        summary = {
            "implementation_revision": IMPLEMENTATION_REVISION,
            "generation": generation,
            "jobs_run": 1 if last_summary is None else last_summary["jobs_run"] + 1,
            "baseline": baseline_summary,
            "evaluated": [item.to_dict() for item in evaluated],
            "population": [item.to_dict() for item in population],
            "pareto": [item.to_dict() for item in pareto],
        }
        _write_json(root / "generations" / f"generation_{generation:03d}.json", summary)
        completed += 1
        state = {
            "implementation_revision": IMPLEMENTATION_REVISION,
            "signature": signature,
            "completed_generations": completed,
            "population": [item.to_dict() for item in population],
            "seen_genomes": [genome.to_dict() for genome in sorted(seen, key=lambda item: tuple(item.to_dict().values()))],
        }
        _write_json(state_path, state)
        last_summary = summary

    result = {
        "implementation_revision": IMPLEMENTATION_REVISION,
        "completed_generations": completed,
        "target_generations": settings.generations,
        "complete": completed == settings.generations,
        "last_generation": last_summary,
    }
    _write_json(root / "summary.json", result)
    return result
