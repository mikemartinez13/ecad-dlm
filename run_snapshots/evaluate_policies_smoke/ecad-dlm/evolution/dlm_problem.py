"""NSGA-II decisions for grouped attention/MLP residual-output-cache schedules.

The module and class names remain stable for historical serialized checkpoints.
"""

from __future__ import annotations

from math import ceil

import numpy as np
import numpy.typing as npt
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling


class DLMCachingScheduleProblem(ElementwiseProblem):
    """Two-objective ECAD problem evaluated through pymoo's ask/tell API."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_inference_steps: int,
        step_group_size: int,
        layer_group_size: int,
        min_cached_groups: int = 1,
        enforce_syntax_validity: bool = False,
        max_syntax_invalid: int = 0,
    ) -> None:
        self.num_layers = num_layers
        self.num_inference_steps = num_inference_steps
        self.step_group_size = step_group_size
        self.layer_group_size = layer_group_size
        self.min_cached_groups = min_cached_groups
        self.enforce_syntax_validity = enforce_syntax_validity
        self.max_syntax_invalid = max_syntax_invalid
        self.genome_shape = (
            ceil((num_inference_steps - 1) / step_group_size),
            ceil(num_layers / layer_group_size),
            2,
        )
        n_var = int(np.prod(self.genome_shape))
        if n_var < 1:
            raise ValueError("DLM genome must contain at least one decision")
        if not 0 <= min_cached_groups <= n_var:
            raise ValueError("min_cached_groups must be between zero and n_var")
        if max_syntax_invalid < 0:
            raise ValueError("max_syntax_invalid must be non-negative")
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_ieq_constr=2 if enforce_syntax_validity else 1,
            xl=np.zeros(n_var),
            xu=np.ones(n_var),
            vtype=np.bool_,
        )

    def constraint_violations(
        self,
        population: npt.NDArray[np.bool_],
        syntax_invalid_counts: npt.NDArray[np.int64] | None = None,
    ) -> npt.NDArray[np.int64]:
        cached_groups = np.sum(~population.astype(np.bool_), axis=1)
        structural = (self.min_cached_groups - cached_groups).reshape(-1, 1)
        if not self.enforce_syntax_validity:
            return structural
        if syntax_invalid_counts is None:
            raise ValueError(
                "syntax_invalid_counts are required when syntax validity is enforced"
            )
        syntax = np.asarray(syntax_invalid_counts, dtype=np.int64).reshape(-1, 1)
        if len(syntax) != len(population):
            raise ValueError("syntax_invalid_counts must match population size")
        return np.column_stack(
            [structural, syntax - self.max_syntax_invalid]
        ).astype(np.int64)

    def _evaluate(self, x, out, *args, **kwargs):
        raise NotImplementedError("DLM schedules use persistent offline ask/tell evaluation")


class DLMGroupedRandomSampling(Sampling):
    """Random ECAD initialization biased toward conservative recomputation."""

    def __init__(
        self,
        recompute_probability: float = 0.8,
        min_cached_groups: int = 1,
        cached_groups: int | None = None,
    ) -> None:
        super().__init__()
        self.recompute_probability = recompute_probability
        self.min_cached_groups = min_cached_groups
        self.cached_groups = cached_groups

    def _do(self, problem, n_samples, **kwargs) -> npt.NDArray[np.bool_]:
        if self.cached_groups is not None:
            if not self.min_cached_groups <= self.cached_groups <= problem.n_var:
                raise ValueError(
                    "cached_groups must be between min_cached_groups and n_var"
                )
            population = np.ones((n_samples, problem.n_var), dtype=np.bool_)
            for row in population:
                selected = np.random.choice(
                    problem.n_var, size=self.cached_groups, replace=False
                )
                row[selected] = False
            return population

        population = (
            np.random.random((n_samples, problem.n_var))
            < self.recompute_probability
        )
        for row in population:
            cached = int((~row).sum())
            if cached < self.min_cached_groups:
                choices = np.flatnonzero(row)
                count = min(self.min_cached_groups - cached, len(choices))
                if count:
                    row[np.random.choice(choices, size=count, replace=False)] = False
        return population.astype(np.bool_)
