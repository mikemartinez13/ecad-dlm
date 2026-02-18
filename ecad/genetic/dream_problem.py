import numpy as np
from pymoo.core.problem import ElementwiseProblem

from ecad.schedulers.cache_scheduler.dream_cache_schedule import Dream7bCacheSchedule
from ecad.schedulers.cache_scheduler.generators.dream_schedule_generators import (
    gen_default as cache_gen_default,
)


class Dream7bCachingScheduleProblem(ElementwiseProblem):
    """
    Genetic algorithm problem definition for optimizing a Dream 7B *diffusion* LM cache schedule.

    Decision variables:
      A binary vector x of length:
        num_inference_steps * num_layers * num_component_types

      with component order (MVP):
        0 -> "layer"
        1 -> "kv"
        2 -> "mlp"

    Objectives (minimize both):
      1) NLL: Negative log-likelihood of the gold answer tokens (lower is better)
      2) Cost: compute cost proxy (e.g., total_macs_T, latency, etc.) (lower is better)

    Constraint:
      - schedule must differ from the default by at least min_diff_from_default bits

    NOTE:
      _evaluate is not implemented because you are using pymoo's ask/tell API and
      doing offline evaluation (write eval JSONs, then load them via PopulationIOManager).
    """

    def __init__(
        self,
        num_inference_steps: int = 20,      # Dream diffusion denoise steps
        num_layers: int = 32,
        num_component_types: int = 3,       # MVP: ["layer", "kv", "mlp"]
        min_diff_from_default: int = 1,
        default_schedule: Dream7bCacheSchedule | None = None,
        **kwargs,
    ):
        self.num_inference_steps: int = num_inference_steps
        self.num_layers: int = num_layers
        self.num_component_types: int = num_component_types
        self.min_diff_from_default: int = min_diff_from_default

        # If no default schedule is provided, construct one from default generator.
        if default_schedule is None:
            default_schedule = next(
                cache_gen_default(num_layers, num_inference_steps)
            )

        # Flattened bool vector representation of the default schedule
        self.default_schedule = default_schedule.to_numpy(flatten=True)

        # Total decision variables
        self.n_var = num_inference_steps * num_layers * num_component_types

        if self.default_schedule.size != self.n_var:
            raise ValueError(
                f"Default schedule size {self.default_schedule.size} does not match n_var {self.n_var} "
                f"(expected {num_inference_steps}*{num_layers}*{num_component_types})."
            )

        # Binary decision variable bounds
        xl = np.zeros(self.n_var)
        xu = np.ones(self.n_var)

        # Two objectives:
        # 1) Minimize NLL (gold-answer NLL on your evaluation set)
        # 2) Minimize compute cost (MACs, latency, etc.)
        super().__init__(
            n_var=self.n_var,
            n_obj=2,
            n_ieq_constr=1,
            xl=xl,
            xu=xu,
            vtype=np.bool_,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        raise NotImplementedError(
            "Evaluation is performed offline via ask/tell; this Problem only defines "
            "the search space (binary vector) and the number of objectives/constraints."
        )
