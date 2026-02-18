import inspect
import sys
from typing import Callable, Iterator

from ecad.schedulers.cache_scheduler.dream_cache_schedule import (
    Dream7bCacheSchedule,
)


def same_for_all_layers_one_step(
    num_layers: int,
    layer: bool,
    kv: bool,
    mlp: bool,
) -> dict[str, dict[str, bool]]:
    return {
        str(layer_idx): {
            "layer": layer,
            "kv": kv,
            "mlp": mlp,
        }
        for layer_idx in range(num_layers)
    }


def default_one_step(num_layers: int) -> dict[str, dict[str, bool]]:
    # Baseline: recompute all components for every layer.
    return same_for_all_layers_one_step(
        num_layers=num_layers,
        layer=True,
        kv=True,
        mlp=True,
    )


def default_all_timesteps(
    num_layers: int,
    num_inference_steps: int,
) -> dict[int, dict[str, dict[str, bool]]]:
    return {
        step: default_one_step(num_layers)
        for step in range(num_inference_steps)
    }


def gen_default(
    num_layers: int,
    num_inference_steps: int,
) -> Iterator[Dream7bCacheSchedule]:
    yield Dream7bCacheSchedule(
        num_blocks=num_layers,
        num_inference_steps=num_inference_steps,
        name="default",
        schedule=default_all_timesteps(num_layers, num_inference_steps),
        attributes={},
    )


def helper_recompute_every_n(
    num_layers: int,
    num_inference_steps: int,
    always_layer: bool,
    always_kv: bool,
    always_mlp: bool,
    name_prefix: str,
) -> Iterator[Dream7bCacheSchedule]:
    for n in range(2, num_inference_steps + 1):
        schedule = {}
        num_affected_steps = 0
        num_affected_layers = 0

        for step in range(num_inference_steps):
            recompute = step % n == 0
            schedule[step] = same_for_all_layers_one_step(
                num_layers=num_layers,
                layer=(recompute or always_layer),
                kv=(recompute or always_kv),
                mlp=(recompute or always_mlp),
            )
            if recompute:
                num_affected_steps += 1
                num_affected_layers = num_layers

        yield Dream7bCacheSchedule(
            num_blocks=num_layers,
            num_inference_steps=num_inference_steps,
            name=f"{name_prefix}_every_{n:03}",
            schedule=schedule,
            attributes={
                "num_affected_layers": num_affected_layers,
                "num_affected_steps": num_affected_steps,
                "recompute_layer_every": n if not always_layer else 1,
                "recompute_kv_every": n if not always_kv else 1,
                "recompute_mlp_every": n if not always_mlp else 1,
            },
        )


def gen_recompute_all_every_n(
    num_layers: int,
    num_inference_steps: int,
) -> Iterator[Dream7bCacheSchedule]:
    yield from helper_recompute_every_n(
        num_layers,
        num_inference_steps,
        always_layer=False,
        always_kv=False,
        always_mlp=False,
        name_prefix="recompute_all",
    )


def get_gen_functions() -> dict[str, Callable[..., Iterator[Dream7bCacheSchedule]]]:
    current_module = sys.modules[__name__]
    return {
        name: obj
        for name, obj in inspect.getmembers(current_module, inspect.isfunction)
        if name.startswith("gen_")
    }


GEN_FUNCTIONS = get_gen_functions()
