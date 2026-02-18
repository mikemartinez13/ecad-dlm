import inspect
import sys
from typing import Callable, Iterator

from ecad.schedulers.dit_scheduler.dream_dit_schedule import DreamDiTSchedule


def gen_default(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[DreamDiTSchedule]:
    schedule_dict = {step: {} for step in range(num_inference_steps)}
    yield DreamDiTSchedule(
        num_blocks=num_blocks,
        num_inference_steps=num_inference_steps,
        name="default",
        schedule=schedule_dict,
    )


def get_gen_functions() -> dict[str, Callable[..., Iterator[DreamDiTSchedule]]]:
    current_module = sys.modules[__name__]
    return {
        name: obj
        for name, obj in inspect.getmembers(current_module, inspect.isfunction)
        if name.startswith("gen_")
    }


GEN_FUNCTIONS = get_gen_functions()
