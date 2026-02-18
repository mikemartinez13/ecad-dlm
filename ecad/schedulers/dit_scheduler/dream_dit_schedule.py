import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ecad.schedulers.dit_scheduler.dit_schedule import DiTSchedule
from ecad.utils import ForwardArgs


class DreamDiTSchedule(DiTSchedule[ForwardArgs, Any]):
    """
    Minimal DiT schedule for Dream text generation.

    Dream's generation loop currently drives per-step execution directly through
    `diffusion_generate`, so this schedule is primarily used to keep the shared
    callback/scheduler plumbing in TextGenerator consistent with PixArt/FLUX.
    """

    def update_transformer_blocks(
        self, transformer_blocks: nn.ModuleList, **kwargs: Any
    ) -> None:
        self.transformer_blocks = transformer_blocks

    def forward(
        self,
        forward_args: ForwardArgs,
        inference_step: int,
        transformer_blocks: nn.ModuleList | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if transformer_blocks is not None:
            self.update_transformer_blocks(transformer_blocks, **kwargs)
        return forward_args.hidden_states

    def to_dict(self) -> dict[str, Any]:
        return {
            "dit_schedule": {
                "num_blocks": self.num_blocks,
                "num_inference_steps": self.num_inference_steps,
                "name": self.name,
                "attributes": self.attributes,
                "schedule": {
                    f"{step:03}": {}
                    for step in range(self.num_inference_steps)
                },
            },
            "config": self.top_level_config,
            "metrics": self.metrics,
        }

    @classmethod
    def from_json(cls, file_path: Path) -> "DreamDiTSchedule":
        with file_path.open("r") as f:
            data = json.load(f)

        if "dit_schedule" not in data:
            raise KeyError("dit_schedule")

        top_level_config = data.get("config", None)
        metrics = data.get("metrics", None)
        inner = data["dit_schedule"]

        num_blocks = inner["num_blocks"]
        num_inference_steps = inner["num_inference_steps"]
        name = inner["name"]
        attributes = inner.get("attributes", None)

        schedule_dict = {
            int(step_num): {}
            for step_num in inner.get("schedule", {}).keys()
        }
        if len(schedule_dict) == 0:
            schedule_dict = {i: {} for i in range(num_inference_steps)}

        return cls(
            num_blocks=num_blocks,
            num_inference_steps=num_inference_steps,
            name=name,
            schedule=schedule_dict,
            top_level_config=top_level_config,
            attributes=attributes,
            metrics=metrics,
        )
