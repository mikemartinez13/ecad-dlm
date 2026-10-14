"""Residual-branch caching hooks for DiffuCoder's Dream decoder layers."""

from __future__ import annotations

from types import MethodType
from typing import Any, Callable

import torch

from ecad.language_models.residual_output_cache_runtime import ResidualOutputCacheRuntime
from ecad.schedulers.cache_scheduler.dlm_cache_schedule import DLMCacheSchedule


def _cached_attention_forward(
    attention: torch.nn.Module,
    hidden_states: torch.Tensor,
    *args: Any,
    **kwargs: Any,
):
    runtime: ResidualOutputCacheRuntime = attention._ecad_cache_runtime
    layer_id = int(attention._ecad_layer_id)
    past_key_value = kwargs.get("past_key_value")
    use_cache = bool(kwargs.get("use_cache", False))
    output_attentions = bool(kwargs.get("output_attentions", False))
    if use_cache or past_key_value is not None:
        raise ValueError(
            "ECAD residual caching is incompatible with autoregressive KV "
            "caching; DiffuCoder must use use_cache=False"
        )
    if output_attentions:
        raise ValueError(
            "ECAD cannot return attention weights when an attention residual is reused"
        )

    cached = runtime.cached_or_none(layer_id, "attention", hidden_states)
    if cached is not None:
        return cached, None, None

    original_forward: Callable = attention._ecad_original_forward
    output = original_forward(hidden_states, *args, **kwargs)
    if not isinstance(output, tuple) or len(output) != 3:
        raise TypeError(
            "DiffuCoder attention must return "
            "(attention_output, attention_weights, present_key_value)"
        )
    runtime.store(layer_id, "attention", output[0])
    return output


def _cached_mlp_forward(
    mlp: torch.nn.Module,
    hidden_states: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    runtime: ResidualOutputCacheRuntime = mlp._ecad_cache_runtime
    layer_id = int(mlp._ecad_layer_id)
    cached = runtime.cached_or_none(layer_id, "mlp", hidden_states)
    if cached is not None:
        return cached

    original_forward: Callable = mlp._ecad_original_forward
    output = original_forward(hidden_states, *args, **kwargs)
    if not isinstance(output, torch.Tensor):
        raise TypeError("DiffuCoder MLP must return a tensor")
    runtime.store(layer_id, "mlp", output)
    return output


def _find_diffucoder_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    core_model = getattr(model, "model", None)
    layers = getattr(core_model, "layers", None)
    if layers is None:
        raise TypeError("Could not find model.model.layers on the DiffuCoder model")
    result = list(layers)
    if not result:
        raise TypeError("DiffuCoder model.model.layers is empty")
    for layer_id, layer in enumerate(result):
        if not hasattr(layer, "self_attn") or not hasattr(layer, "mlp"):
            raise TypeError(
                f"DiffuCoder layer {layer_id} does not expose self_attn and mlp"
            )
    return result


class DiffuCoderCacheController:
    """Install ECAD attention/MLP caching on a loaded DiffuCoder model."""

    def __init__(self, model: torch.nn.Module, schedule: DLMCacheSchedule) -> None:
        self.model = model
        self.blocks = _find_diffucoder_layers(model)
        if schedule.schedule_domain != "global_denoising_step":
            raise ValueError(
                "DiffuCoder requires a global_denoising_step cache schedule"
            )
        if len(self.blocks) != schedule.num_layers:
            raise ValueError(
                f"Model has {len(self.blocks)} layers but schedule has "
                f"{schedule.num_layers}"
            )
        self.runtime = ResidualOutputCacheRuntime(schedule)
        self._patched_modules: list[torch.nn.Module] = []
        self._installed = False
        self.install()

    def install(self) -> None:
        if self._installed:
            return
        try:
            for layer_id, layer in enumerate(self.blocks):
                attention = layer.self_attn
                mlp = layer.mlp
                for module, component, replacement in (
                    (attention, "attention", _cached_attention_forward),
                    (mlp, "mlp", _cached_mlp_forward),
                ):
                    if hasattr(module, "_ecad_original_forward"):
                        raise RuntimeError(
                            f"DiffuCoder layer {layer_id} {component} is already patched"
                        )
                    module._ecad_original_forward = module.forward
                    module._ecad_cache_runtime = self.runtime
                    module._ecad_layer_id = layer_id
                    module.forward = MethodType(replacement, module)
                    self._patched_modules.append(module)
        except Exception:
            self.uninstall()
            raise
        self._installed = True

    def uninstall(self) -> None:
        for module in reversed(self._patched_modules):
            module.forward = module._ecad_original_forward
            for attribute in (
                "_ecad_original_forward",
                "_ecad_cache_runtime",
                "_ecad_layer_id",
            ):
                if hasattr(module, attribute):
                    delattr(module, attribute)
        self._patched_modules.clear()
        self.runtime.start_generation()
        self._installed = False

    def set_schedule(self, schedule: DLMCacheSchedule) -> None:
        if schedule.schedule_domain != "global_denoising_step":
            raise ValueError(
                "DiffuCoder requires a global_denoising_step cache schedule"
            )
        self.runtime.set_schedule(schedule)

    def __enter__(self) -> "DiffuCoderCacheController":
        self.install()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.uninstall()


def attach_diffucoder_residual_cache(
    model: torch.nn.Module, schedule: DLMCacheSchedule
) -> DiffuCoderCacheController:
    return DiffuCoderCacheController(model, schedule)
