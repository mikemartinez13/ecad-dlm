"""Residual-branch caching hooks for the Hugging Face LLaDA implementation.

The block equations follow the MIT-licensed official LLaDA implementation at
https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct. The hooks are installed after
loading the checkpoint, so the model's remote
code and state-dict layout remain untouched. They reproduce the forward pass of
LLaDA's ``llama`` and ``sequential`` blocks while allowing the attention and
MLP residual outputs to be reused independently across denoising steps.
"""

from __future__ import annotations

from types import MethodType
from typing import Any, Callable

import torch

from ecad.language_models.residual_output_cache_runtime import ResidualOutputCacheRuntime
from ecad.schedulers.cache_scheduler.dlm_cache_schedule import DLMCacheSchedule


def _maybe_checkpoint(block: torch.nn.Module, function: Callable, *args: Any, **kwargs: Any):
    checkpoint_function = getattr(block, "_activation_checkpoint_fn", None)
    if checkpoint_function is None:
        return function(*args, **kwargs)
    return checkpoint_function(function, *args, **kwargs)


def _cached_llama_forward(
    block: torch.nn.Module,
    x: torch.Tensor,
    attention_bias: torch.Tensor | None = None,
    layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
    use_cache: bool = False,
) -> tuple[torch.Tensor, None]:
    runtime: ResidualOutputCacheRuntime = block._ecad_cache_runtime
    layer_id = int(block.layer_id)
    if use_cache or layer_past is not None:
        raise ValueError(
            "ECAD residual caching is incompatible with autoregressive KV caching; "
            "LLaDA diffusion generation must use use_cache=False"
        )

    attention_residual = runtime.cached_or_none(layer_id, "attention", x)
    if attention_residual is None:
        x_normed = _maybe_checkpoint(block, block.attn_norm, x)
        q = block.q_proj(x_normed)
        k = block.k_proj(x_normed)
        v = block.v_proj(x_normed)
        attention, _ = _maybe_checkpoint(
            block,
            block.attention,
            q,
            k,
            v,
            attention_bias,
            layer_past=None,
            use_cache=False,
        )
        attention_residual = block.dropout(attention)
        runtime.store(layer_id, "attention", attention_residual)
    x = x + attention_residual

    mlp_residual = runtime.cached_or_none(layer_id, "mlp", x)
    if mlp_residual is None:
        x_normed = _maybe_checkpoint(block, block.ff_norm, x)
        x_proj = block.ff_proj(x_normed)
        x_up = block.up_proj(x_normed)
        x_proj = _maybe_checkpoint(block, block.act, x_proj)
        mlp_residual = block.dropout(block.ff_out(x_proj * x_up))
        runtime.store(layer_id, "mlp", mlp_residual)
    x = x + mlp_residual
    return x, None


def _cached_sequential_forward(
    block: torch.nn.Module,
    x: torch.Tensor,
    attention_bias: torch.Tensor | None = None,
    layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
    use_cache: bool = False,
) -> tuple[torch.Tensor, None]:
    runtime: ResidualOutputCacheRuntime = block._ecad_cache_runtime
    layer_id = int(block.layer_id)
    if use_cache or layer_past is not None:
        raise ValueError(
            "ECAD residual caching is incompatible with autoregressive KV caching"
        )

    attention_residual = runtime.cached_or_none(layer_id, "attention", x)
    if attention_residual is None:
        x_normed = _maybe_checkpoint(block, block.attn_norm, x)
        q, k, v = block.att_proj(x_normed).split(block.fused_dims, dim=-1)
        attention, _ = _maybe_checkpoint(
            block,
            block.attention,
            q,
            k,
            v,
            attention_bias,
            layer_past=None,
            use_cache=False,
        )
        attention_residual = block.dropout(attention)
        runtime.store(layer_id, "attention", attention_residual)
    x = x + attention_residual

    mlp_residual = runtime.cached_or_none(layer_id, "mlp", x)
    if mlp_residual is None:
        x_normed = _maybe_checkpoint(block, block.ff_norm, x)
        x_proj = block.ff_proj(x_normed)
        x_proj = _maybe_checkpoint(block, block.act, x_proj)
        mlp_residual = block.dropout(block.ff_out(x_proj))
        runtime.store(layer_id, "mlp", mlp_residual)
    x = x + mlp_residual
    return x, None


def _find_llada_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    core_model = getattr(model, "model", model)
    transformer = getattr(core_model, "transformer", None)
    if transformer is None:
        raise TypeError("Could not find model.model.transformer on the LLaDA model")

    blocks = getattr(transformer, "blocks", None)
    if blocks is not None:
        return list(blocks)

    block_groups = getattr(transformer, "block_groups", None)
    if block_groups is not None:
        return [block for group in block_groups for block in group]
    raise TypeError("Could not find LLaDA transformer blocks")


class LLaDACacheController:
    """Install, configure, and remove ECAD hooks on a loaded LLaDA model."""

    def __init__(self, model: torch.nn.Module, schedule: DLMCacheSchedule) -> None:
        self.model = model
        self.blocks = _find_llada_blocks(model)
        if schedule.schedule_domain != "local_step_repeated_per_block":
            raise ValueError(
                "LLaDA requires a local_step_repeated_per_block cache schedule"
            )
        if len(self.blocks) != schedule.num_layers:
            raise ValueError(
                f"Model has {len(self.blocks)} layers but schedule has "
                f"{schedule.num_layers}"
            )
        self.runtime = ResidualOutputCacheRuntime(schedule)
        self._original_forwards: list[Callable] = []
        self._installed = False
        self.install()

    def install(self) -> None:
        if self._installed:
            return
        for expected_layer_id, block in enumerate(self.blocks):
            layer_id = int(getattr(block, "layer_id", expected_layer_id))
            if layer_id != expected_layer_id:
                raise ValueError(
                    f"Unexpected LLaDA layer order: position {expected_layer_id} "
                    f"contains layer {layer_id}"
                )
            self._original_forwards.append(block.forward)
            block._ecad_cache_runtime = self.runtime
            if all(hasattr(block, name) for name in ("q_proj", "k_proj", "v_proj", "up_proj")):
                block.forward = MethodType(_cached_llama_forward, block)
            elif all(hasattr(block, name) for name in ("att_proj", "fused_dims")):
                block.forward = MethodType(_cached_sequential_forward, block)
            else:
                self.uninstall()
                raise TypeError(
                    f"Unsupported LLaDA block implementation {type(block).__name__}"
                )
        self._installed = True

    def uninstall(self) -> None:
        for block, original_forward in zip(self.blocks, self._original_forwards):
            block.forward = original_forward
            if hasattr(block, "_ecad_cache_runtime"):
                delattr(block, "_ecad_cache_runtime")
        self._original_forwards.clear()
        self.runtime.start_generation()
        self._installed = False

    def set_schedule(self, schedule: DLMCacheSchedule) -> None:
        if schedule.schedule_domain != "local_step_repeated_per_block":
            raise ValueError(
                "LLaDA requires a local_step_repeated_per_block cache schedule"
            )
        self.runtime.set_schedule(schedule)

    def __enter__(self) -> "LLaDACacheController":
        self.install()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.uninstall()


def attach_llada_residual_cache(
    model: torch.nn.Module, schedule: DLMCacheSchedule
) -> LLaDACacheController:
    return LLaDACacheController(model, schedule)
