from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import os

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers import PretrainedConfig

from ecad.schedulers.cache_scheduler.dream_cache_schedule import (
    Dream7bCacheSchedule,
)
from ecad.schedulers.dit_scheduler.dit_scheduler import DiTScheduler
from ecad.lm_models.dream.configuration_dream import ODreamConfig
from ecad.lm_models.dream.modeling_dream import (
    DreamDecoderLayer,
    DreamForCausalLM,
)


class CachedDreamDecoderLayer(DreamDecoderLayer):
    def __init__(
        self,
        layer_num: str,
        cache_schedule: Dream7bCacheSchedule,
        config: ODreamConfig,
        layer_idx: int,
    ) -> None:
        super().__init__(config=config, layer_idx=layer_idx)
        self.layer_num = layer_num
        self.cache_schedule = cache_schedule

        self.cached_layer_output: torch.Tensor | None = None
        self.cached_attn_output: torch.Tensor | None = None
        self.cached_mlp_output: torch.Tensor | None = None

    def reset_cache(self) -> None:
        self.cached_layer_output = None
        self.cached_attn_output = None
        self.cached_mlp_output = None

    def _compute_attn_cached(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_value: Optional[Cache],
        output_attentions: bool,
        use_cache: bool,
        cache_position: Optional[torch.LongTensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        # recompute = self.cache_schedule.get_recompute(self.layer_num, "kv")
        recompute = True # temporary forcing recompute of attention to avoid cache bugs
        
        no_cache = self.cached_attn_output is None

        if not recompute and no_cache:
            print(
                f"WARNING: No cached attention found at layer {self.layer_num}. Recomputing."
            )

        if recompute or no_cache:
            attn_output, attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        else:
            attn_output = self.cached_attn_output  # type: ignore[assignment]
            attn_weights = None
            present_key_value = past_key_value

        self.cached_attn_output = attn_output
        return attn_output, attn_weights, present_key_value

    def _compute_mlp_cached(
        self,
        norm_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # recompute = self.cache_schedule.get_recompute(self.layer_num, "mlp")
        recompute = True # temporary forcing recompute of MLP to avoid cache bugs
        
        no_cache = self.cached_mlp_output is None

        if not recompute and no_cache:
            print(
                f"WARNING: No cached MLP output found at layer {self.layer_num}. Recomputing."
            )

        if recompute or no_cache:
            mlp_output = self.mlp(norm_hidden_states)
        else:
            mlp_output = self.cached_mlp_output  # type: ignore[assignment]

        self.cached_mlp_output = mlp_output
        return mlp_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        # recompute_layer = self.cache_schedule.get_recompute(
        #     self.layer_num, "layer"
        # )
        recompute_layer = True # temporary forcing recompute of entire layer to avoid cache bugs    
        
        no_cache = self.cached_layer_output is None

        if not recompute_layer and no_cache:
            print(
                f"WARNING: No cached layer output found at layer {self.layer_num}. Recomputing."
            )

        if recompute_layer or no_cache:
            residual = hidden_states
            norm_hidden_states = self.input_layernorm(hidden_states)

            attn_output, self_attn_weights, present_key_value = (
                self._compute_attn_cached(
                    hidden_states=norm_hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=bool(output_attentions),
                    use_cache=bool(use_cache),
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            )
            hidden_states = residual + attn_output

            residual = hidden_states
            norm_hidden_states = self.post_attention_layernorm(hidden_states)
            mlp_output = self._compute_mlp_cached(norm_hidden_states)
            hidden_states = residual + mlp_output
        else:
            hidden_states = self.cached_layer_output  # type: ignore[assignment]
            self_attn_weights = None
            present_key_value = past_key_value

        self.cached_layer_output = hidden_states

        outputs: tuple[torch.Tensor, ...] = (hidden_states,)
        if output_attentions:
            if self_attn_weights is None:
                self_attn_weights = torch.empty(
                    0, device=hidden_states.device, dtype=hidden_states.dtype
                )
            outputs += (self_attn_weights,)
        if use_cache:
            if present_key_value is None:
                present_key_value = past_key_value
            outputs += (present_key_value,)  # type: ignore[arg-type]
        return outputs


class DreamForCausalLMEdited(DreamForCausalLM):
    def __init__(
        self,
        config: ODreamConfig,
        dit_scheduler: DiTScheduler | None = None,
        cache_schedule: Dream7bCacheSchedule | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config)
        self.dit_scheduler: DiTScheduler = dit_scheduler  # type: ignore[assignment]
        self.cache_schedule: Dream7bCacheSchedule | None = cache_schedule

    def _post_init(
        self,
        dit_scheduler: DiTScheduler | None,
        cache_schedule: Dream7bCacheSchedule | None,
    ) -> None:
        if dit_scheduler is None:
            raise ValueError("A DiTScheduler object must be provided.")

        self.cache_schedule = cache_schedule
        if self.cache_schedule is not None:
            print("Using cached Dream decoder layers.")
            self.init_cached_decoder_layers()
        else:
            print("Using basic Dream decoder layers.")

        dit_scheduler.update_transformer_blocks(self.model.layers)
        self.dit_scheduler = dit_scheduler

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args: Any,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        dit_scheduler: DiTScheduler | None = None,
        cache_schedule: Dream7bCacheSchedule | None = None,
        **kwargs: Any,
    ) -> "DreamForCausalLMEdited":
        model: DreamForCausalLMEdited = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )  # type: ignore[assignment]

        model._post_init(dit_scheduler, cache_schedule)
        return model

    def init_cached_decoder_layers(self) -> None:
        if self.cache_schedule is None:
            raise ValueError("cache_schedule must be set before replacing layers.")

        new_layers = []
        for layer_idx, base_layer in enumerate(self.model.layers):
            param = next(base_layer.parameters())
            cached_layer = CachedDreamDecoderLayer(
                layer_num=str(layer_idx),
                cache_schedule=self.cache_schedule,
                config=self.config,
                layer_idx=layer_idx,
            ).to(device=param.device, dtype=param.dtype)
            cached_layer.load_state_dict(base_layer.state_dict(), strict=True)
            new_layers.append(cached_layer)

        self.model.layers = nn.ModuleList(new_layers)

    def reset_cache(self) -> None:
        for layer in self.model.layers:
            if hasattr(layer, "reset_cache"):
                layer.reset_cache()
