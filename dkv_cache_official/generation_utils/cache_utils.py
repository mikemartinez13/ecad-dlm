import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version

from transformers.utils.deprecation import deprecate_kwarg

from transformers.cache_utils import (
    Cache
)


class PrefillCache(Cache):
    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.num_hidden_layers = num_hidden_layers

        #print("Init Prefill Cache")

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")
        
    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or not self.key_cache[layer_idx].numel()  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length
    
    def is_empty(self) -> bool:
        return len(self.key_cache) < self.num_hidden_layers or len(self.value_cache) < self.num_hidden_layers
    
    def refresh_cache(self) -> None:
        self.key_cache = []
        self.value_cache = []
        #print("Cache size:", len(self.key_cache), len(self.value_cache))
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        B, n_h, _, h_d = key_states.shape
        prv_cache_position = cache_kwargs.get("prv_cache_position", None)
        cache_position = cache_kwargs.get("cache_position", None)

        if prv_cache_position is None:
            key_states_selected = key_states[cache_position[:, None, :, None].expand_as(key_states)].view(B, n_h, -1, h_d)
            self.key_cache.append(key_states_selected)

            value_states_selected = value_states[cache_position[:, None, :, None].expand_as(value_states)].view(B, n_h, -1, h_d)
            self.value_cache.append(value_states_selected)
        else:
            pass
            #raise NotImplementedError("Cache update for two steps has not beed implemented")


class DynamicCache(Cache):
    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.num_hidden_layers = num_hidden_layers

        self._transfer_order = None

        #print("Init Dynamic cache")

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")
        
    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or not self.key_cache[layer_idx].numel()  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length
    
    def is_empty(self) -> bool:
        return len(self.key_cache) < self.num_hidden_layers or len(self.value_cache) < self.num_hidden_layers
    
    def refresh_cache(self) -> None:
        self.key_cache = []
        self.value_cache = []
        #print("Refresh Cache. New Cache size:", len(self.key_cache), len(self.value_cache))

    def get_transfer_cache(self, layer_idx, cache_position, prv_cache_position):

        if layer_idx > 0:
            assert self._transfer_order is not None, "Transfer order is not set"
            order = self._transfer_order

            if layer_idx == self.num_hidden_layers - 1:
                self._transfer_order = None
            return order
        else:
            B = cache_position.shape[0]
            #print()
            current_order = torch.cat([
                (~prv_cache_position).nonzero(as_tuple=True)[1].view(B, -1),
                prv_cache_position.nonzero(as_tuple=True)[1].view(B, -1),
            ], dim=-1)

            #print(prv_cache_position.nonzero(as_tuple=True)[1].shape, cache_position.nonzero(as_tuple=True)[1].shape)
            #print((~prv_cache_position).nonzero(as_tuple=True)[1].shape, (~cache_position).nonzero(as_tuple=True)[1].shape)

            next_order = torch.cat([
                (~cache_position).nonzero(as_tuple=True)[1].view(B, -1),
                cache_position.nonzero(as_tuple=True)[1].view(B, -1),
            ], dim=-1)
            #print("next", next_order)
            #print("current", current_order)

            transfer_order = []
            for b in range(B):
                value_to_index = {v.item(): i for i, v in enumerate(current_order[b])}
                indices = torch.tensor([value_to_index.get(v.item(), -1) for v in next_order[b]])
                transfer_order.append(indices)
            transfer_order = torch.stack(transfer_order, dim=0).to(cache_position.device)

            self._transfer_order = transfer_order
            return transfer_order

    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        B, n_h, _, h_d = key_states.shape
        prv_cache_position = cache_kwargs.get("prv_cache_position", None)
        cache_position = cache_kwargs.get("cache_position", None)

        if prv_cache_position is None:
            assert self.is_empty(), "Cache is not empty, please refresh the cache"
            key_states_selected = key_states[cache_position[:, None, :, None].expand_as(key_states)].view(B, n_h, -1, h_d)
            self.key_cache.append(key_states_selected)

            value_states_selected = value_states[cache_position[:, None, :, None].expand_as(value_states)].view(B, n_h, -1, h_d)
            self.value_cache.append(value_states_selected)
        else:
            transfer_order = self.get_transfer_cache(layer_idx, cache_position, prv_cache_position)
            
            key_states = torch.gather(key_states, 2, transfer_order[:, None, :, None].expand_as(key_states))
            value_states = torch.gather(value_states, 2, transfer_order[:, None, :, None].expand_as(value_states))

            cache_start_pos = torch.sum(~cache_position, dim=-1)
            
            self.key_cache[layer_idx] = key_states[:, :, cache_start_pos[0]:, :]
            self.value_cache[layer_idx] = value_states[:, :, cache_start_pos[0]:, :]
            

            

class PrefillDynamicCache(Cache):
    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        self.prefill_key_cache: List[torch.Tensor] = []
        self.prefill_value_cache: List[torch.Tensor] = []

        self.num_hidden_layers = num_hidden_layers

        self._transfer_order = None

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
       
        if len(self.prefill_key_cache) < layer_idx:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

        if layer_idx < len(self.key_cache):
            decode_key_cache, decode_value_cache = self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            decode_key_cache, decode_value_cache = None, None

        prefill_key_cache, prefill_value_cache = self.prefill_key_cache[layer_idx], self.prefill_value_cache[layer_idx]
        key = torch.cat([prefill_key_cache, decode_key_cache], dim=-2) if decode_key_cache is not None else prefill_key_cache
        value = torch.cat([prefill_value_cache, decode_value_cache], dim=-2) if decode_value_cache is not None else prefill_value_cache

        return (key, value)
   
            
    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.__getitem__(layer_idx), self.__getitem__(layer_idx))

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.prefill_key_cache)
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        raise NotImplementedError("This function is not implemented for PrefillDynamicCache")
    
    def is_empty(self) -> bool:
        return len(self.prefill_key_cache) < self.num_hidden_layers or len(self.prefill_value_cache) < self.num_hidden_layers
    
    def is_decoded_empty(self) -> bool:
        return len(self.key_cache) < self.num_hidden_layers or len(self.value_cache) < self.num_hidden_layers
    
    def refresh_decode_cache(self) -> None:
        self.key_cache = []
        self.value_cache = []
        #print("Refresh Cache. New Cache size:", len(self.key_cache), len(self.value_cache))

    def set_prefill_cache(self, layer_idx, prefill_key, prefill_values) -> None:
        assert len(self.prefill_key_cache) <= layer_idx
        self.prefill_key_cache.append(prefill_key)
        self.prefill_value_cache.append(prefill_values)
        
    def get_transfer_cache(self, layer_idx, cache_position, prv_cache_position):

        if layer_idx > 0:
            assert self._transfer_order is not None, "Transfer order is not set"
            order = self._transfer_order

            if layer_idx == self.num_hidden_layers - 1:
                self._transfer_order = None
            return order
        else:
            B = cache_position.shape[0]
            #print()
            current_order = torch.cat([
                (prv_cache_position).nonzero(as_tuple=True)[1].view(B, -1),
                (~prv_cache_position).nonzero(as_tuple=True)[1].view(B, -1),
            ], dim=-1)

            #print(prv_cache_position.nonzero(as_tuple=True)[1].shape, cache_position.nonzero(as_tuple=True)[1].shape)
            #print((~prv_cache_position).nonzero(as_tuple=True)[1].shape, (~cache_position).nonzero(as_tuple=True)[1].shape)

            next_order = torch.cat([
                (cache_position).nonzero(as_tuple=True)[1].view(B, -1),
                (~cache_position).nonzero(as_tuple=True)[1].view(B, -1),
            ], dim=-1)
            #print("next", next_order)
            #print("current", current_order)
            #exit()

            transfer_order = []
            for b in range(B):
                value_to_index = {v.item(): i for i, v in enumerate(current_order[b])}
                indices = torch.tensor([value_to_index.get(v.item(), -1) for v in next_order[b]])
                transfer_order.append(indices)
            transfer_order = torch.stack(transfer_order, dim=0).to(cache_position.device)
            #print(transfer_order)
            #exit()
            self._transfer_order = transfer_order
            return transfer_order

    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        B, n_h, _, h_d = key_states.shape
        prv_cache_position = cache_kwargs.get("prv_cache_position", None)
        cache_position = cache_kwargs.get("cache_position", None)
        prefill_position = cache_kwargs.get("prefill_cache_position", None)

        if prefill_position is not None and len(self.prefill_key_cache) < self.num_hidden_layers:
            prefill_length = prefill_position.sum(dim=-1)[0]
            self.set_prefill_cache(layer_idx, key_states[:, :, :prefill_length], value_states[:, :, :prefill_length])
        elif len(self.key_cache) < self.num_hidden_layers:
            assert self.is_decoded_empty(), "Cache is not empty, please refresh the cache"
            cache_position = cache_position & (~prefill_position)
            if cache_position.sum() != 0:
                #print("Set first time key/value cache", cache_position.nonzero())
                key_states_selected = key_states[cache_position[:, None, :, None].expand_as(key_states)].view(B, n_h, -1, h_d)
                self.key_cache.append(key_states_selected)

                value_states_selected = value_states[cache_position[:, None, :, None].expand_as(value_states)].view(B, n_h, -1, h_d)
                self.value_cache.append(value_states_selected)
        else:
            #print("Take decoded cache")
            #print(cache_position.nonzero(as_tuple=True)[1], prv_cache_position.nonzero(as_tuple=True)[1], prefill_position.nonzero(as_tuple=True)[1])
            #cache_position = cache_position & (~prefill_position)
            #prv_cache_position = prv_cache_position & (~prefill_position)
            #print(cache_position.nonzero(), prv_cache_position.nonzero())

            transfer_order = self.get_transfer_cache(layer_idx, cache_position, prv_cache_position)
            
            key_states = torch.gather(key_states, 2, transfer_order[:, None, :, None].expand_as(key_states))
            value_states = torch.gather(value_states, 2, transfer_order[:, None, :, None].expand_as(value_states))

            cache_start_pos = torch.sum(cache_position, dim=-1)[0]
            prefill_end_pos = torch.sum(prefill_position, dim=-1)[0]
            
            self.key_cache[layer_idx] = key_states[:, :, prefill_end_pos:cache_start_pos, :]
            self.value_cache[layer_idx] = value_states[:, :, prefill_end_pos:cache_start_pos:, :]