"""One decoder layer as a function of tensors, so all layers share one compiled graph.

Optional path of BatchedDiffuCoderKVBackend (``compile_layers=True``). torch.compile
fuses the small per-layer operations, which is worth about 15% at batch 8 on an
A100. Fused arithmetic rounds differently: about a third of requests change some
tokens against the uncompiled backend, with no net pass@1 change on the audit set.
The first batches are slow while each new sequence length is compiled.

The function repeats the layer's arithmetic instead of calling its modules, because
a graph traced through modules would be compiled once per layer. ``check_layer``
confirms on the loaded model that it reproduces the modules before it is used.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

_compiled = None


def _rms_norm(hidden, weight, eps: float):
    values = hidden.float()
    values = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + eps)
    return weight * values.to(hidden.dtype)


def _rotate_half(values):
    first, second = values[..., : values.shape[-1] // 2], values[..., values.shape[-1] // 2:]
    return torch.cat((-second, first), dim=-1)


def layer_function(hidden, keys_cache, values_cache, positions, cos, sin, keymask, num_heads: int, num_kv_heads: int,
                   eps_in: float, eps_post: float, norm_in, norm_post, q_weight, q_bias, k_weight, k_bias,
                   v_weight, v_bias, o_weight, gate_weight, up_weight, down_weight):
    """Grouped-attention layer on packed rows; writes the rows' K/V into the caches in place."""

    batch, count, hidden_size = hidden.shape
    head_dim = keys_cache.shape[-1]
    normalized = _rms_norm(hidden, norm_in, eps_in)
    query = F.linear(normalized, q_weight, q_bias).view(batch, count, num_heads, head_dim).transpose(1, 2)
    keys = F.linear(normalized, k_weight, k_bias).view(batch, count, num_kv_heads, head_dim).transpose(1, 2)
    values = F.linear(normalized, v_weight, v_bias).view(batch, count, num_kv_heads, head_dim).transpose(1, 2)
    cos, sin = cos[:, None], sin[:, None]
    query = query * cos + _rotate_half(query) * sin
    keys = keys * cos + _rotate_half(keys) * sin
    index = positions[:, None, :, None].expand(-1, num_kv_heads, -1, head_dim)
    keys_cache.scatter_(2, index, keys)
    values_cache.scatter_(2, index, values)
    output = F.scaled_dot_product_attention(
        query.reshape(batch, num_kv_heads, (num_heads // num_kv_heads) * count, head_dim),
        keys_cache, values_cache, attn_mask=keymask).reshape(batch, num_heads, count, head_dim)
    hidden = hidden + F.linear(output.transpose(1, 2).reshape(batch, count, hidden_size), o_weight)
    normalized = _rms_norm(hidden, norm_post, eps_post)
    return hidden + F.linear(F.silu(F.linear(normalized, gate_weight)) * F.linear(normalized, up_weight), down_weight)


def layer_arguments(layer, num_heads: int, num_kv_heads: int) -> tuple:
    """The constant arguments of ``layer_function`` for one layer; raises if the layer has another form."""

    attention, mlp = layer.self_attn, layer.mlp
    try:
        if attention.o_proj.bias is not None or any(m.bias is not None for m in (mlp.gate_proj, mlp.up_proj, mlp.down_proj)):
            raise AttributeError("unexpected bias")
        if not isinstance(mlp.act_fn, torch.nn.SiLU):
            raise AttributeError("activation is not SiLU")
        return (num_heads, num_kv_heads, float(layer.input_layernorm.variance_epsilon),
                float(layer.post_attention_layernorm.variance_epsilon), layer.input_layernorm.weight,
                layer.post_attention_layernorm.weight, attention.q_proj.weight, attention.q_proj.bias,
                attention.k_proj.weight, attention.k_proj.bias, attention.v_proj.weight, attention.v_proj.bias,
                attention.o_proj.weight, mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight)
    except AttributeError as error:
        raise TypeError(f"compile_layers supports DiffuCoder's decoder layer only ({error})") from error


@torch.no_grad()
def check_layer(layer, arguments: tuple, reference) -> None:
    """Fail unless ``layer_function`` matches ``reference`` (the module path) on random rows.

    ``reference(hidden, keys_cache, values_cache, positions, cos, sin, keymask)`` must
    apply the same layer with grouped attention. Tolerance covers kernel choice only.
    """

    weight = arguments[6]
    head_dim = weight.shape[0] // arguments[0]
    generator = torch.Generator(device="cpu").manual_seed(0)
    def random(*shape):
        return torch.randn(*shape, generator=generator).to(device=weight.device, dtype=weight.dtype)
    hidden, cos, sin = random(2, 3, weight.shape[1]), random(2, 3, head_dim), random(2, 3, head_dim)
    positions = torch.tensor([[0, 2, 4], [1, 3, 5]], device=weight.device)
    keymask = torch.ones(2, 1, 1, 6, dtype=torch.bool, device=weight.device)
    outputs = []
    for function in (lambda *a: layer_function(*a, *arguments), reference):
        caches = [torch.full((2, arguments[1], 6, head_dim), 0.5, device=weight.device, dtype=weight.dtype) for _ in range(2)]
        outputs.append((function(hidden, *caches, positions, cos, sin, keymask).float(), *(c.float() for c in caches)))
    for ours, theirs in zip(*outputs):
        if not torch.allclose(ours, theirs, rtol=5e-2, atol=5e-2):
            raise TypeError("compile_layers: the layer function does not reproduce this model's layers")


def compiled_layer_function():
    """The shared compiled function; built on first use and reused by every backend."""

    global _compiled
    if _compiled is None:
        _compiled = torch.compile(layer_function, dynamic=True)
    return _compiled
