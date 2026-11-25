"""Dream-Instruct adapter for native and active-row batched inference.

No checkpoint patches: the existing Dream-shaped K/V backend is reused explicitly.
Native diffusion completes its fixed canvas; EOS/until trimming is postprocessing.
"""
from __future__ import annotations
import torch
from language_models.diffucoder_loader import load_diffucoder_model_and_tokenizer
from dlm_kv_policy.batched_diffucoder_backend import BatchedDiffuCoderKVBackend
from dlm_kv_policy.dense_diffucoder_baseline import BatchedDenseDiffuCoderBaseline
from dlm_kv_policy.diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
from dlm_kv_policy.teacher_forcing import load_top_p_logits


def finite_padding_mask(mask):
    """Avoid all-masked padding queries (NaN on some SDPA versions).

    Only invalid query rows change. Every valid query keeps exactly its native
    allowed keys, and padded positions can never be attended as keys.
    """
    if not isinstance(mask, torch.Tensor) or mask.ndim != 4 or mask.dtype != torch.bool:
        return mask
    valid_keys = mask.any(dim=-2, keepdim=True)
    return mask | (~mask.any(dim=-1, keepdim=True) & valid_keys)


class DreamAdapter:
    def __init__(self, path, device="cuda", dtype="bfloat16"):
        self.model, self.tokenizer = load_diffucoder_model_and_tokenizer(path, device=device, dtype=dtype)
        self.device = device
        self.top_p_logits = load_top_p_logits(self.model)
        self.stop_ids = {x for x in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id) if x is not None}

    def prepare(self, prompts):
        # Official Dream-Instruct harness: one user message; no injected system prompt.
        texts = [self.tokenizer.apply_chat_template([{"role": "user", "content": p}],
                 tokenize=False, add_generation_prompt=True) for p in prompts]
        encoded = self.tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt")
        # Dream's official instruct harness left-truncates to max_prompt_len=1024.
        return encoded.input_ids[:, -1024:].to(self.device), encoded.attention_mask[:, -1024:].to(self.device)

    def backend(self, genome=None, generation=None, grouped_attention=False):
        if genome is None:
            return BatchedDenseDiffuCoderBaseline(self.model)
        backend = BatchedDiffuCoderKVBackend(self.model, genome.policy(self.stop_ids),
                                            grouped_attention=grouped_attention)
        if generation is not None:
            backend.confidence_temperature = generation.temperature
            backend.confidence_top_p = generation.top_p
            backend.confidence_top_p_logits = self.top_p_logits
        return backend

    @staticmethod
    def settings(generation):
        return {**{k: getattr(generation, k) for k in ("steps", "gen_length", "temperature", "top_p",
                "top_k", "algorithm", "algorithm_temperature")}, "eps":getattr(generation,"eps",1e-3)}

    def generate(self, ids, mask, generation, genome=None, history=False, grouped_attention=False):
        backend = self.backend(genome, generation, grouped_attention)
        return DiffuCoderDenoisingSampler(backend).generate(ids, attention_mask=mask,
                  collect_history=history, **self.settings(generation))

    @torch.no_grad()
    def native(self, ids, mask, generation, history=False):
        def safe_padding(module, args, kwargs):
            if len(args)>1:
                args = (args[0],finite_padding_mask(args[1]),*args[2:])
            elif 'attention_mask' in kwargs:
                kwargs = {**kwargs,'attention_mask':finite_padding_mask(kwargs['attention_mask'])}
            return args,kwargs
        hook = self.model.register_forward_pre_hook(safe_padding, with_kwargs=True)
        try:
            return self.model.diffusion_generate(ids, attention_mask=mask,
                max_new_tokens=generation.gen_length, steps=generation.steps,
                temperature=generation.temperature, top_p=generation.top_p, top_k=generation.top_k,
                alg=generation.algorithm, alg_temp=generation.algorithm_temperature,
                eps=getattr(generation,"eps",1e-3),
                return_dict_in_generate=True, output_history=history)
        finally:
            hook.remove()

    def response(self, tokens, until):
        text = self.tokenizer.decode(tokens)
        for stop in [self.tokenizer.eos_token, *until]:
            if stop:
                text = text.split(stop)[0]
        return text
