import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments import masked_row_smoothness_study as study

CADD_CODE = Path(__file__).resolve().parents[2] / "ml-cadd"


def load_tiny_cadd_model():
    """A random 3-layer Dream model built from the unmodified ml-cadd code."""

    import importlib.util
    import types

    package = types.ModuleType("cadd_pkg")
    package.__path__ = [str(CADD_CODE)]
    sys.modules["cadd_pkg"] = package
    modules = {}
    for name in ("configuration_dream", "generation_utils", "modeling_dream"):
        spec = importlib.util.spec_from_file_location(f"cadd_pkg.{name}", CADD_CODE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"cadd_pkg.{name}"] = module
        spec.loader.exec_module(module)
        modules[name] = module
    config = modules["configuration_dream"].DreamConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        mask_token_id=63, pad_token_id=0, bos_token_id=0, eos_token_id=0,
    )
    config._attn_implementation = "sdpa"
    return modules["modeling_dream"].DreamModel(config).eval()


def ban_mask_token(step, tokens, logits):
    """A random model may predict the mask id, which would leave rows masked."""

    logits = logits.clone()
    logits[..., 63] = -torch.inf
    return logits


class MaskedRowSmoothnessTest(unittest.TestCase):
    def run_recorder(self, **generate):
        torch.manual_seed(0)
        model = load_tiny_cadd_model()
        prompt = torch.tensor([[1, 2, 3, 4, 5]])
        steps = 12
        recorder = study.StepRecorder(model, 63, prompt.shape[1], steps)
        try:
            with torch.no_grad():
                model.diffusion_generate(
                    prompt, attention_mask=None, max_new_tokens=steps, steps=steps, temperature=0.0,
                    top_p=1.0, alg="entropy", alg_temp=0.0, return_dict_in_generate=True,
                    generation_tokens_hook_func=recorder.token_hook,
                    generation_logits_hook_func=ban_mask_token, **generate,
                )
        finally:
            recorder.close()
        return recorder

    def test_masked_counts_fall_one_per_step_and_stats_are_filled(self):
        for generate in ({"use_cadd": True, "cadd_sampling_mode": "weighted"}, {"use_cadd": False}):
            recorder = self.run_recorder(**generate)
            np.testing.assert_array_equal(recorder.n_masked, np.arange(12, 0, -1))
            lag1 = recorder.rel[:, 0]
            self.assertTrue(np.isnan(lag1[0]).all())
            self.assertTrue(np.isfinite(lag1[1:, 0]).all())  # masked group, every kind and layer
            self.assertTrue((lag1[1:] >= 0).all())
            # lag 4 only exists from step 4 on
            self.assertTrue(np.isnan(recorder.rel[3, 1]).all() and np.isfinite(recorder.rel[4, 1, 0]).all())
            # counts: masked rows at a step equal n_masked; prompt rows are constant
            np.testing.assert_array_equal(recorder.counts[1:, 0, 0], recorder.n_masked[1:])
            self.assertTrue((recorder.counts[1:, 0, 2] == 5).all())

    def test_change_matches_direct_computation(self):
        torch.manual_seed(0)
        model = load_tiny_cadd_model()
        prompt = torch.tensor([[1, 2, 3, 4, 5]])
        seen = []

        def capture(module, inputs, output):
            seen.append(output[0].detach().clone())

        handle = model.model.layers[1].self_attn.v_proj.register_forward_hook(capture)
        recorder = self.run_recorder(use_cadd=False)
        handle.remove()
        # Recompute the same run with an independent hook on layer 1 V.
        torch.manual_seed(0)
        model = load_tiny_cadd_model()
        handle = model.model.layers[1].self_attn.v_proj.register_forward_hook(capture)
        seen.clear()
        steps = 12
        masks = []

        def token_hook(step, tokens, logits):
            masks.append(tokens[0] == 63)
            return tokens

        with torch.no_grad():
            model.diffusion_generate(
                prompt, attention_mask=None, max_new_tokens=steps, steps=steps, temperature=0.0, top_p=1.0,
                alg="entropy", alg_temp=0.0, return_dict_in_generate=True,
                generation_tokens_hook_func=token_hook,
                generation_logits_hook_func=ban_mask_token, use_cadd=False,
            )
        handle.remove()
        step = 5
        now, before = seen[step].float(), seen[step - 1].float()
        change = (now - before).norm(dim=-1) / now.norm(dim=-1)
        self.assertEqual(change.shape, masks[step].shape)
        expected = change[masks[step]].mean().item()
        self.assertAlmostEqual(float(recorder.rel[step, 0, 0, 1, 1]), expected, places=5)


if __name__ == "__main__":
    unittest.main()
