"""Integration with local DiffuCoder source, tiny random weights, no GPU/download.

Set DIFFUCODER_SOURCE_DIR to a checkpoint's code directory elsewhere. Tests skip
when source is unavailable. These are not substitutes for 7B bf16 GPU checks.
"""
import importlib
import os
from pathlib import Path
import sys
import types
import unittest
import torch
import yaml
from dlm_kv_policy.batched_decoding_study import build_backend
from dlm_kv_policy.diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
from dlm_kv_policy.dkv_decode_refresh_policy import DKVDecodeRefreshPolicy
from dlm_kv_policy.denoising_token_state import DenoisingTokenState


class TinyDreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_path = Path(__file__).parents[1] / 'configs/diffucoder7b_output_variance_greedy_64tasks.yaml'
        default = yaml.safe_load(config_path.read_text())['model']['path']
        root = Path(os.environ.get('DIFFUCODER_SOURCE_DIR', default))
        if not (root / 'modeling_dream.py').is_file():
            raise unittest.SkipTest('Local DiffuCoder source unavailable')
        package = types.ModuleType('ecad_test_dream')
        package.__path__ = [str(root)]
        sys.modules[package.__name__] = package
        previous = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            cls.module = importlib.import_module('ecad_test_dream.modeling_dream')
        finally:
            sys.dont_write_bytecode = previous

    def setUp(self):
        config = self.module.DreamConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, mask_token_id=63, pad_token_id=0)
        config._attn_implementation = 'sdpa'
        torch.manual_seed(3)
        self.model = self.module.DreamModel(config).eval()
        self.settings = dict(steps=16, gen_length=16, temperature=0, top_p=1, top_k=None,
            algorithm='entropy', algorithm_temperature=0, collect_history=True)

    def test_real_model_code_batch_one_identity(self):
        for depth in (None, 1, 3):
            for window in ((None,) if depth is None else (None, 1, 2, 3, 4, 8)):
                with self.subTest(depth=depth, window=window):
                    base = None if depth is None else DKVDecodeRefreshPolicy(cache_reloading_step=8, refresh_depth=depth)
                    options = dict(base=base, window=window, stop_ids={0}, m=2)
                    prompt = torch.tensor([[5, 6, 7]])
                    a = DiffuCoderDenoisingSampler(build_backend(self.model, **options, reference=True)).generate(
                        prompt, attention_mask=None, **self.settings)
                    b = DiffuCoderDenoisingSampler(build_backend(self.model, **options)).generate(
                        prompt, attention_mask=None, **self.settings)
                    self.assertTrue(all(torch.equal(x, y) for x,y in zip(a.history, b.history)))

    def test_native_dense_padding_is_finite_and_matches_unpadded_requests(self):
        x = torch.tensor([[0, 0, 5, 63, 63, 63], [5, 6, 7, 63, 63, 63]])
        mask = torch.tensor([[0, 0, 1], [1, 1, 1]])
        _, attn, pos = DiffuCoderDenoisingSampler._attention_inputs(x[:, :3], mask, 3)
        state = DenoisingTokenState.create(3, 6, mask, 63)
        inputs = dict(input_ids=x, attention_mask=attn, position_ids=pos, state=state, step=0, total_steps=3)
        a = build_backend(self.model).forward_mask_logits(**inputs).mask_logits
        b = build_backend(self.model, base=DKVDecodeRefreshPolicy(cache_reloading_step=1)).forward_mask_logits(**inputs).mask_logits
        self.assertTrue(torch.isfinite(a).all())
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        for row, offset in ((0, 2), (1, 0)):
            unpadded = x[row:row+1, offset:]
            s = DenoisingTokenState.create(3-offset, 6-offset, None, 63)
            c = build_backend(self.model, reference=True).forward_mask_logits(input_ids=unpadded,
                attention_mask='full', position_ids=torch.arange(6-offset)[None], state=s, step=0, total_steps=3)
            torch.testing.assert_close(a.reshape(2,3,-1)[row], c.mask_logits, atol=1e-6, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
