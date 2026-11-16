"""The replay must keep each policy's own stale cache, not rebuild it from dense.

If the cache were rebuilt from dense values every step, the replay would only
measure one-step error and would underrate long refresh intervals. Here the
"dense" reference is the same backend recomputing everything, so a policy's
logits must equal it exactly on its full-refresh steps and drift in between.
"""

from types import SimpleNamespace
import unittest

import torch

from dlm_kv_policy.diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend
from dlm_kv_policy.diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
from dlm_kv_policy.dkv_decode_refresh_policy import DKVDecodeRefreshPolicy
from dlm_kv_policy.forced_trajectory_replay import replay_task, step_divergence
from test_backend_cpu import FakeModel

MASK = 31


def greedy_sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
    """Stand-in for DiffuCoder's sample_tokens: greedy, never proposes the mask token."""

    logits = logits.clone()
    logits[:, MASK] = -torch.inf
    probs = torch.softmax(logits, dim=-1)
    confidence, tokens = probs.max(dim=-1)
    if neg_entropy:
        confidence = torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return confidence, tokens


class ReplayCompoundingTests(unittest.TestCase):
    def test_policy_cache_goes_stale_between_its_own_refreshes(self) -> None:
        torch.manual_seed(1)
        model = FakeModel().eval()
        for parameter in model.parameters():
            parameter.data.mul_(3)
        reference = DiffuCoderActiveTokenKVBackend(model, DKVDecodeRefreshPolicy(cache_reloading_step=1))
        sampler = object.__new__(DiffuCoderDenoisingSampler)
        sampler.backend, sampler.model, sampler.sample_tokens = reference, model, greedy_sample_tokens
        policy = DiffuCoderActiveTokenKVBackend(model, DKVDecodeRefreshPolicy(cache_reloading_step=4))
        generation = SimpleNamespace(
            steps=12, gen_length=12, temperature=0.0, top_p=1.0, top_k=None,
            algorithm="entropy", algorithm_temperature=0.0,
        )
        prompt = torch.tensor([[4, 5, 6]])

        rows = replay_task(model, sampler, {"refresh_4": policy}, prompt, torch.ones_like(prompt), generation, None)[
            "policies"
        ]["refresh_4"]

        full_steps = [row for row in rows if row["step"] <= 1 or row["step"] % 4 == 0]
        cached_steps = [row for row in rows if row not in full_steps]
        self.assertTrue(full_steps and cached_steps)
        self.assertTrue(all(row["kl"] == 0.0 and step_divergence(row) == 0.0 for row in full_steps))
        self.assertGreater(max(row["kl"] for row in cached_steps), 0.0)


if __name__ == "__main__":
    unittest.main()
