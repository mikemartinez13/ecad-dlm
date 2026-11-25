import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import unittest
import torch
import torch.nn.functional as F
from dlm_kv_policy.teacher_forcing import expected_acceptance
from dlm_kv_policy.batched_teacher_forcing import _DenseStep, teacher_force_batch, commit_steps
from language_models.reasoning_tasks import score_response, gsm_extract
from language_models.dream_adapter import DreamAdapter
from evolution.nsga2_kv_policy_search import KVRefreshPolicyGenome


class ReasoningScoringTests(unittest.TestCase):
    def test_full_ear_has_unit_self_overlap_for_flat_distributions(self):
        p = torch.full((2,100),.01).log()
        torch.testing.assert_close(expected_acceptance(p,p,None),torch.ones(2))
        torch.testing.assert_close(expected_acceptance(p,p),torch.full((2,),.1))

    def test_full_ear_equals_one_minus_tv(self):
        p,q=F.log_softmax(torch.randn(3,100),-1),F.log_softmax(torch.randn(3,100),-1)
        torch.testing.assert_close(expected_acceptance(p,q,None),1-.5*(p.exp()-q.exp()).abs().sum(-1))

    def test_gsm_upstream_strict_and_flexible_are_separate(self):
        task={'dataset':'gsm8k','answer':'1200'}
        self.assertEqual(score_response(task,'The answer is 1,200.')['exact_match'],1)
        result=score_response(task,'We get 1200')
        self.assertEqual(result['exact_match'],0)
        self.assertEqual(result['flexible_extract'],1)

    def test_vectorized_full_ear(self):
        g=SimpleNamespace(temperature=0.,top_p=1.,ear_top_k=None)
        logits=torch.randn(4,32)
        step=_DenseStep(logits,[[0,1],[0,1]],[[0],[1]],g,None)
        torch.testing.assert_close(step.score(logits,None)['ear'],torch.ones(4))


class NativeDreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(os.environ.get('DREAM_SOURCE_DIR','/scratch/gqs7zr/cache/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae'))
        if not (root/'modeling_dream.py').exists():
            raise unittest.SkipTest('Pinned Dream source unavailable')
        pkg=ModuleType('ecad_native_dream_test');pkg.__path__=[str(root)]
        sys.modules[pkg.__name__]=pkg
        cls.module=importlib.import_module(pkg.__name__+'.modeling_dream')

    def test_native_and_all_refresh_batch8_stochastic(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        c=self.module.DreamConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=3,
            num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,mask_token_id=63,pad_token_id=0)
        c._attn_implementation='sdpa'
        model=self.module.DreamModel(c).eval()
        model.lm_head.bias=torch.nn.Parameter(torch.zeros(64))
        model.lm_head.bias.data[63]=-1000
        adapter=DreamAdapter.__new__(DreamAdapter)
        adapter.model=model;adapter.stop_ids={0};adapter.device='cpu'
        adapter.top_p_logits=importlib.import_module(model.__class__.__module__.replace('modeling_dream','generation_utils')).top_p_logits
        ids=torch.tensor([[0,5,6],[4,5,6]]*4); mask=ids.ne(0)
        g=SimpleNamespace(steps=8,gen_length=8,temperature=.1,top_p=.9,top_k=None,algorithm='entropy',algorithm_temperature=0.)
        torch.manual_seed(7); native=adapter.native(ids,mask,g,history=True)
        torch.manual_seed(7); dense=adapter.generate(ids,mask,g,history=True)
        torch.manual_seed(7); refresh=adapter.generate(ids,mask,g,KVRefreshPolicyGenome(1,1,1,0),history=True)
        self.assertTrue(torch.equal(native.sequences,dense.sequences))
        self.assertTrue(torch.equal(native.sequences,refresh.sequences))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(native.history,refresh.history)))
        g.ear_top_k=None
        trajectories=[{'dense_token_ids':dense.sequences[b,3:].tolist(),
            'commit_steps':commit_steps([h[b:b+1] for h in dense.history],3,63)} for b in range(8)]
        for scoring in ('vectorized','per_task'):
            with torch.no_grad():
                results=teacher_force_batch(model,adapter.backend(),
                    {'refresh':adapter.backend(KVRefreshPolicyGenome(1,1,1,0),g)},ids,mask,
                    trajectories,g,adapter.top_p_logits,answer_ends=[8]*8,scoring=scoring)
            for result in results:
                for step in result['policies']['refresh']:
                    self.assertTrue(all(abs(v-1)<1e-5 for v in step['ear']))


if __name__=='__main__':
    unittest.main()
