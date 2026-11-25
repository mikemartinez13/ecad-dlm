import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import unittest
import json
import tempfile
from unittest.mock import patch
import torch
import torch.nn.functional as F
from dlm_kv_policy.teacher_forcing import expected_acceptance
from dlm_kv_policy.batched_teacher_forcing import _DenseStep, teacher_force_batch, commit_steps
from language_models.reasoning_tasks import score_response, gsm_extract
from language_models.dream_adapter import DreamAdapter
from evolution.nsga2_kv_policy_search import KVRefreshPolicyGenome


class ReasoningScoringTests(unittest.TestCase):
    def test_math_latex_gold_and_boxed_response(self):
        for gold, response in [(r'\sqrt{53}', r'The answer is $\boxed{\sqrt{53}}$.'),
                               (r'\frac{1}{2}', r'Therefore $\boxed{0.5}$.'),
                               ('42', r'The answer is $\boxed{42}$.'),
                               (r'\frac{2}{\sqrt{3}}', r'$\boxed{\frac{2\sqrt{3}}{3}}$')]:
            with self.subTest(gold=gold):
                result=score_response({'dataset':'math500','answer':gold},response)
                self.assertEqual(result['math_verify'],1)
                self.assertEqual(result['gold_parse_ok'],1)
                self.assertEqual(result['boxed_match'],1)

    def test_math_wrong_missing_and_earlier_answer(self):
        task={'dataset':'math','answer':r'\sqrt{53}'}
        for response in ['', r'$\boxed{\sqrt{52}}$',
                         r'We first considered $\sqrt{53}$, but the final answer is $\boxed{7}$.']:
            self.assertEqual(score_response(task,response)['math_verify'],0)

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

    def test_shared_teacher_resume_and_batch16_objective(self):
        from experiments.dream_reasoning import Experiment, padded_cost
        from utils.compute_cost import ModelShape
        from utils.results_io import save
        torch.set_num_threads(1)
        torch.manual_seed(19)
        c=self.module.DreamConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=3,
            num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,mask_token_id=63,pad_token_id=0)
        c._attn_implementation='sdpa'
        model=self.module.DreamModel(c).eval()
        model.lm_head.bias=torch.nn.Parameter(torch.zeros(64))
        model.lm_head.bias.data[63]=-1000
        adapter=DreamAdapter.__new__(DreamAdapter)
        adapter.model=model; adapter.stop_ids={0}; adapter.device='cpu'
        adapter.top_p_logits=importlib.import_module(model.__class__.__module__.replace('modeling_dream','generation_utils')).top_p_logits
        def prepare(prompts):
            tokens=[[int(x) for x in p.split()] for p in prompts]
            width=max(map(len,tokens))
            ids=torch.tensor([[0]*(width-len(t))+t for t in tokens])
            return ids,ids.ne(0)
        adapter.prepare=prepare
        groups=[]; dense=[]
        g=SimpleNamespace(steps=8,gen_length=8,temperature=.1,top_p=.9,top_k=None,algorithm='entropy',algorithm_temperature=0.)
        for index in range(2):
            # Different canonical widths expose accidental batch-16 FLOPs charging.
            tasks=[{'task_id':f'{index}/{b}','prompt':'4 5 6' if index==0 else '3 4 5 6 7','gen_length':8} for b in range(8)]
            ids,mask=prepare([t['prompt'] for t in tasks])
            torch.manual_seed(7+index)
            output=adapter.generate(ids,mask,g,history=True)
            rows=[{'task_id':t['task_id'],'dense_token_ids':output.sequences[b,ids.shape[1]:].tolist(),
                'commit_steps':commit_steps([h[b:b+1] for h in output.history],ids.shape[1],63),
                'answer_end':8,'prompt_tokens':int(mask[b].sum())} for b,t in enumerate(tasks)]
            groups.append((index,tasks,tasks)); dense.append({'tasks':rows})
        genomes={'refresh':KVRefreshPolicyGenome(1,1,1,0), 'dkv':KVRefreshPolicyGenome(1,4,1,0),
                 'reuse':KVRefreshPolicyGenome(1,4,2,0,2,2,'split+recent',4,1,4,.5,'local')}
        with tempfile.TemporaryDirectory() as tmp:
            def experiment(label,batch=8,grouped=False):
                e=Experiment.__new__(Experiment)
                e.output=Path(tmp)/label; e.batch=8; e.adapter=adapter; e.grouped=grouped
                e.shape=ModelShape.from_model(model); e.activity_state={}
                e.config={'teacher_forcing_batch_size':batch,'temperature':.1,'top_p':.9,
                          'search':{'objective_batch_size':8}}
                e.generate=lambda split:None
                e.batches=lambda split:iter(groups)
                for i,d in enumerate(dense): save(e.output/'gsm/dense'/f'batch_{i:04d}.json',d)
                return e
            serial=experiment('serial')
            reference={n:serial.proxy('gsm',n,g) for n,g in genomes.items()}
            shared=experiment('shared')
            with patch('experiments.dream_reasoning.teacher_force_batch', wraps=teacher_force_batch) as call:
                actual=shared.proxy_many('gsm',genomes)
                self.assertEqual(call.call_count,2)  # one teacher trajectory per batch, not per policy
            self.assertEqual(reference,actual)
            retained=shared.output/'gsm/refresh/proxy_0000.json'
            before=retained.read_bytes()
            (shared.output/'gsm/dkv/proxy_0001.json').unlink()
            with patch('experiments.dream_reasoning.teacher_force_batch', wraps=teacher_force_batch) as call:
                resumed=shared.proxy_many('gsm',genomes)
                self.assertEqual(call.call_count,1)
                self.assertEqual(list(call.call_args.args[2]),['dkv'])
            self.assertEqual(before,retained.read_bytes())
            self.assertEqual(reference,resumed)
            for grouped in (False,True):
                larger=experiment(f'b16_{grouped}',16,grouped)
                with patch('experiments.dream_reasoning.teacher_force_batch', wraps=teacher_force_batch) as call:
                    metrics=larger.proxy_many('gsm',genomes)
                    self.assertEqual(call.call_count,1)
                    self.assertEqual(call.call_args.args[3].shape[0],16)
                for name,genome in genomes.items():
                    rows=metrics[name]['tasks']
                    costs=[padded_cost(larger.shape,rows[i:i+8],genome.refresh_depth,8,8) for i in (0,8)]
                    expected=sum(x[0] for x in costs)/sum(x[1] for x in costs)
                    self.assertEqual(metrics[name]['padded_compute_fraction'],expected)
                    self.assertEqual([r['task_id'] for r in rows],[r['task_id'] for r in reference[name]['tasks']])
                    self.assertLess(abs(metrics[name]['ear']-reference[name]['ear']),1e-5)

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
