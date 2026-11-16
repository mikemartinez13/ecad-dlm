"""CPU invariants; real-model bf16 GPU equivalence is a separate validation rung."""
import copy
import unittest
import torch
from test_backend_cpu import FakeModel
from test_replay_compounding import greedy_sample_tokens
from dlm_kv_policy.batched_diffucoder_backend import BatchedDiffuCoderKVBackend
from dlm_kv_policy.diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend
from dlm_kv_policy.dkv_decode_refresh_policy import DKVDecodeRefreshPolicy
from dlm_kv_policy.masked_row_reuse_policy import MaskedRowReusePolicy, top_positions
from experiments.masked_row_reuse_replay import MaskedRowReuseBackend
from dlm_kv_policy.denoising_token_state import DenoisingTokenState
from dlm_kv_policy.diffucoder_denoising_sampler import DiffuCoderDenoisingSampler, commit_schedule
from dlm_kv_policy.batched_row_index import pack_rows, pack_rows_padded


def generate(backend, prompt, mask=None, diagnostics=False):
    sampler = object.__new__(DiffuCoderDenoisingSampler)
    sampler.backend, sampler.model, sampler.sample_tokens = backend, backend.model, greedy_sample_tokens
    return sampler.generate(prompt, attention_mask=mask, steps=16, gen_length=16,
        temperature=0, top_p=1, top_k=None, algorithm='entropy', algorithm_temperature=0,
        collect_history=True, collect_diagnostics=diagnostics)


class BatchedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = FakeModel().eval()

    def policy(self, depth=1, window=None):
        base = DKVDecodeRefreshPolicy(cache_reloading_step=8 if depth>1 else 256, refresh_depth=depth)
        return MaskedRowReusePolicy(base, window, 2, 'split+recent', {0}) if window else base

    def test_batch_one_generation_matches_reference_for_all_windows_and_depths(self):
        for depth in (1, 2):
            for window in (None, 1, 2, 3, 4, 8):
                with self.subTest(depth=depth, window=window):
                    policy = self.policy(depth, window)
                    reference = (MaskedRowReuseBackend if window else DiffuCoderActiveTokenKVBackend)(self.model, copy.deepcopy(policy))
                    batch = BatchedDiffuCoderKVBackend(self.model, policy)
                    a = generate(reference, torch.tensor([[4, 5, 6]]))
                    b = generate(batch, torch.tensor([[4, 5, 6]]))
                    for x, y in zip(a.history, b.history):
                        self.assertTrue(torch.equal(x, y))
                    self.assertEqual(set(batch.cache.boundary_hidden), {1} if depth == 2 else set())

    def test_uneven_left_padding_matches_independent_forced_trajectories(self):
        # Different commit locations and protected sets; compare every step to
        # independent unpadded executions while retaining each arm's own cache.
        for depth in (1, 2):
            policy = self.policy(depth, 3)
            backend = BatchedDiffuCoderKVBackend(self.model, copy.deepcopy(policy))
            refs = [MaskedRowReuseBackend(self.model, copy.deepcopy(policy)) for _ in range(2)]
            x = torch.tensor([[0, 0, 4, 31, 31, 31, 31, 31, 31], [5, 6, 7, 31, 31, 31, 31, 31, 31]])
            valid = torch.tensor([[0, 0, 1], [1, 1, 1]])
            state = DenoisingTokenState.create(3, 9, valid, 31)
            states = [DenoisingTokenState.create(n, n+6, None,31) for n in (1,3)]
            refx = [x[0:1,2:].clone(),x[1:2].clone()]
            _, attention, positions = DiffuCoderDenoisingSampler._attention_inputs(x[:,:3],valid,6)
            for step in range(6):
                out = backend.forward_mask_logits(input_ids=x, attention_mask=attention, position_ids=positions,
                    state=state,step=step,total_steps=6)
                rows = out.mask_logits.reshape(2,6-step,-1)
                for b in range(2):
                    ref = refs[b].forward_mask_logits(input_ids=refx[b],attention_mask='full',
                        position_ids=torch.arange(refx[b].shape[1])[None],state=states[b],step=step,total_steps=6)
                    torch.testing.assert_close(rows[b],ref.mask_logits,rtol=1e-5,atol=1e-6)
                    torch.testing.assert_close(out.candidates.reshape(2,-1)[b],ref.candidates)
                previous=x.clone()
                for b in range(2):
                    old=refx[b].clone()
                    remaining=torch.where(x[b].eq(31))[0]
                    pos=remaining[0 if b==0 else -1]
                    x[b,pos]=8+step
                    refx[b][0,pos-(2 if b==0 else 0)]=8+step
                    states[b].update_after_sampling(old,refx[b],step)
                state.update_after_sampling(previous,x,step)
            for keys in backend.cache.keys:
                self.assertTrue(keys[0,:,:2].eq(0).all())

    def test_dummy_writes_are_discarded(self):
        backend = BatchedDiffuCoderKVBackend(self.model,self.policy())
        backend.start_generation(5,batch_size=2)
        selected=torch.tensor([[0,1,0,0,0],[1,0,1,0,1]],dtype=torch.bool)
        self.assertEqual(pack_rows(selected).tolist(),[[1,-1,-1],[0,2,4]])
        positions=pack_rows_padded(selected,3)
        self.assertEqual(positions.tolist(),[[1,5,5],[0,2,4]])
        keys=torch.ones(2,1,3,4)
        backend.cache.update_kv(0,positions,keys,keys,0)
        # Dummy rows land in the spare slot (index 5); real positions are untouched.
        self.assertEqual(backend.cache.keys[0][0,:,:5].sum().item(),4)
        self.assertTrue(backend.cache.keys[0][0,:,0].eq(0).all())

    def test_trims_and_packed_sampling_do_not_change_generation(self):
        # Uneven left padding, both depths, with and without masked-row reuse.
        prompt=torch.tensor([[0,4,5],[6,7,8]])
        mask=torch.tensor([[0,1,1],[1,1,1]])
        for depth in (1,2):
            for window in (None,2,4):
                plain=BatchedDiffuCoderKVBackend(self.model,self.policy(depth,window),grouped_attention=False,lean_head=False)
                # Hiding forward_step sends the sampler down the forward_mask_logits path.
                plain.forward_step=None
                expected=generate(plain,prompt,mask)
                for options in ({},{'grouped_attention':False},{'lean_head':False},{'grouped_attention':False,'lean_head':False}):
                    with self.subTest(depth=depth,window=window,**options):
                        actual=generate(BatchedDiffuCoderKVBackend(self.model,self.policy(depth,window),**options),prompt,mask)
                        for x,y in zip(expected.history,actual.history):
                            self.assertTrue(torch.equal(x,y))

    def test_compiled_layers_refuse_other_layer_forms(self):
        # The compiled function repeats DiffuCoder's layer arithmetic; the fake layers differ.
        with self.assertRaises(TypeError):
            BatchedDiffuCoderKVBackend(self.model,self.policy(),compile_layers=True)
        with self.assertRaises(ValueError):
            BatchedDiffuCoderKVBackend(self.model,self.policy(),compile_layers=True,grouped_attention=False)

    def test_lean_head_projects_fewer_rows(self):
        prompt=torch.tensor([[4,5,6]])
        rows={}
        for lean in (False,True):
            backend=BatchedDiffuCoderKVBackend(self.model,self.policy(1,4),lean_head=lean)
            rows[lean]=sum(r['head_rows'] for r in generate(backend,prompt,diagnostics=True).diagnostics['rows'])
        self.assertLess(rows[True],rows[False])

    def test_schedule_keeps_zero_and_final_double_commit(self):
        schedule=commit_schedule(torch.linspace(1,1e-12,257),256)
        self.assertEqual(schedule[1],0)
        self.assertEqual(schedule[-1],2)
        self.assertEqual(sum(schedule),256)

    def test_stable_ranking_per_request(self):
        scores=torch.zeros(2,5,dtype=torch.bfloat16)
        allowed=torch.tensor([[0,1,1,1,1],[1,0,1,1,1]],dtype=torch.bool)
        self.assertEqual(top_positions(scores,allowed,2).tolist(),
            [[False,True,True,False,False],[True,False,True,False,False]])

    def test_cache_reset_isolates_requests_and_reuses_storage(self):
        backend = BatchedDiffuCoderKVBackend(self.model, self.policy(2, 2))
        backend.start_generation(5, batch_size=2)
        pointer = backend.cache.keys[0].data_ptr()
        backend.cache.keys[0].fill_(9)
        backend.start_generation(5, batch_size=2)
        self.assertEqual(pointer, backend.cache.keys[0].data_ptr())
        self.assertTrue(backend.cache.keys[0].eq(0).all())
        self.assertFalse(backend.cache.ready)

    def test_invalid_windows_are_rejected(self):
        for window in (0, -1, 2.5):
            with self.assertRaises(ValueError):
                BatchedDiffuCoderKVBackend(self.model, MaskedRowReusePolicy(
                    DKVDecodeRefreshPolicy(), window, 2, 'split+recent', {0}))

    def test_diagnostics_do_not_change_generation(self):
        prompt=torch.tensor([[0,4,5],[6,7,8]])
        mask=torch.tensor([[0,1,1],[1,1,1]])
        a=generate(BatchedDiffuCoderKVBackend(self.model,self.policy(2,4)),prompt,mask)
        b=generate(BatchedDiffuCoderKVBackend(self.model,self.policy(2,4)),prompt,mask,True)
        self.assertTrue(torch.equal(a.sequences,b.sequences))
        self.assertEqual(len(b.diagnostics['rows']),16)

if __name__ == '__main__':
    unittest.main()
