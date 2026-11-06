"""Study rules, interpolation and batch-maximum accounting without a GPU."""
import unittest
from dataclasses import replace
import numpy as np
import torch
from test_backend_cpu import FakeModel
from experiments.proxy_recheck import PANEL, assessment, category, policies
from experiments.batched_cost_model import (BLOCKS, predict_step, interpolate, calibration_metrics,
    schedule_inputs, SchedulePolicy)
from dlm_kv_policy.batched_diffucoder_backend import BatchedDiffuCoderKVBackend


class PrerequisiteTests(unittest.TestCase):
    def test_panel_is_twelve_unique_policies_with_eight_local_alternatives(self):
        self.assertEqual(len(policies({0})),12)
        self.assertEqual(sum(n.startswith('depth_') and n!='depth_22' for n,*_ in PANEL),8)

    def test_spearman_point_nine_and_auc_are_both_required(self):
        passes=np.ones((3,20),dtype=bool)
        passes[0,:2]=False;passes[1,:5]=False;passes[2,:9]=False
        proxy=np.repeat([[1.],[2.],[3.]],20,axis=1)
        task_proxy=(~passes).astype(float)
        result=assessment(proxy,task_proxy,passes,np.ones(20,dtype=bool),np.arange(20))
        self.assertEqual(result['verdict'],'PASS')
        reverse=assessment(proxy[::-1],task_proxy,passes,np.ones(20,dtype=bool),np.arange(20))
        self.assertEqual(reverse['verdict'],'FAIL')
        self.assertEqual(reverse['regressions'],16)

    def test_ties_and_small_samples_are_inconclusive(self):
        x=np.zeros((3,8));passes=np.ones((3,8),dtype=bool)
        r=assessment(x,x,passes,np.ones(8,dtype=bool),np.arange(8))
        self.assertEqual(r['verdict'],'INCONCLUSIVE')
        self.assertIsNone(r['spearman']);self.assertIsNone(r['auroc'])

    def test_batch_maxima_are_taken_separately_per_block(self):
        curve={f'{a}:{b}':{'0':0.,'100':float(b-a)*100} for a,b in BLOCKS}
        curve['head']={'0':0.,'256':2560.}
        table={'curves':{'8:512:1':curve}}
        # Request 1 has most full rows, request 2 has most deep rows.
        total=predict_step(table,8,512,22,[40,20],[60,90],128)
        self.assertEqual(total,21*40+7*90+1280)
        with self.assertRaises(ValueError):predict_step(table,8,999,22,[40],[90],128)
        with self.assertRaises(ValueError):interpolate({'0':0.,'10':1.},11)

    def test_cost_gate_is_permutation_sensitive(self):
        self.assertTrue(calibration_metrics([1,2,3,4],[1,2,3,4])['passed'])
        self.assertFalse(calibration_metrics([4,3,2,1],[1,2,3,4])['passed'])

    def test_synthetic_backend_decisions_and_cache_are_valid(self):
        model=FakeModel().eval()
        initial,frames,decisions,counts=schedule_inputs(3,384,(1,12,12,8,4),31,torch.device('cpu'))
        backend=BatchedDiffuCoderKVBackend(model,SchedulePolicy(1,decisions))
        backend.forward_mask_logits(**initial)
        for frame,(full,active,masked) in zip(frames,counts):
            result=backend.forward_mask_logits(**frame)
            self.assertEqual(result.mask_logits.shape,(3*masked,32))
            self.assertEqual(result.decision.active.sum(-1).tolist(),active)
            self.assertTrue((result.decision.active & ~frame['state'].valid_mask).sum()==0)

if __name__=='__main__':unittest.main()
