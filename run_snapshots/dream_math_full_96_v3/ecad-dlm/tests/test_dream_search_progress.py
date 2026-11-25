"""Checkpoint continuation and reporting tests with a deterministic proxy stub."""
import json
from pathlib import Path
import tempfile
import unittest
from experiments.dream_reasoning import Experiment
from experiments.dream_search_progress import summarize
from utils.results_io import save


class SearchContinuationTests(unittest.TestCase):
    def make_experiment(self, root):
        exp=Experiment.__new__(Experiment)
        exp.output=root
        exp.config={'search':{'population_size':8,'generations':2,
            'quality_objective':'constrained_ear_flops','cost_objective':'padded_flops'},
            'search_split':'test','audit_top':2}
        exp.activity_state={}
        exp.calls=[]
        def proxy(split, name, genome):
            exp.calls.append(name)
            return {'ear':.99,'ear_95_ci':[.98,1.], 'tail_ear':.97,
                    'padded_compute_fraction':.3+1/(genome.refresh_interval+2),
                    'answer_coverage':1.,'tasks':[]}
        exp.proxy=proxy
        save(root/'run_config.json',exp.config)
        save(root/'ear_calibration.json',{'min_ear':.95,'min_tail_ear':.94})
        return exp

    def test_chain_resume_and_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            exp=self.make_experiment(root)
            exp.search(0)
            self.assertEqual(len(exp.calls),8)
            self.assertFalse((root/'finalists.json').exists())
            self.assertEqual(summarize(root)['completed_generations'],1)
            exp.search(0)
            self.assertEqual(len(exp.calls),8)  # completed generation is not reevaluated
            exp.search(1)
            self.assertEqual(len(exp.calls),16)
            self.assertEqual(len(set(exp.calls)),16)
            report=summarize(root)
            self.assertEqual(report['evaluated_candidates'],16)
            self.assertEqual(report['completed_generations'],2)
            self.assertEqual(len(json.loads((root/'finalists.json').read_text())),2)
            self.assertTrue((root/'generation_history.csv').exists())
            self.assertGreater(report['history'][0]['best_ear'],.98)

    def test_missing_predecessor_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=self.make_experiment(Path(tmp))
            with self.assertRaisesRegex(ValueError,'Previous generation'):
                exp.search(1)
            self.assertFalse(exp.calls)

    def test_partial_and_no_feasible_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            exp=self.make_experiment(root)
            save(root/'ear_calibration.json',{'min_ear':1.,'min_tail_ear':1.})
            exp.search(0)
            report=summarize(root)
            self.assertIsNone(report['history'][0]['best_flops_fraction'])
            save(root/'search/generation_001/partial.json',[{'policy_name':'unfinished'}])
            self.assertEqual(summarize(root)['evaluated_candidates'],9)


if __name__=='__main__':
    unittest.main()
