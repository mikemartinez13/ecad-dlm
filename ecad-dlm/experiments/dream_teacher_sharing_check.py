"""Bounded, single-GPU GSM8K comparison on identical recorded teacher trajectories."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch
import yaml

from experiments.dream_reasoning import Experiment, Genome, PANEL, seed
from language_models.dream_adapter import DreamAdapter
from language_models.reasoning_tasks import load_tasks
from utils.results_io import save


VARIANTS = [
    ('serial_b8', 8, False, 1),
    ('shared_b8', 8, False, 16),
    ('shared_grouped_b8', 8, True, 16),
    ('shared_b16', 16, False, 16),
    ('shared_grouped_b16', 16, True, 16),
]


def compare(reference, candidate, threshold):
    rows = {}
    for name, base in reference.items():
        new = candidate[name]
        assert [r['task_id'] for r in base['tasks']] == [r['task_id'] for r in new['tasks']]
        def feasible(r):
            return r['ear_95_ci'][0] >= threshold['min_ear'] and r['tail_ear'] >= threshold['min_tail_ear']
        rows[name] = {
            'ear_delta':new['ear']-base['ear'],
            'ear_ci_lower_delta':new['ear_95_ci'][0]-base['ear_95_ci'][0],
            'tail_ear_delta':new['tail_ear']-base['tail_ear'],
            'max_task_ear_delta':max(abs(a['ear']-b['ear']) for a,b in zip(base['tasks'],new['tasks'])),
            'padded_flops_fraction_delta':new['padded_compute_fraction']-base['padded_compute_fraction'],
            'answer_coverage_delta':new['answer_coverage']-base['answer_coverage'],
            'tasks_with_changed_rows':sum(a['rows'] != b['rows'] for a,b in zip(base['tasks'],new['tasks'])),
            'reference_feasible':feasible(base), 'candidate_feasible':feasible(new),
        }
    return {'policies':rows, 'feasibility_changes':sum(
        r['reference_feasible'] != r['candidate_feasible'] for r in rows.values()),
        'note':'Pilot thresholds on 16 prompts are descriptive; this is not a new calibration or accuracy claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('/scratch/gqs7zr/dream_reasoning/results/dream_reasoning_v1'))
    parser.add_argument('--prepare-only', action='store_true', help='Validate and prepare the frozen panel without loading a model')
    args = parser.parse_args()
    if (args.output/'report.json').exists():
        raise ValueError('Use a fresh output directory: checkpoint resumes are not valid timing measurements')
    config = yaml.safe_load(args.config.read_text())
    original = json.loads((args.source/'provenance.json').read_text())['config']
    source_tasks = {t['task_id']:t for t in load_tasks(Path(original['data_root'])/'gsm8k_search.jsonl')}
    # One short and one long original batch, preserving their stochastic trajectories.
    sources = sorted((args.source/'gsm8k_search'/'dense').glob('batch_*.json'))
    chosen = [sources[0], sources[-1]]
    dense = [json.loads(p.read_text()) for p in chosen]
    tasks = [source_tasks[t['task_id']] for d in dense for t in d['tasks']]
    if len(tasks) != 16 or len({t['task_id'] for t in tasks}) != 16:
        raise ValueError('Benchmark requires two distinct original batches of eight')
    data = args.output/'data'
    data.mkdir(parents=True, exist_ok=True)
    task_path = data/'gsm8k_sharing_check.jsonl'
    task_path.write_text(''.join(json.dumps(t)+'\n' for t in tasks))
    save(data/'manifest.json', {'source':str(args.source), 'tasks_sha256':hashlib.sha256(task_path.read_bytes()).hexdigest(),
        'teacher_sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in chosen}})
    raw = [row for path in sorted((args.source/'search').glob('generation_*/genomes.json'))
           for row in json.loads(path.read_text())]
    genomes = {PANEL['all_refresh'].name:PANEL['all_refresh']}
    for row in raw:
        genome = Genome.from_dict(row)
        genomes[genome.name] = genome
        if len(genomes) == 16:
            break
    if len(genomes) != 16:
        raise ValueError('Expected 16 distinct policies')
    save(args.output/'genomes.json', [g.to_dict() for g in genomes.values()])
    threshold = json.loads((args.source/'ear_calibration.json').read_text())
    if args.prepare_only:
        print(json.dumps({'questions':len(tasks),'policies':len(genomes),'teacher_sources':[str(p) for p in chosen]}))
        return
    report = {'status':'running','questions':16,'policies':16,'steps':256,'objective_batch_size':8,
        'variants':{},'source':str(args.source),
        'note':'One warmed pass per variant, no repeat-based confidence interval. No MATH, search, or successor jobs.'}
    save(args.output/'report.json', report)
    adapter = DreamAdapter(config['model_path'])
    for label, batch, grouped, group_size in VARIANTS:
        root = args.output/label
        variant = {**config, 'data_root':str(data), 'batch_size':8,
            'teacher_forcing_batch_size':batch, 'teacher_policy_group_size':group_size,
            'grouped_attention':grouped, 'stages':[], 'search_split':'gsm8k_sharing_check'}
        exp = Experiment(variant, root, adapter=adapter)
        for i, d in enumerate(dense):
            save(root/'gsm8k_sharing_check/dense'/f'batch_{i:04d}.json', d)
        # Warm kernels for this shape/attention path; no data is scored or generated here.
        ids, mask = adapter.prepare([t['prompt'] for t in tasks[:batch]])
        warm = exp.generation(16)
        warm.steps = 4
        seed(config['seed'])
        adapter.generate(ids, mask, warm, PANEL['combined'], grouped_attention=grouped)
        del ids, mask
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        report['active_variant'] = label
        save(args.output/'report.json', report)
        started = perf_counter()
        metrics = None
        try:
            if group_size == 1:
                metrics = {name:exp.proxy('gsm8k_sharing_check', name, genome) for name,genome in genomes.items()}
            else:
                metrics = exp.proxy_many('gsm8k_sharing_check', genomes)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            report['variants'][label] = {'status':'out_of_memory', 'seconds':perf_counter()-started,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
            if group_size == 1:
                save(args.output/'report.json', report)
                raise
        else:
            seconds = perf_counter()-started
            save(root/'metrics.json', metrics)
            result = {'status':'complete','seconds':seconds,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                'teacher_batch_size':batch,'grouped_attention':grouped,'policy_group_size':group_size}
            if label != 'serial_b8':
                result['speedup'] = report['variants']['serial_b8']['seconds']/seconds
                baseline = json.loads((args.output/'serial_b8/metrics.json').read_text())
                result['comparison'] = compare(baseline, metrics, threshold)
                del baseline
            report['variants'][label] = result
            print(json.dumps({'variant':label, **{k:v for k,v in result.items() if k!='comparison'}}), flush=True)
        save(args.output/'report.json', report)
        del metrics, exp
        gc.collect(); torch.cuda.empty_cache()
    report['status'] = 'complete'
    report.pop('active_variant', None)
    save(args.output/'report.json', report)


if __name__ == '__main__':
    main()
