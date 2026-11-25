"""Batched Dream reasoning pilot: parity -> EAR calibration -> search -> OOD audit.

Search uses full-vocabulary EAR at temperature 1 and padded FLOPs. Answer accuracy
is descriptive and never a search objective or an audit pass/fail criterion.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
import numpy as np
import torch
import yaml

from language_models.dream_adapter import DreamAdapter, finite_padding_mask
from language_models.reasoning_tasks import load_tasks, score_response, ROOT, SCORING_VERSION
from dlm_kv_policy.batched_teacher_forcing import teacher_force_batch, commit_steps
from dlm_kv_policy.denoising_token_state import DenoisingTokenState
from dlm_kv_policy.diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
from dlm_kv_policy.kv_policy_experiment_config import KVPolicyNSGA2SearchConfig
from evolution.nsga2_kv_policy_search import (KVRefreshPolicyGenome as Genome, EvaluatedKVPolicy,
    environmental_selection, offspring_population, _random_genome)
from experiments.ear_validity import task_metrics
from utils.compute_cost import ModelShape, layer_flops, lm_head_flops
from utils.results_io import save, source_provenance
from utils.ranking import rank_score


PANEL = {
    'all_refresh': Genome(1, 1, 1, 0),
    'dkv8': Genome(1, 8, 1, 0),
    'depth22': Genome(1, 8, 22, 0),
    'combined': Genome(2, 8, 22, 8, 4, 32, 'split+recent', 32, 2, 16, .5, 'local'),
    'aggressive': Genome(1, 32, 1, 0, 8, 16, 'split', 128, 0, 32, 0., 'full'),
}


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def bootstrap(values, seed_value=20261116):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Nonfinite or empty metric samples')
    rng = np.random.default_rng(seed_value)
    means = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def padded_cost(shape, records, depth, steps, length):
    """Actual batch rectangle; reference uses the same masked-only vocabulary head.

    records includes duplicate filler requests in a short final batch, so their
    execution cost is counted even though their accuracy is excluded.
    """
    width = max(r['prompt_tokens'] for r in records) + length
    one = layer_flops(shape, 1, width)
    policy = dense = 0
    for step in range(steps):
        rows = [r['rows'][step] for r in records]
        active, full, masked = (max(r[k] for r in rows) for k in ('active','full_depth','masked'))
        policy += len(records) * (((depth-1)*full + (shape.num_layers-depth+1)*active)*one
                                  + lm_head_flops(shape, masked))
        dense += len(records) * (shape.num_layers*width*one + lm_head_flops(shape, masked))
    return policy, dense


def summary_pairs(dense, cached):
    if [r['task_id'] for r in dense] != [r['task_id'] for r in cached]:
        raise ValueError('Paired scoring requires identical task order')
    out = {}
    for metric in ('exact_match', 'flexible_extract', 'math_verify', 'boxed_match', 'upstream_math_verify'):
        if not all(metric in row['score'] for row in dense + cached):
            continue
        a = np.array([r['score'][metric] for r in dense])
        b = np.array([r['score'][metric] for r in cached])
        out[metric] = {'dense':float(a.mean()), 'cached':float(b.mean()), 'delta':float((b-a).mean()),
            'paired_95_ci':bootstrap(b-a), 'lost':int(((a==1)&(b==0)).sum()),
            'gained':int(((a==0)&(b==1)).sum())}
    return out


class Experiment:
    def __init__(self, config, output):
        self.config, self.output = config, output
        self.batch = int(config['batch_size'])
        if self.batch < 8:
            raise ValueError('Dream reasoning experiments require batch size >= 8')
        self.adapter = DreamAdapter(config['model_path'])
        self.shape = ModelShape.from_model(self.adapter.model)
        self.data = Path(config['data_root'])
        self.grouped = config.get('grouped_attention', False)
        import transformers
        source = source_provenance()
        # The frozen source hashes identify this run; unrelated working-tree
        # changes outside the snapshot must not prevent a checkpoint resume.
        source.pop('git_status', None)
        meta = {**source, 'config':config, 'torch':torch.__version__, 'scoring_version':SCORING_VERSION,
                'transformers':transformers.__version__, 'gpu':torch.cuda.get_device_name(),
                'scorer_sha256':hashlib.sha256((ROOT/'Dream/eval_instruct/lm_eval/tasks/minerva_math/utils.py').read_bytes()).hexdigest(),
                'data':json.loads((self.data/'manifest.json').read_text())}
        signature = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
        # Source identity is frozen by launcher; hardware/library identity also cannot
        # silently change on resume.
        old = output/'provenance.json'
        if old.exists() and json.loads(old.read_text())['signature'] != signature:
            raise ValueError('Run provenance changed; choose a new output directory')
        save(old, {**meta, 'signature':signature})

    def generation(self, length):
        return SimpleNamespace(steps=length, gen_length=length, temperature=self.config['temperature'],
            top_p=self.config['top_p'], top_k=None, algorithm='entropy', algorithm_temperature=0.,
            ear_top_k=None, eps=1e-3)

    def batches(self, split):
        tasks = load_tasks(self.data/f'{split}.jsonl')
        # Stable length bucketing is identical for dense, proxy, and free generation.
        tasks.sort(key=lambda t:(t['gen_length'], len(self.adapter.tokenizer.encode(t['prompt'])), t['task_id']))
        for start in range(0,len(tasks),self.batch):
            real = tasks[start:start+self.batch]
            group = real + [real[-1]]*(self.batch-len(real))
            if len({t['gen_length'] for t in group}) != 1:
                raise ValueError('Manifest mixes generation lengths within a batch')
            yield start//self.batch, real, group

    def generate(self, split, name='dense', genome=None):
        results = []
        for index, real, group in self.batches(split):
            path = self.output/split/name/f'batch_{index:04d}.json'
            if path.exists():
                results.extend(json.loads(path.read_text())['tasks'])
                continue
            ids, mask = self.adapter.prepare([t['prompt'] for t in group])
            generation = self.generation(group[0]['gen_length'])
            seed(self.config['seed']+index)
            print(f'generate {split} {name} batch {index}', flush=True)
            torch.cuda.reset_peak_memory_stats()
            history = genome is None and self.config.get('record_dense_history', True)
            result = self.adapter.generate(ids, mask, generation, genome, history=history,
                                           grouped_attention=self.grouped)
            seconds = result.elapsed_seconds
            if history:
                # Trajectory recording transfers history to CPU every step. Do not
                # count that overhead in the dense speed baseline.
                seed(self.config['seed']+index)
                timed = self.adapter.generate(ids,mask,generation,history=False,grouped_attention=self.grouped)
                if not torch.equal(timed.sequences,result.sequences):
                    raise RuntimeError('Dense timing and trajectory runs differ despite the same seed')
                seconds = timed.elapsed_seconds
                del timed
            tokens = result.sequences[:,ids.shape[1]:].tolist()
            rows = []
            for b, task in enumerate(real):
                response = self.adapter.response(tokens[b], task['until'])
                row = {'task_id':task['task_id'], 'response':response, 'score':score_response(task,response),
                    'dense_token_ids':tokens[b], 'prompt_tokens':int(mask[b].sum()),
                    'answer_end':next((p+1 for p,t in enumerate(tokens[b]) if t in self.adapter.stop_ids),len(tokens[b]))}
                if history:
                    row['commit_steps'] = commit_steps([h[b:b+1] for h in result.history],ids.shape[1],
                                                       self.adapter.model.config.mask_token_id)
                rows.append(row)
            save(path, {'tasks':rows, 'seconds':seconds, 'batch_size':self.batch,
                'real_requests':len(real), 'filler_requests':len(group)-len(real),
                'peak_memory_bytes':torch.cuda.max_memory_allocated(), 'generation':vars(generation)})
            results.extend(rows)
            del result
        return results

    @torch.no_grad()
    def parity(self):
        _, _, group = next(self.batches('gsm8k_calibration'))
        ids, mask = self.adapter.prepare([t['prompt'] for t in group])
        generation = self.generation(32)
        seed(self.config['seed'])
        native = self.adapter.native(ids, mask, generation, history=True)
        seed(self.config['seed'])
        dense = self.adapter.generate(ids, mask, generation, history=True)
        seed(self.config['seed'])
        refresh = self.adapter.generate(ids, mask, generation,PANEL['all_refresh'],history=True,
                                        grouped_attention=self.grouped)
        report = {'batch_size':self.batch, 'generation':vars(generation),
            'native_vs_dense_token_agreement':float(native.sequences.eq(dense.sequences).float().mean()),
            'native_vs_dense_generated_agreement':float(native.sequences[:,ids.shape[1]:].eq(
                dense.sequences[:,ids.shape[1]:]).float().mean()),
            'dense_vs_all_refresh_generated_agreement':float(dense.sequences[:,ids.shape[1]:].eq(
                refresh.sequences[:,ids.shape[1]:]).float().mean()),
            'native_dense_identical_steps':sum(torch.equal(a.cpu(),b) for a,b in zip(native.history,dense.history)),
            'steps':generation.steps}
        # Numerical parity on identical native states isolates arithmetic differences
        # from stochastic trajectories. Check beginning, middle, and final denoising.
        errors = []
        for step in (0, 15, 31):
            x = torch.nn.functional.pad(ids,(0,generation.gen_length),value=self.adapter.model.config.mask_token_id)
            if step:
                x = native.history[step-1].clone()
            _, attention, positions = DiffuCoderDenoisingSampler._attention_inputs(ids,mask,generation.gen_length)
            state = DenoisingTokenState.create(ids.shape[1], x.shape[1],mask,
                self.adapter.model.config.mask_token_id,batch_size=self.batch,device=ids.device)
            inputs = dict(input_ids=x,attention_mask=attention,position_ids=positions,state=state,
                          step=step,total_steps=generation.steps)
            reference = self.adapter.model(x,finite_padding_mask(attention),positions).logits
            reference = torch.cat([reference[:,:1],reference[:,:-1]],1)[state.classify(x,step).eq(2)]
            for label, backend in [('dense',self.adapter.backend()),('all_refresh',self.adapter.backend(
                    PANEL['all_refresh'],generation,self.grouped))]:
                backend.start_generation(x.shape[1],batch_size=self.batch)
                actual = backend.forward_mask_logits(**inputs).mask_logits
                error = (actual.float()-reference.float()).abs()
                errors.append({'step':step,'backend':label,'max_abs':float(error.max()),'mean_abs':float(error.mean())})
                torch.testing.assert_close(actual,reference,atol=.125,rtol=.01)
                del backend, actual, error
        report['same_state_logits'] = errors
        report['native_padding_fix'] = 'Only all-masked padded queries are allowed to attend valid keys to avoid SDPA NaNs'
        report['note'] = 'bf16 arithmetic can change stochastic trajectories; same-state logits are checked separately'
        save(self.output/'parity.json',report)
        print(json.dumps(report),flush=True)

    def proxy(self, split, name, genome):
        self.generate(split)
        records = []
        costs = [0.,0.]
        for index, real, group in self.batches(split):
            path = self.output/split/name/f'proxy_{index:04d}.json'
            if path.exists():
                saved = json.loads(path.read_text())
            else:
                dense = json.loads((self.output/split/'dense'/f'batch_{index:04d}.json').read_text())['tasks']
                trajectories = dense + [dense[-1]]*(self.batch-len(dense))
                ids, mask = self.adapter.prepare([t['prompt'] for t in group])
                generation = self.generation(group[0]['gen_length'])
                backend = self.adapter.backend(genome,generation,self.grouped)
                print(f'proxy {split} {name} batch {index}',flush=True)
                with torch.no_grad():
                    results = teacher_force_batch(self.adapter.model,self.adapter.backend(),{name:backend},
                        ids,mask,trajectories,generation,self.adapter.top_p_logits,
                        answer_ends=[t['answer_end'] for t in trajectories])
                all_records = []
                for dense_row,result in zip(trajectories,results):
                    metrics = task_metrics(result['policies'][name],dense_row['answer_end'])
                    outside = sum(p < dense_row['answer_end'] for row in result['policies'][name]
                                  for p in row.get('outside_positions',[]))
                    all_records.append({'task_id':dense_row['task_id'],**metrics,
                        'answer_coverage':1-outside/dense_row['answer_end'],
                        'prompt_tokens':dense_row['prompt_tokens'],'rows':result['rows'][name]})
                cost = padded_cost(self.shape,all_records,genome.refresh_depth,generation.steps,generation.gen_length)
                saved = {'tasks':all_records[:len(real)],'padded_flops':cost[0],'dense_padded_flops':cost[1],
                         'ear_definition':'full_vocabulary_temperature_1','genome':genome.to_dict()}
                save(path,saved)
                del backend, results
                gc.collect()
            records.extend(saved['tasks'])
            costs[0] += saved['padded_flops']
            costs[1] += saved['dense_padded_flops']
        ear = [r['ear'] for r in records]
        return {'ear':float(np.mean(ear)), 'ear_95_ci':bootstrap(ear),
            'tail_ear':float(np.quantile(ear,.1)), 'padded_compute_fraction':costs[0]/costs[1],
            'answer_coverage':float(np.mean([r['answer_coverage'] for r in records])), 'tasks':records}

    def calibration(self):
        split = self.config.get('calibration_split', 'gsm8k_calibration')
        dense = self.generate(split)
        panel = {}
        controls = {**PANEL, **{Genome.from_dict(g).name:Genome.from_dict(g)
                    for g in self.config.get('calibration_genomes', [])}}
        for name,genome in controls.items():
            proxy = self.proxy(split,name,genome)
            cached = self.generate(split,name,genome)
            panel[name] = {**proxy,'genome':genome.to_dict(),'quality':summary_pairs(dense,cached)}
            save(self.output/'calibration_panel.json',panel)
        # Exploratory EAR anchor, not a claimed accuracy guarantee. Calibration
        # does not inspect the search set or OOD responses, and sets no accuracy gate.
        anchor = panel['dkv8']
        threshold = {'min_ear':anchor['ear_95_ci'][0], 'min_tail_ear':anchor['tail_ear'],
            'anchor':'dkv8', 'definition':'match conservative control fidelity on calibration prompts',
            'note':'Exploratory thresholds; no statistical-losslessness claim or downstream pass/fail rule'}
        names = list(panel)
        x = [panel[n]['ear'] for n in names]
        metric = self.config.get('quality_metric', 'flexible_extract')
        y = [panel[n]['quality'][metric]['delta'] for n in names]
        threshold['quality_metric'] = metric
        threshold['calibration_split'] = split
        threshold['ear_accuracy_spearman'] = float(rank_score(x,y)) if len(set(x))>1 and len(set(y))>1 else None
        save(self.output/'ear_calibration.json',threshold)

    def search(self):
        threshold = json.loads((self.output/'ear_calibration.json').read_text())
        settings = KVPolicyNSGA2SearchConfig(**self.config['search'])
        settings.validate()
        population, seen = [], set()
        for generation in range(settings.generations):
            directory = self.output/'search'/f'generation_{generation:03d}'
            if (directory/'summary.json').exists():
                old = json.loads((directory/'summary.json').read_text())
                population = [EvaluatedKVPolicy.from_dict(r) for r in old['population']]
                seen = {Genome.from_dict(r) for r in old['seen']}
                continue
            if (directory/'genomes.json').exists():
                genomes = [Genome.from_dict(g) for g in json.loads((directory/'genomes.json').read_text())]
            elif generation:
                genomes = offspring_population(population,settings,generation,forbidden=seen)
            else:
                genomes = [PANEL[k] for k in ('dkv8','depth22','combined','aggressive')]
                for item in self.config.get('initial_genomes', []):
                    genome = Genome.from_dict(item)
                    if genome not in genomes:
                        genomes.append(genome)
                rng = random.Random(settings.seed)
                while len(genomes)<settings.population_size:
                    item = _random_genome(settings,rng)
                    if item not in genomes:
                        genomes.append(item)
            save(directory/'genomes.json',[g.to_dict() for g in genomes])
            evaluated=[]
            for genome in genomes:
                metrics = self.proxy(self.config.get('search_split', 'gsm8k_search'),genome.name,genome)
                violation = max(0.,threshold['min_ear']-metrics['ear_95_ci'][0]) + max(0.,threshold['min_tail_ear']-metrics['tail_ear'])
                cost = metrics['padded_compute_fraction']
                evaluated.append(EvaluatedKVPolicy(genome,1-metrics['ear'],cost,cost=cost,violation=violation,
                    constraints={k:v for k,v in metrics.items() if k!='tasks'}))
                save(directory/'partial.json',[e.to_dict() for e in evaluated])
            population = environmental_selection(population+evaluated,settings.population_size)
            seen.update(genomes)
            save(directory/'summary.json',{'evaluated':[x.to_dict() for x in evaluated],
                'population':[x.to_dict() for x in population], 'seen':[g.to_dict() for g in sorted(seen,key=lambda g:g.name)]})
        ranked = sorted(population,key=lambda p:(p.violation,p.cost))
        save(self.output/'finalists.json',[p.to_dict() for p in ranked[:self.config.get('audit_top',2)]])

    def audit(self):
        policies = self.audit_policies()
        report={}
        for split in self.config.get('audit_splits', ['gsm8k_dev_holdout','math500_ood_pilot']):
            dense=self.generate(split)
            report[split]={}
            for name,genome in policies.items():
                cached=self.generate(split,name,genome)
                dense_time=sum(json.loads(p.read_text())['seconds'] for p in (self.output/split/'dense').glob('batch_*.json'))
                cached_time=sum(json.loads(p.read_text())['seconds'] for p in (self.output/split/name).glob('batch_*.json'))
                report[split][name]={'quality':summary_pairs(dense,cached),'measured_speedup':dense_time/cached_time,
                    'dense_seconds':dense_time,'cached_seconds':cached_time,'tasks':len(dense),
                    'note':'Single pilot timing pass, excluding loading/tokenization/scoring; not a stable throughput claim'}
                save(self.output/'audit.json',report)

    def audit_policies(self):
        path = Path(self.config.get('finalists_path', self.output/'finalists.json'))
        finalists = json.loads(path.read_text())
        policies = {'dkv8':PANEL['dkv8']} if self.config.get('audit_control', True) else {}
        policies.update({r['policy_name']:Genome.from_dict(r['genome']) for r in finalists})
        return policies

    def timing(self):
        """Warm each path, rotate execution order, and time identical requests/seeds.

        Each repeat covers a fixed subset. No trajectory copies or scoring occurs
        inside timings. These repeats measure latency variability, not seed variance.
        """
        split = self.config.get('timing_split', 'gsm8k_timing')
        policies = {'dense':None, **self.audit_policies()}
        groups = list(self.batches(split))
        _, _, group = groups[0]
        ids, mask = self.adapter.prepare([t['prompt'] for t in group])
        for name, genome in policies.items():
            seed(self.config['seed'])
            self.adapter.generate(ids,mask,self.generation(16),genome,
                                  history=False,grouped_attention=self.grouped)
        for repeat in range(self.config.get('timing_repeats', 3)):
            names = list(policies)
            names = names[repeat % len(names):] + names[:repeat % len(names)]
            for index, real, group in groups:
                ids, mask = self.adapter.prepare([t['prompt'] for t in group])
                generation = self.generation(group[0]['gen_length'])
                for name in names:
                    path = self.output/'timing'/f'repeat_{repeat}'/name/f'batch_{index:04d}.json'
                    if path.exists():
                        continue
                    seed(self.config['seed']+index)
                    print(f'timing repeat {repeat} {name} batch {index}', flush=True)
                    result = self.adapter.generate(ids,mask,generation,policies[name],
                                                   history=False,grouped_attention=self.grouped)
                    digest = hashlib.sha256(result.sequences.cpu().numpy().tobytes()).hexdigest()
                    save(path, {'seconds':result.elapsed_seconds, 'output_sha256':digest,
                         'task_ids':[t['task_id'] for t in real], 'batch_size':self.batch})
        times = {}
        for name in policies:
            times[name] = [sum(json.loads(p.read_text())['seconds'] for p in
                (self.output/'timing'/f'repeat_{r}'/name).glob('batch_*.json'))
                for r in range(self.config.get('timing_repeats', 3))]
            for index, _, _ in groups:
                hashes = {json.loads((self.output/'timing'/f'repeat_{r}'/name/f'batch_{index:04d}.json').read_text())['output_sha256']
                          for r in range(self.config.get('timing_repeats', 3))}
                if len(hashes) != 1:
                    raise RuntimeError(f'Fixed-seed timing outputs changed: {name}, batch {index}')
        save(self.output/'timing.json', {'split':split, 'batch_size':self.batch, 'seconds':times,
            'speedup_per_repeat':{n:[a/b for a,b in zip(times['dense'],t)] for n,t in times.items()},
            'note':'Warmed paths, rotating order, same seeds, no history; excludes tokenization and scoring'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['parity','calibration','search','audit','timing','all'])
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text())
    exp=Experiment(config,args.output)
    stages=config.get('stages',['parity','calibration','search','audit']) if args.stage=='all' else [args.stage]
    for stage in stages:
        getattr(exp,stage)()


if __name__=='__main__':
    main()
