"""Registered diverse-policy proxy check; CPU preparation/report, GPU task shards."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import numpy as np
import torch
import yaml

from .dkv_decode_refresh_policy import DKVDecodeRefreshPolicy
from .masked_row_reuse_policy import MaskedRowReusePolicy
from .diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend
from .masked_row_reuse_replay import MaskedRowReuseBackend
from .batched_diffucoder_backend import BatchedDiffuCoderKVBackend
from .dense_diffucoder_baseline import DenseDiffuCoderBaseline, BatchedDenseDiffuCoderBaseline
from .diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
from .batched_decoding_study import source_provenance
from .forced_trajectory_replay import replay_task, load_top_p_logits, step_divergence, large_margin_decision_changes

SEED = 20261104
SPEARMAN = 0.90
AUROC = 0.75
MIN_FAILURES = 10
# name, full refresh interval, depth, masked window, protected size, rule
PANEL = [
    ('refresh_256',256,1,None,32,'split+recent'), ('depth_22',8,22,None,32,'split+recent'),
    ('refresh_256_r2',256,1,2,32,'split+recent'), ('refresh_256_r4',256,1,4,32,'split+recent'),
    ('depth_22_r2',8,22,2,32,'split+recent'), ('depth_22_r4',8,22,4,32,'split+recent'),
    ('depth_22_r4_m16',8,22,4,16,'split+recent'), ('depth_22_r4_m64',8,22,4,64,'split+recent'),
    ('depth_22_r8',8,22,8,32,'split+recent'), ('depth_22_r4_split',8,22,4,32,'split'),
    ('depth_18_r4',8,18,4,32,'split+recent'), ('depth_24_r4',8,24,4,32,'split+recent'),
]


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=str, allow_nan=False))
    temporary.replace(path)


def policies(stop_ids):
    result = {}
    for name, interval, depth, window, m, rule in PANEL:
        base = DKVDecodeRefreshPolicy(name=name, cache_reloading_step=interval, refresh_depth=depth)
        result[name] = base if window is None else MaskedRowReusePolicy(base,window,m,rule,stop_ids)
    return result


def backend(model, policy, batched=False):
    if batched: return BatchedDiffuCoderKVBackend(model,policy)
    return (MaskedRowReuseBackend if isinstance(policy,MaskedRowReusePolicy) else DiffuCoderActiveTokenKVBackend)(model,policy)


def category(prompt):
    # Fixed lexical tags, used only for balancing; no outputs or tests consulted.
    patterns = [('strings',r'string|character|substring|palindrome|word'),
                ('collections',r'list|tuple|array|dictionary|set of|sort'),
                ('parsing',r'parse|format|expression|parenthes|bracket|file'),
                ('numerical',r'number|integer|prime|digit|sum|product|factor|divis|float')]
    return next((name for name,pattern in patterns if re.search(pattern,prompt,re.I)), 'control_other')


def prepare(config_path, audit_path, manifest_path, output_config):
    from transformers import AutoTokenizer
    from ecad.language_models.diffucoder_loader import prepare_diffucoder_prompt_batch
    from ecad.language_models.code_tasks_and_response_processing import CodeGenerationTask, render_strict_python_prompt
    raw=yaml.safe_load(config_path.read_text()); audit=json.loads(audit_path.read_text())
    seen=set(audit['seen_ids'])
    tokenizer=AutoTokenizer.from_pretrained(raw['model']['path'],trust_remote_code=True,local_files_only=True)
    tokenizer.padding_side='left'
    rng=random.Random(SEED); selected=[]; details={}; all_ids=set()
    for dataset, filename in [('humaneval','HumanEvalPlus.jsonl'),('mbpp','MbppPlus.jsonl')]:
        items=[]
        for line in (Path('/scratch/gqs7zr/evalplus_data')/filename).read_text().splitlines():
            rec=json.loads(line); task_id=rec['task_id']; all_ids.add(task_id)
            if task_id in seen: continue
            original=rec['prompt'].strip(); prompt,mode=render_strict_python_prompt(dataset,original,task_id)
            task=CodeGenerationTask(task_id,dataset,prompt,original,mode,entry_point=rec.get('entry_point'))
            tokens,_=prepare_diffucoder_prompt_batch(tokenizer,[prompt],'cpu')
            items.append((tokens.shape[1],task_id,category(original),task))
        items.sort(key=lambda t:(t[0],t[1]))
        for tertile, indices in enumerate(np.array_split(np.arange(len(items)),3)):
            groups=defaultdict(list)
            for i in indices: groups[items[i][2]].append(items[i])
            for values in groups.values(): rng.shuffle(values)
            chosen=[]
            while len(chosen)<16:
                progress=False
                for key in sorted(groups):
                    if groups[key] and len(chosen)<16:
                        chosen.append(groups[key].pop()); progress=True
                if not progress: raise ValueError('Not enough fresh tasks in stratum')
            for length,task_id,tag,task in chosen:
                selected.append(task);details[task_id]={'prompt_tokens':length,'category':tag,'length_tertile':tertile,'dataset':dataset}
    selected.sort(key=lambda t:(details[t.task_id]['length_tertile'],t.dataset,t.task_id))
    if len(selected)!=96 or len({t.task_id for t in selected})!=96: raise ValueError('Task count mismatch')
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    manifest_path.write_text(''.join(json.dumps(t.to_dict())+'\n' for t in selected))
    save(manifest_path.with_suffix('.selection.json'), {'seed':SEED,'audit_root':audit['root'],
        'audit_files':audit['files_scanned'],'previously_exposed':sorted(seen),'selected':details,
        'remaining_unexposed_ids':sorted(all_ids-seen-set(details)),
        'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest()})
    save(manifest_path.with_suffix('.exposure.json'),audit)
    raw['run']={'name':'diffucoder_proxy_recheck_20261104_v1','resume':True}
    raw['evaluation']['code_tasks_file']=str(manifest_path.resolve());raw['evaluation']['max_prompts']=96
    raw.pop('variance_study',None)
    output_config.write_text(yaml.safe_dump(raw,sort_keys=False))
    print('Prepared',len(selected),'tasks;',Counter(t.dataset for t in selected), 'lengths',min(d['prompt_tokens'] for d in details.values()),max(d['prompt_tokens'] for d in details.values()))


def generation_settings(config):
    g=config.generation
    return {k:getattr(g,k) for k in ('steps','gen_length','temperature','top_p','top_k','algorithm','algorithm_temperature')}


def assert_same(a,b,label):
    if len(a.history)!=len(b.history) or any(not torch.equal(x,y) for x,y in zip(a.history,b.history)):
        raise RuntimeError(f'Batch-one identity failed: {label}')


def run(config_path, output, part, parts):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from .code_correctness_and_latency_evaluator import DiffuCoderKVPolicyEvaluator, seed_everything
    from .kv_policy_experiment_config import load_kv_policy_experiment_config
    from ecad.language_models.diffucoder_loader import prepare_diffucoder_prompt_batch
    from ecad.language_models.code_tasks_and_response_processing import make_code_task_output
    config=load_kv_policy_experiment_config(config_path)
    ev=DiffuCoderKVPolicyEvaluator(config);model,tok=ev.model,ev.tokenizer
    if torch.cuda.get_device_name()!='NVIDIA A100-SXM4-80GB': raise RuntimeError('Study requires A100 SXM4 80GB')
    stop={x for x in (tok.eos_token_id,tok.pad_token_id) if x is not None}
    provenance={**source_provenance(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'config':asdict(config),'part':part,'parts':parts,'panel':PANEL,
        'manifest_sha256':hashlib.sha256(Path(config.evaluation.code_tasks_file).read_bytes()).hexdigest()}
    signature=hashlib.sha256(json.dumps({k:provenance[k] for k in ('source_sha256','config','panel','manifest_sha256')},sort_keys=True,default=str).encode()).hexdigest()
    provenance['signature']=signature
    meta=output/f'part_{part}_provenance.json'
    if meta.exists() and json.loads(meta.read_text())['signature']!=signature: raise RuntimeError('Refusing mixed-source resume')
    save(meta,provenance)
    settings=generation_settings(config)
    reference_sampler=DiffuCoderDenoisingSampler(DenseDiffuCoderBaseline(model))
    for index in range(part,len(ev.tasks),parts):
        path=output/'tasks'/f'task_{index:03d}.json'
        if path.exists():
            if json.loads(path.read_text())['signature']!=signature: raise RuntimeError('Task signature mismatch')
            continue
        task=ev.tasks[index];seed_everything(config.generation.seed+index)
        prompt,mask=prepare_diffucoder_prompt_batch(tok,[task.prompt],config.model.device)
        candidates=policies(stop)
        replay_backends={name:backend(model,p) for name,p in candidates.items()}
        # Identity sentinel verifies the wrapper with r=1 on both committed-row bases.
        for name in ('refresh_256','depth_22'):
            replay_backends[name+'_check_r1']=backend(model,MaskedRowReusePolicy(candidates[name],1,32,'split+recent',stop))
        row_counts={name:[] for name in candidates}
        for name in candidates:
            original=replay_backends[name].forward_mask_logits
            def record_forward(_original=original,_name=name,**kw):
                result=_original(**kw);d=result.decision
                row_counts[_name].append({'active':int(d.active.sum()),'full_depth':int((d.active & d.refresh_depth.le(1)).sum()),'masked':result.mask_logits.shape[0]})
                return result
            replay_backends[name].forward_mask_logits=record_forward
        print(f'part {part}: task {index} replay',flush=True)
        replay=replay_task(model,reference_sampler,replay_backends,prompt,mask,config.generation,load_top_p_logits(model))
        from .stale_kv_error_study import same_steps
        for name in ('refresh_256','depth_22'):
            if not same_steps(replay['policies'][name],replay['policies'][name+'_check_r1']): raise RuntimeError('r1 mismatch')
        for instance in replay_backends.values():
            instance.__dict__.pop('forward_mask_logits',None)
        del instance, original, record_forward, replay_backends
        generations={}; dense_result=None
        for name in ['dense','dense_repeat','dense_math',*candidates]:
            print(f'part {part}: task {index} generation {name}',flush=True)
            attention=sdpa_kernel(SDPBackend.MATH) if name=='dense_math' else nullcontext()
            with attention:
                if name.startswith('dense'):
                    result=reference_sampler.generate(prompt,attention_mask=mask,collect_history=True,**settings)
                else:
                    ref=DiffuCoderDenoisingSampler(backend(model,policies(stop)[name]))
                    actual=DiffuCoderDenoisingSampler(backend(model,policies(stop)[name],batched=True))
                    expected=ref.generate(prompt,attention_mask=mask,collect_history=True,**settings)
                    result=actual.generate(prompt,attention_mask=mask,collect_history=True,**settings)
                    assert_same(result,expected,name)
                    del ref,actual,expected
            if name=='dense':
                dense_result=result
                dense_batched=DiffuCoderDenoisingSampler(BatchedDenseDiffuCoderBaseline(model)).generate(prompt,attention_mask=mask,collect_history=True,**settings)
                assert_same(result,dense_batched,'dense');del dense_batched
            elif name=='dense_repeat': assert_same(result,dense_result,name)
            ids=result.sequences[0,prompt.shape[1]:].cpu().tolist()
            if name=='dense' and ids!=replay['dense_token_ids']: raise RuntimeError('Dense replay/generation mismatch')
            response=tok.decode(ids,skip_special_tokens=True)
            generations[name]={'token_ids':ids,'output':make_code_task_output(task,response)}
        record={'signature':signature,'task_id':task.task_id,'index':index,'prompt_tokens':prompt.shape[1],
            'gpu':torch.cuda.get_device_name(),'stop_ids':sorted(stop),'replay':replay,'rows':row_counts,'generations':generations,
            'identity_passed':True}
        save(path,record);print(f'part {part}: {index} {task.task_id} complete',flush=True)


def rank_score(x,y):
    from scipy.stats import spearmanr
    if len(set(x))<2 or len(set(y))<2: return None
    return float(spearmanr(x,y).statistic)


def auc_score(scores,labels):
    from scipy.stats import rankdata
    labels=np.asarray(labels,dtype=bool);scores=np.asarray(scores)
    n=int(labels.sum());m=len(labels)-n
    if n==0 or m==0:return None
    return float((rankdata(scores)[labels].sum()-n*(n+1)/2)/(n*m))


def assessment(proxy, task_proxy, passes, reference, indices):
    # Arrays [policy,task]; bootstrap indices preserve clusters of policy outcomes.
    x=proxy[:,indices].mean(axis=1)
    ref=reference[indices];p=passes[:,indices]
    losses=ref[None,:] & ~p;gains=~ref[None,:] & p
    rank=rank_score(x,losses.sum(axis=1))
    scores=task_proxy[:,indices][:,ref].ravel();labels=losses[:,ref].ravel()
    auc=auc_score(scores,labels)
    a='INCONCLUSIVE' if rank is None else ('PASS' if rank>=SPEARMAN else 'FAIL')
    b='INCONCLUSIVE' if int(losses.sum())<MIN_FAILURES or auc is None else ('PASS' if auc>=AUROC else 'FAIL')
    verdict='FAIL' if 'FAIL' in (a,b) else ('PASS' if a==b=='PASS' else 'INCONCLUSIVE')
    return {'spearman':rank,'auroc':auc,'ranking_verdict':a,'task_verdict':b,'verdict':verdict,
        'regressions':int(losses.sum()),'distinct_regression_tasks':int(losses.any(0).sum()),
        'per_policy_regressions':losses.sum(1).tolist(),'per_policy_recoveries':gains.sum(1).tolist()}


def evaluate(config_path, output, cost_table=None):
    from .kv_policy_experiment_config import load_kv_policy_experiment_config
    from .validate_proxy import base_statuses
    from ecad.language_models.code_execution import run_evalplus_apptainer
    config=load_kv_policy_experiment_config(config_path)
    records=[json.loads(p.read_text()) for p in sorted((output/'tasks').glob('task_*.json'))]
    if len(records)!=96 or len({r['task_id'] for r in records})!=96:raise RuntimeError('Need exactly 96 completed tasks')
    if len({r['signature'] for r in records})!=1: raise RuntimeError('Mixed source/task signatures')
    if not all(r.get('identity_passed') for r in records):raise RuntimeError('Identity precondition not met')
    names=[r[0] for r in PANEL];all_names=['dense','dense_repeat','dense_math',*names]
    statuses={};scores={}
    for name in all_names:
        for scope in ('base','full'):
            path=output/'scores'/f'{name}_{scope}.json'
            if path.exists():result=json.loads(path.read_text())
            else:
                e=config.evaluation
                result=run_evalplus_apptainer([r['generations'][name]['output'] for r in records],
                    image=e.evalplus_image,work_dir=output/'execution'/f'{name}_{scope}',parallel=e.evalplus_parallel,
                    apptainer_bin=e.evalplus_apptainer_bin,scope=scope,timeout_seconds=e.evalplus_timeout_seconds,
                    dataset_paths={'humaneval':e.evalplus_humaneval_path,'mbpp':e.evalplus_mbpp_path})
                save(path,result)
            if scope=='base':
                status=base_statuses({'code_execution':result})
                statuses[name]=np.array([status[r['task_id']]=='pass' for r in records])
            scores[name+'_'+scope]={'aggregate':result['aggregate'],'datasets':{k:v['aggregate'] for k,v in result['datasets'].items()}}
    proxy=np.array([[sum(step_divergence(s) for s in r['replay']['policies'][name]) for r in records] for name in names])
    task_proxy=np.array([[large_margin_decision_changes(r['replay']['policies'][name]) for r in records] for name in names])
    passes=np.array([statuses[name] for name in names]);anchor=names.index('depth_22')
    local=[i for i,n in enumerate(names) if n.startswith('depth_') and n!='depth_22']
    summary={'thresholds':{'spearman':SPEARMAN,'auroc':AUROC,'minimum_failures':MIN_FAILURES},'scores':scores,'views':{}}
    rng=np.random.default_rng(SEED)
    for label,pr,tp,pa,ref,view_names in [
        ('overall',proxy,task_proxy,passes,statuses['dense'],names),
        ('depth_family',proxy[local]-proxy[anchor],task_proxy[local]-task_proxy[anchor],passes[local],statuses['depth_22'],[names[i] for i in local])]:
        value=assessment(pr,tp,pa,ref,np.arange(96));value['policies']=view_names
        boot=[assessment(pr,tp,pa,ref,rng.integers(0,96,96)) for _ in range(2000)]
        for metric in ('spearman','auroc'):
            vals=[b[metric] for b in boot if b[metric] is not None]
            value[metric+'_ci95']=np.quantile(vals,[.025,.975]).tolist() if vals else None
            value[metric+'_valid_bootstraps']=len(vals)
        summary['views'][label]=value
    summary['verdict']='FAIL' if any(v['verdict']=='FAIL' for v in summary['views'].values()) else ('PASS' if all(v['verdict']=='PASS' for v in summary['views'].values()) else 'INCONCLUSIVE')
    summary['dense_kernel_noise']={'pass_to_fail':int((statuses['dense'] & ~statuses['dense_math']).sum()),'fail_to_pass':int((~statuses['dense'] & statuses['dense_math']).sum())}
    summary['policy_details']={}
    for i,name in enumerate(names):
        answer_total=padding_total=answer_out=padding_out=0
        for r in records:
            ids=r['replay']['dense_token_ids'];stops=set(r['stop_ids'])
            end=next((j for j,t in enumerate(ids) if t in stops),len(ids))
            answer_total+=end;padding_total+=len(ids)-end
            for row in r['replay']['policies'][name]:
                for pos in row.get('outside_positions',[]):
                    if pos<end:answer_out+=1
                    else:padding_out+=1
        summary['policy_details'][name]={'mean_divergence':float(proxy[i].mean()),
            'compute_fraction':sum(r['replay']['flops'][name] for r in records)/sum(r['replay']['dense_flops'] for r in records),
            'answer_coverage':1-answer_out/answer_total if answer_total else None,
            'padding_coverage':1-padding_out/padding_total if padding_total else None}
    if cost_table is not None:
        from .batched_cost_model import project_records
        table=json.loads(cost_table.read_text())
        summary['cost_projections']={str(b):project_records(table,records,b) for b in (1,8,16)}
    save(output/'proxy_report.json',summary);print(json.dumps(summary['views'],indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run','report'])
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=4)
    p.add_argument('--cost-table',type=Path)
    p.add_argument('--audit',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--output-config',type=Path)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.config,a.audit,a.manifest,a.output_config)
    elif a.mode=='run':
        if not 0<=a.part<a.parts: p.error('Invalid shard')
        run(a.config,a.output,a.part,a.parts)
    else:evaluate(a.config,a.output,a.cost_table)
if __name__=='__main__': main()
