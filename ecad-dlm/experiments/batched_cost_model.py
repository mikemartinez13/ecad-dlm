"""Component timing table and held-out schedule calibration using the batched backend.

No search or pass/fail selection occurs here. Table construction precedes and is
independent of calibration. Model-compute predictions omit policy/sampler overhead;
calibration includes the backend's indexing/selection overhead and may reject them.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
from statistics import median
import numpy as np
import torch
from .batched_row_timing import BatchedMinimalForward
from dlm_kv_policy.batched_diffucoder_backend import BatchedDiffuCoderKVBackend
from dlm_kv_policy.dkv_decode_refresh_policy import DKVDecodeRefreshPolicy
from dlm_kv_policy.fixed_token_refresh_policy import TokenRefreshDecision
from dlm_kv_policy.denoising_token_state import DenoisingTokenState
from dlm_kv_policy.diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
from .batched_decoding_study import source_provenance
from .proxy_recheck import save, rank_score

BLOCKS = ((0,17),(17,21),(21,23),(23,28))
ROWS = (0,1,2,4,8,16,24,32,48,64,96,128,160,192,224,256)
# depth, full-depth rows, total deep rows, request-count spread, refresh interval
CASES = ((1,12,12,0,4),(1,40,40,24,8),(1,80,80,0,4),(1,144,144,48,8),
         (18,12,80,24,4),(18,40,144,48,8),(18,80,224,0,4),
         (22,12,80,0,8),(22,40,144,48,4),(22,80,224,24,8),
         (24,24,144,48,4),(24,112,224,0,8))


def interpolate(curve, rows):
    x=sorted(map(int,curve));
    if rows<x[0] or rows>x[-1]:raise ValueError(f'Row count {rows} outside calibrated grid')
    return float(np.interp(rows,x,[curve[str(i)] for i in x]))


def predict_step(table, batch, length, depth, full_rows, active_rows, masked_rows, masked=True):
    key=f'{batch}:{length}:{int(masked)}'
    if key not in table['curves']:raise ValueError(f'Unmeasured batch/context/mask shape: {key}')
    if depth not in (1,18,22,24):raise ValueError('Depth has no calibrated block boundary')
    curve=table['curves'][key]
    total=interpolate(curve['head'],masked_rows)
    for start,end in BLOCKS:
        count=max(full_rows) if end<=depth-1 else max(active_rows)
        total+=interpolate(curve[f'{start}:{end}'],count)
    return total


def measurements(fn, warmup=2, repeats=5):
    with torch.inference_mode():
        for _ in range(warmup):fn()
        torch.cuda.synchronize()
        events=[];walls=[]
        for _ in range(repeats):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            wall=perf_counter();start.record();fn();end.record();torch.cuda.synchronize()
            events.append(start.elapsed_time(end)/1000);walls.append(perf_counter()-wall)
    return {'seconds':median(events),'samples':events,'wall_seconds':median(walls)}


def build_table(model, lengths, output):
    table={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'lengths':lengths,
           'blocks':BLOCKS,'curves':{},'samples':{}}
    device=next(model.parameters()).device;dtype=next(model.parameters()).dtype
    for batch in (1,8,16):
        for length in lengths:
            forward=BatchedMinimalForward(model,length,batch)
            for masked in (False,True):
                key=f'{batch}:{length}:{int(masked)}';curve={};raw={}
                keymask=torch.ones(batch,1,1,length,device=device,dtype=torch.bool) if masked else None
                if masked:
                    for b in range(batch):keymask[b,:,:,:1+(b%8)*4]=False
                for start,end in BLOCKS:
                    name=f'{start}:{end}';curve[name]={'0':0.0};raw[name]={}
                    for rows in sorted(set([r for r in ROWS if 0<r<=length]+[length])):
                        result=measurements(lambda:forward(rows,start_layer=start,end_layer=end,project_head=False,attention_mask=keymask))
                        curve[name][str(rows)]=result['seconds'];raw[name][str(rows)]=result
                head={};head_raw={}
                for rows in ROWS:
                    if rows==0:head['0']=0.0;continue
                    hidden=torch.randn(batch,rows,model.config.hidden_size,device=device,dtype=dtype)
                    result=measurements(lambda:model.lm_head(hidden))
                    head[str(rows)]=result['seconds'];head_raw[str(rows)]=result
                curve['head']=head;raw['head']=head_raw
                table['curves'][key]=curve;table['samples'][key]=raw
                save(output/'table.json',table);print('table',key,flush=True)
            del forward
    return table


class SchedulePolicy(DKVDecodeRefreshPolicy):
    """Fixed synthetic decisions for calibration only, never code generation."""
    def __init__(self, depth, decisions):
        super().__init__(refresh_depth=depth)
        object.__setattr__(self,'decisions',decisions)
    def decide(self, *, states, step, **kwargs):
        if step==0:
            return TokenRefreshDecision(states.ne(0),torch.ones_like(states),{'initial_full':0})
        return self.decisions[step-1]


def schedule_inputs(batch,length,case,mask_token,device):
    depth,nfull,ndeep,spread,refresh=case
    prompt_width=length-256
    pad=torch.ones(batch,prompt_width,dtype=torch.long,device=device)
    for b in range(batch):pad[b,:min(1+b*3,prompt_width-1)]=0
    tokens=torch.full((batch,length),7,dtype=torch.long,device=device)
    tokens[:,:prompt_width].masked_fill_(pad.eq(0),0);tokens[:,prompt_width:]=mask_token
    state=DenoisingTokenState.create(prompt_width,length,pad,mask_token,device=device)
    _,attention,positions=DiffuCoderDenoisingSampler._attention_inputs(tokens[:,:prompt_width],pad,256)
    initial=dict(input_ids=tokens.clone(),attention_mask=attention,position_ids=positions,state=state,step=0,total_steps=9)
    frames=[];decisions=[];counts=[]
    for step in range(1,9):
        x=tokens.clone();x[:,prompt_width:prompt_width+7*step]=7
        active=torch.zeros_like(tokens,dtype=torch.bool);full=torch.zeros_like(active)
        nf=[];nd=[]
        for b in range(batch):
            valid=torch.where(state.valid_mask[b])[0]
            # Per-request different row positions, even when counts agree.
            order=valid.roll(b+step)
            d=min(len(order),max(0,ndeep-(b*spread//max(1,batch-1))))
            f=min(d,max(0,nfull-(b*spread//max(1,batch-1))))
            if step%refresh==0:f=d=len(order)
            if depth==1:f=d
            active[b,order[:d]]=True;full[b,order[:f]]=True;nf.append(f);nd.append(d)
        depths=torch.full_like(tokens,depth);depths[full]=1
        decisions.append(TokenRefreshDecision(active,depths,{'synthetic':0}))
        frames.append(dict(input_ids=x,attention_mask=attention,position_ids=positions,state=state,step=step,total_steps=9))
        counts.append((nf,nd,256-7*step))
    return initial,frames,decisions,counts


def calibration_metrics(predicted,observed):
    predicted=np.asarray(predicted);observed=np.asarray(observed)
    if np.any(observed<=0):raise ValueError('Calibration times must be positive')
    error=np.abs(predicted-observed)/observed
    rho=rank_score(predicted.tolist(),observed.tolist())
    return {'median_absolute_relative_error':float(np.median(error)),
        'p90_absolute_relative_error':float(np.quantile(error,.9)), 'spearman':rho,
        'passed':bool(np.median(error)<=.10 and np.quantile(error,.9)<=.20 and rho is not None and rho>=.95)}


def calibrate(model,table,output):
    records=[];device=next(model.parameters()).device
    # Distinct synthetic query counts are held out from the interpolation grid.
    for batch in (1,8,16):
        for length in sorted(set([table['lengths'][len(table['lengths'])//2],table['lengths'][-1]])):
            for case_id,case in enumerate(CASES):
                initial,frames,decisions,counts=schedule_inputs(batch,length,case,model.config.mask_token_id,device)
                policy=SchedulePolicy(case[0],decisions);backend=BatchedDiffuCoderKVBackend(model,policy)
                backend.forward_mask_logits(**initial)
                def execute():
                    for frame in frames:backend.forward_mask_logits(**frame)
                torch.cuda.reset_peak_memory_stats()
                measured=measurements(execute,warmup=2,repeats=5)
                predicted=sum(predict_step(table,batch,length,case[0],f,a,m) for f,a,m in counts)
                records.append({'batch':batch,'length':length,'case':case_id,'parameters':case,
                    'predicted_seconds':predicted,'measured':measured,'rows':counts,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
                save(output/'calibration_records.json',records)
                print('calibration',batch,length,case_id,round(predicted/measured['seconds'],3),flush=True)
                del backend
    result={'interpretation':'table excludes request indexing; calibration includes actual backend overhead; no sampler',
            'by_batch':{str(b):calibration_metrics([r['predicted_seconds'] for r in records if r['batch']==b],
                        [r['measured']['seconds'] for r in records if r['batch']==b]) for b in (1,8,16)}}
    result['passed']=all(v['passed'] for v in result['by_batch'].values())
    save(output/'calibration_report.json',result)
    return result


def smoke(model,tokenizer,tasks,config,output):
    from language_models.diffucoder_loader import prepare_diffucoder_prompt_batch
    from .proxy_recheck import policies,backend,assert_same,generation_settings
    from dlm_kv_policy.dense_diffucoder_baseline import DenseDiffuCoderBaseline,BatchedDenseDiffuCoderBaseline
    settings=generation_settings(config);settings['collect_history']=True
    stop={x for x in (tokenizer.eos_token_id,tokenizer.pad_token_id) if x is not None}
    prompt,mask=prepare_diffucoder_prompt_batch(tokenizer,[tasks[0].prompt],config.model.device)
    ref=DiffuCoderDenoisingSampler(DenseDiffuCoderBaseline(model)).generate(prompt,attention_mask=mask,**settings)
    actual=DiffuCoderDenoisingSampler(BatchedDenseDiffuCoderBaseline(model)).generate(prompt,attention_mask=mask,**settings)
    assert_same(ref,actual,'dense')
    for name in policies(stop):
        ref=DiffuCoderDenoisingSampler(backend(model,policies(stop)[name])).generate(prompt,attention_mask=mask,**settings)
        actual=DiffuCoderDenoisingSampler(backend(model,policies(stop)[name],True)).generate(prompt,attention_mask=mask,**settings)
        assert_same(ref,actual,name)
    # Padding integration check: dense and fully refreshed cached backend must
    # produce close logits for valid rows before calibration uses that path.
    for b in (8,16):
        prompt,mask=prepare_diffucoder_prompt_batch(tokenizer,[t.prompt for t in tasks[:b]],config.model.device)
        x=torch.nn.functional.pad(prompt,(0,256),value=model.config.mask_token_id)
        state=DenoisingTokenState.create(prompt.shape[1],x.shape[1],mask,model.config.mask_token_id,device=x.device)
        _,attention,pos=DiffuCoderDenoisingSampler._attention_inputs(prompt,mask,256)
        inputs=dict(input_ids=x,attention_mask=attention,position_ids=pos,state=state,step=0,total_steps=256)
        a=BatchedDenseDiffuCoderBaseline(model).forward_mask_logits(**inputs).mask_logits
        z=BatchedDiffuCoderKVBackend(model,DKVDecodeRefreshPolicy(cache_reloading_step=1)).forward_mask_logits(**inputs).mask_logits
        if not torch.isfinite(a).all() or not torch.isfinite(z).all():raise RuntimeError('Nonfinite padded logits')
        relative=float(torch.linalg.vector_norm((a-z).float())/torch.linalg.vector_norm(a.float()))
        if relative>.01:raise RuntimeError(f'Full-forward padded logit relative error {relative} exceeds 0.01')
    save(output/'smoke.json',{'batch_one_identity':True,'padded_forward_finite_and_relative_l2_at_most_0.01':True,
        'note':'Engineering check only, not a pass@1 batch-invariance claim'})



def project_records(table, records, batch):
    """Sum costs of batch maxima; context is conservatively rounded up to a table bucket."""
    from .proxy_recheck import PANEL
    ordered=sorted(records,key=lambda r:(r['prompt_tokens'],r['task_id']))
    if len(ordered)%batch:raise ValueError('Explicit partial-batch calibration required')
    totals={name:0.0 for name,*_ in PANEL};dense=0.0;manifest=[]
    for offset in range(0,len(ordered),batch):
        group=ordered[offset:offset+batch];length=max(r['prompt_tokens']+256 for r in group)
        bucket=next((n for n in table['lengths'] if n>=length),None)
        if bucket is None:raise ValueError('Context exceeds calibrated table')
        manifest.append({'tasks':[r['task_id'] for r in group],'actual_length':length,'cost_key_length':bucket})
        for step in range(256):
            masked={r['rows'][PANEL[0][0]][step]['masked'] for r in group}
            if len(masked)!=1:raise ValueError('Requests have different commit schedules')
            m=masked.pop()
            dense+=predict_step(table,batch,bucket,1,[length]*batch,[length]*batch,m,masked=True)
            for name,interval,depth,window,size,rule in PANEL:
                full=[r['rows'][name][step]['full_depth'] for r in group]
                active=[r['rows'][name][step]['active'] for r in group]
                totals[name]+=predict_step(table,batch,bucket,depth,full,active,m,masked=True)
    return {'batch':batch,'batches':manifest,'predicted_dense_seconds':dense,
        'policies':{name:{'predicted_seconds':seconds,'predicted_speedup':dense/seconds} for name,seconds in totals.items()},
        'scope':'model-compute projection, not measured end-to-end speed'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from dlm_kv_policy.kv_policy_experiment_config import load_kv_policy_experiment_config
    from dlm_kv_policy.code_correctness_and_latency_evaluator import DiffuCoderKVPolicyEvaluator
    config=load_kv_policy_experiment_config(args.config)
    ev=DiffuCoderKVPolicyEvaluator(config)
    if torch.cuda.get_device_name()!='NVIDIA A100-SXM4-80GB':raise RuntimeError('Requires A100 SXM4 80GB')
    if args.output.exists() and any(args.output.iterdir()):raise RuntimeError('Cost study output must be empty')
    selection=json.loads(Path(config.evaluation.code_tasks_file).with_suffix('.selection.json').read_text())
    longest=max(d['prompt_tokens']+256 for d in selection['selected'].values())
    lengths=list(range(384,((longest+127)//128)*128+1,128))
    save(args.output/'provenance.json',{**source_provenance(),'config':str(args.config),'lengths':lengths,'cases':CASES,
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name()})
    smoke(ev.model,ev.tokenizer,ev.tasks,config,args.output)
    table=build_table(ev.model,lengths,args.output)
    print(calibrate(ev.model,table,args.output),flush=True)
if __name__=='__main__':main()
