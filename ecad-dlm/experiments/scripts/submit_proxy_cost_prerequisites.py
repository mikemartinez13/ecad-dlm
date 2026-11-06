"""Submit Michael's approved, frozen proxy/cost studies; never commits or launches evolution."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

ROOT=Path(__file__).resolve().parents[2].parent
SNAP=ROOT/'run_snapshots/proxy_cost_20261104_v1'
JOBFILE=SNAP/'jobs.json'
if JOBFILE.exists():raise SystemExit('jobs.json exists; refusing duplicate submission')
meta=json.loads((SNAP/'SNAPSHOT.json').read_text())
for name,digest in meta['files'].items():
    if hashlib.sha256((SNAP/name).read_bytes()).hexdigest()!=digest:raise SystemExit(f'Snapshot changed: {name}')
env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SBATCH_'))}
script=str(SNAP/'dlm_kv_policy/scripts/slurm/run_proxy_cost_prerequisites.sh')
config=str(SNAP/'dlm_kv_policy/configs/diffucoder7b_proxy_recheck_96.yaml')
artifact=Path('/scratch/gqs7zr/ai4mat/ecad/results/dlm_kv_policy/proxy_cost_20261104_v1')
proxy=artifact/'proxy';cost=artifact/'cost'
common=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=8',f'--chdir={SNAP}',f'--output={SNAP}/logs/%x-%j.out','--export=ALL']
gpu=['--partition=gpu-a100-80','--constraint=a100_80gb','--gres=gpu:1','--mem=96G']
jobs={}
def submit(name,resources,mode,arguments):
    command=common+[f'--job-name={name}']+resources+[script,mode,'--config',config]+arguments
    result=subprocess.check_output(command,cwd=SNAP,env=env,text=True).strip()
    job=result.split(';')[0]
    if not job.isdigit():raise RuntimeError(result)
    jobs[name]={'job_id':job,'command':command};JOBFILE.write_text(json.dumps({'artifact_root':str(artifact),'jobs':jobs},indent=2))
    print(name,job,flush=True);return job
parts=[submit(f'proxy96-p{i}',gpu+['--time=08:00:00'],'proxy',['--output',str(proxy),'--part',str(i),'--parts','4']) for i in range(4)]
costjob=submit('cache-cost-v1',gpu+['--time=04:00:00'],'cost',['--output',str(cost)])
# Scoring is independent of cost success. A second dependent report attaches the
# projections after both studies finish, reusing all cached EvalPlus scores.
report=submit('proxy96-score',['--partition=standard-rivanna','--mem=32G','--time=04:00:00','--dependency=afterok:'+':'.join(parts)],'report',['--output',str(proxy)])
submit('proxy96-cost-report',['--partition=standard-rivanna','--mem=32G','--time=01:00:00',f'--dependency=afterok:{report}:{costjob}'],'report',['--output',str(proxy),'--cost-table',str(cost/'table.json')])
