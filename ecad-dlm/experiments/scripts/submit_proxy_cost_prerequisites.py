"""Submit Michael's approved proxy/cost studies on a frozen commit; never commits or launches evolution.

The source is frozen first (scripts/slurm/freeze_source.sh) and every job runs that commit
through scripts/slurm/run_frozen.sh. Freezing refuses a run name used before, which guards
against submitting the same studies twice.
"""
from pathlib import Path
import json
import os
import subprocess

ROOT=Path(__file__).resolve().parents[3]
NAME='proxy_cost_20261104_v1'
artifact=Path('/scratch/gqs7zr/ai4mat/ecad/results/dlm_kv_policy/proxy_cost_20261104_v1')
JOBFILE=artifact/'jobs.json'
if JOBFILE.exists():raise SystemExit('jobs.json exists; refusing duplicate submission')
commit=subprocess.check_output([str(ROOT/'scripts/slurm/freeze_source.sh'),NAME],cwd=ROOT,text=True).strip()
env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SBATCH_'))}
# Paths inside the source are relative: each job runs in its own unpacked copy of the commit.
frozen=[str(ROOT/'scripts/slurm/run_frozen.sh'),str(ROOT),commit]
script=['bash','ecad-dlm/experiments/scripts/slurm/run_proxy_cost_prerequisites.sh']
config='ecad-dlm/experiments/configs/diffucoder7b_proxy_recheck_96.yaml'
proxy=artifact/'proxy';cost=artifact/'cost'
(artifact/'logs').mkdir(parents=True,exist_ok=True)
common=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=8',f'--chdir={ROOT}',f'--output={artifact}/logs/%x-%j.out','--export=ALL']
gpu=['--partition=gpu-a100-80','--constraint=a100_80gb','--gres=gpu:1','--mem=96G']
jobs={}
def submit(name,resources,mode,arguments):
    command=common+[f'--job-name={name}']+resources+frozen+script+[mode,'--config',config]+arguments
    result=subprocess.check_output(command,cwd=ROOT,env=env,text=True).strip()
    job=result.split(';')[0]
    if not job.isdigit():raise RuntimeError(result)
    jobs[name]={'job_id':job,'command':command}
    JOBFILE.write_text(json.dumps({'artifact_root':str(artifact),'source_commit':commit,'jobs':jobs},indent=2))
    print(name,job,flush=True);return job
parts=[submit(f'proxy96-p{i}',gpu+['--time=08:00:00'],'proxy',['--output',str(proxy),'--part',str(i),'--parts','4']) for i in range(4)]
costjob=submit('cache-cost-v1',gpu+['--time=04:00:00'],'cost',['--output',str(cost)])
# Scoring is independent of cost success. A second dependent report attaches the
# projections after both studies finish, reusing all cached EvalPlus scores.
report=submit('proxy96-score',['--partition=standard-rivanna','--mem=32G','--time=04:00:00','--dependency=afterok:'+':'.join(parts)],'report',['--output',str(proxy)])
submit('proxy96-cost-report',['--partition=standard-rivanna','--mem=32G','--time=01:00:00',f'--dependency=afterok:{report}:{costjob}'],'report',['--output',str(proxy),'--cost-table',str(cost/'table.json')])
