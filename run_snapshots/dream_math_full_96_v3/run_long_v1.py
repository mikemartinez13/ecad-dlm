"""Run a frozen Dream experiment in one bounded allocation, retaining checkpoints."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def save(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2))
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--budget-seconds',type=float,default=71*3600)
    args=parser.parse_args()
    if args.budget_seconds<=0:
        parser.error('--budget-seconds must be positive')
    snapshot=args.snapshot.resolve(); output=args.output.resolve()
    os.chdir(snapshot)
    os.environ.update(PYTHONPATH=str(snapshot/'ecad-dlm')+':'+str(snapshot),
        PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        HF_HOME='/scratch/gqs7zr/cache/huggingface',OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    deadline=time.monotonic()+args.budget_seconds
    def status(state, stage, **extra):
        stamp=datetime.now(timezone.utc).isoformat()
        save(output/'allocation_status.json',dict(status=state,stage=stage,
            updated_utc=stamp,job_id=os.environ.get('SLURM_JOB_ID'),
            account=os.environ.get('SLURM_JOB_ACCOUNT'),budget_seconds=args.budget_seconds,**extra))
        if state!='running':
            path=output/'activity.json'
            activity=json.loads(path.read_text()) if path.exists() else {}
            save(path,{**activity,'status':state,'updated_utc':stamp,**extra})
            subprocess.run([sys.executable,'-c',
                'from pathlib import Path; import sys; from experiments.dream_search_progress import publish; publish(Path(sys.argv[1]))',
                str(output)],check=True)
    if (output/'long_run_complete.json').exists():
        print('Michael, this run already completed all stages.',flush=True)
        return 0
    stages=[]
    if not (output/'ear_calibration.json').exists():
        stages.append('calibration')
    if not (output/'finalists.json').exists():
        stages.append('search')
    stages.append('finish')
    for stage in stages:
        remaining=deadline-time.monotonic()
        if remaining<=0:
            status('checkpointed_time_limit',stage)
            return 0
        status('running',stage)
        print(f'Michael, continuing {stage}; allocation budget remaining {remaining/3600:.2f} hours.',flush=True)
        command=[sys.executable,'-u','-m','experiments.dream_reasoning',stage,
                 '--config',str(snapshot/'config.yaml'),'--output',str(output)]
        process=subprocess.Popen(command,start_new_session=True)
        try:
            code=process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGINT)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL); process.wait()
            status('checkpointed_time_limit',stage,
                note='Completed batches/generations preserved. No automatic continuation submitted.')
            print('Michael, the allocation budget is exhausted. Checkpoints are saved; the search is not marked complete.',flush=True)
            return 0
        if code:
            status('failed',stage,exit_code=code)
            return code if code>0 else 1
    save(output/'long_run_complete.json',{'completed_utc':datetime.now(timezone.utc).isoformat()})
    status('complete','finish')
    return 0


if __name__=='__main__':
    sys.exit(main())
