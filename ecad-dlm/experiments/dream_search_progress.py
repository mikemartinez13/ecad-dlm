"""Inspect a running Dream search without loading torch, a model, or a GPU."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from utils.results_io import save


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def summarize(root):
    provenance=read(root/'provenance.json', {})
    config=provenance.get('config', read(root/'run_config.json', {}))
    generations=[]
    for path in sorted((root/'search').glob('generation_*/summary.json')):
        data=read(path)
        population=data['population']
        feasible=[p for p in population if p['violation']<=0]
        best=min(feasible,key=lambda p:p['objectives']['cost']) if feasible else None
        c=best['constraints'] if best else {}
        generations.append({'generation':int(path.parent.name.split('_')[-1])+1,
            'evaluated':len(data['evaluated']), 'feasible_survivors':len(feasible),
            'best_flops_fraction':best['objectives']['cost'] if best else None,
            'best_ear':c.get('ear'), 'best_ear_ci_lower':c.get('ear_95_ci',[None])[0],
            'best_tail_ear':c.get('tail_ear'), 'best_policy':best['policy_name'] if best else None,
            'seconds':data.get('elapsed_seconds')})
    partial=[]
    for path in sorted((root/'search').glob('generation_*/partial.json')):
        if not (path.parent/'summary.json').exists():
            partial.append({'generation':int(path.parent.name.split('_')[-1])+1,
                            'evaluated':len(read(path))})
    return {'updated_utc':datetime.now(timezone.utc).isoformat(),
        'target_generations':config.get('search',{}).get('generations'),
        'population_size':config.get('search',{}).get('population_size'),
        'completed_generations':len(generations),
        'evaluated_candidates':sum(g['evaluated'] for g in generations)+sum(p['evaluated'] for p in partial),
        'activity':read(root/'activity.json',{}), 'calibration':read(root/'ear_calibration.json'),
        'partial_generations':partial, 'history':generations,
        'note':'EAR and estimated padded FLOPs are search metrics; task accuracy and measured speed are separate audits.'}


def render(report):
    activity=report['activity']
    lines=['Michael, Dream search progress', '',
        f"Generations: {report['completed_generations']}/{report['target_generations']}; population: {report['population_size']}; evaluated candidates: {report['evaluated_candidates']}",
        f"Activity: {json.dumps(activity,sort_keys=True)}", '',
        '| Generation | Feasible survivors | Best FLOPs / dense | EAR | EAR CI lower | Tail EAR |',
        '| --- | --- | --- | --- | --- | --- |']
    for g in report['history']:
        values=[str(g['generation']), str(g['feasible_survivors'])]
        values += ['—' if g[k] is None else f'{g[k]:.5f}' for k in
                   ('best_flops_fraction','best_ear','best_ear_ci_lower','best_tail_ear')]
        lines.append('| '+' | '.join(values)+' |')
    lines += ['', report['note']]
    return '\n'.join(lines)+'\n'


def publish(root):
    report=summarize(root)
    save(root/'progress.json',report)
    for name,text in [('progress.md',render(report))]:
        path=root/name
        tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(text); tmp.replace(path)
    if report['history']:
        buf=io.StringIO()
        writer=csv.DictWriter(buf,fieldnames=list(report['history'][0]))
        writer.writeheader(); writer.writerows(report['history'])
        path=root/'generation_history.csv'
        tmp=path.with_suffix('.csv.tmp'); tmp.write_text(buf.getvalue()); tmp.replace(path)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--json',action='store_true')
    args=parser.parse_args()
    report=summarize(args.output)
    print(json.dumps(report,indent=2) if args.json else render(report))
