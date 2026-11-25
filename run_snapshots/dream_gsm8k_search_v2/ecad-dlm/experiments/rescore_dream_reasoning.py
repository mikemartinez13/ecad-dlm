"""Rescore saved responses without rerunning inference or altering original artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
from language_models.reasoning_tasks import load_tasks, score_response, SCORING_VERSION
from experiments.dream_reasoning import summary_pairs
from utils.results_io import save


def rescore(source, data, output):
    audit = json.loads((source/'audit.json').read_text())
    corrected = {}
    source_hashes = {}
    for split, policies in audit.items():
        tasks = {t['task_id']:t for t in load_tasks(data/f'{split}.jsonl')}
        rows_by_policy = {}
        for name in ['dense', *policies]:
            rows = []
            for path in sorted((source/split/name).glob('batch_*.json')):
                source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                for row in json.loads(path.read_text())['tasks']:
                    rows.append({'task_id':row['task_id'], 'response':row['response'],
                        'original_score':row['score'], 'score':score_response(tasks[row['task_id']],row['response'])})
            if len(rows) != len(tasks) or {r['task_id'] for r in rows} != set(tasks):
                raise ValueError(f'Incomplete or duplicated responses: {split}/{name}')
            rows_by_policy[name] = rows
            save(output/split/f'{name}.json',rows)
        corrected[split] = {name:{**old, 'quality':summary_pairs(rows_by_policy['dense'],rows_by_policy[name])}
                            for name,old in policies.items()}
        save(output/'audit.json',corrected)
    save(output/'provenance.json',{'scoring_version':SCORING_VERSION,
         'source':str(source), 'source_sha256':source_hashes,
         'scoring_source_sha256':hashlib.sha256(Path(__file__).parents[1].joinpath('language_models/reasoning_tasks.py').read_bytes()).hexdigest()})
    print(json.dumps({s:{n:r['quality'] for n,r in policies.items()} for s,policies in corrected.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rescore(a.source,a.data,a.output)
