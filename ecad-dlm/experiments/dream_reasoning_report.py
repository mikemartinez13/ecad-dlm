"""CPU-only reporting of the Dream pilot; never feeds OOD scores into search."""
import argparse
import json
from pathlib import Path
from utils.ranking import rank_score
from utils.results_io import save


def report(root):
    panel = json.loads((root/'calibration_panel.json').read_text())
    audit = json.loads((root/'audit.json').read_text())
    correlations = {}
    for metric in ('exact_match','flexible_extract'):
        x = [p['ear'] for p in panel.values()]
        y = [p['quality'][metric]['delta'] for p in panel.values()]
        correlations[metric] = rank_score(x,y)
    primary = {}
    for split, policies in audit.items():
        metric = 'flexible_extract' if split.startswith('gsm8k') else 'math_verify'
        primary[split] = {name:{**r['quality'][metric], 'metric':metric,
            'measured_speedup':r['measured_speedup'], 'tasks':r['tasks']}
            for name,r in policies.items()}
    result = {'calibration_ear_accuracy_spearman':correlations, 'primary_audit':primary,
        'ear_constraints':json.loads((root/'ear_calibration.json').read_text()),
        'interpretation':[
            'GSM8K primary is flexible numerical extraction; strict phrase match is retained separately.',
            'MATH primary is math_verify; use scoring_v2 rescoring for legacy pilot outputs.',
            'Full metric vectors and paired intervals remain in audit.json.',
            'Pilot sample sizes and one timing pass do not establish approximate losslessness or stable speedup.',
            'MATH-500 audit has not influenced candidate generation or selection.'
        ]}
    save(root/'report.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    report(parser.parse_args().output)
