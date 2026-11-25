"""Writing result files and recording which source produced them."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=str, allow_nan=False))
    temporary.replace(path)


def source_commit():
    """The frozen run's commit (scripts/slurm/run_frozen.sh), or HEAD of the checkout this runs from."""
    frozen = os.environ.get('ECAD_SOURCE_COMMIT')
    if frozen:
        return frozen
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()


def source_provenance():
    root = Path(__file__).resolve().parents[2]
    # Hash the runtime sources as well as recording the commit.
    files = [p for package in ('dlm_kv_policy', 'evolution', 'experiments', 'language_models', 'utils')
        for p in sorted((root / 'ecad-dlm' / package).glob('*.py'))]
    if os.environ.get('ECAD_SOURCE_COMMIT'):
        # A frozen run's commit holds its exact working tree; there is no checkout to diff.
        status = 'frozen run source (scripts/slurm/freeze_source.sh)'
    else:
        status = subprocess.check_output(['git', '-C', str(root), 'status', '--short'], text=True).strip()
    return {'commit': source_commit(), 'git_status': status,
        'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
