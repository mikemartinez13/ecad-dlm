"""Writing result files and recording which source produced them."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=str, allow_nan=False))
    temporary.replace(path)


def source_provenance():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    # Hash dirty and untracked runtime sources as well as the commit.
    files = [p for package in ('dlm_kv_policy', 'evolution', 'experiments', 'language_models', 'utils')
        for p in sorted((root / 'ecad-dlm' / package).glob('*.py'))]
    return {'commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--short'),
        'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
