# ecad-dlm

Diffusion language model code, separated from the image-model code in `ecad/`.
The folder name has a hyphen, so it is not itself importable: put it on the path
and import the packages inside it.

```bash
export PYTHONPATH="$PWD/ecad-dlm:$PWD"   # from the repository root
```

| Folder | Contents |
| --- | --- |
| `language_models/` | Model loading, code tasks, EvalPlus execution, residual-cache runtime, DLM schedules |
| `dlm_kv_policy/` | DiffuCoder K/V caching: policies, backends (batch one and batched), sampler, and the divergence proxy (`teacher_forcing.py`, `validate_proxy.py`) |
| `evolution/` | NSGA-II searches over K/V policies and residual-cache schedules; see `experiments/CONSTRAINED_SEARCH.md` |
| `experiments/` | Studies and their documents, `configs/` and `scripts/`, including the proxy recheck, cost study and EAR validity study |
| `utils/` | Helpers outside the caching method: FLOP counting (`compute_cost.py`, the search's cost), row packing (`row_index.py`), result files and source hashes (`results_io.py`), rank statistics (`ranking.py`) |
| `policies/` | Example policies for `experiments/evaluate_policies.py`, which tests saved policies against dense |
| `tests/` | CPU tests. Run with `cd ecad-dlm/tests && python -m unittest discover -s . -p 'test_*.py'` |

Core packages never import from `experiments/`. Files ending in `_cs` hold CS
cluster paths; the same files without the suffix hold the Rivanna paths.

## Slurm runs and frozen sources

Every `submit_*` script freezes the source before it submits, so later edits cannot
reach a queued or running job. `scripts/slurm/freeze_source.sh NAME` writes the working
tree's `ecad-dlm/` and `scripts/` (committed or not, untracked files included, ignored
files left out) as a git commit under `refs/runs/NAME/<UTC time>`, through a temporary
index: the branch, the index and the files are untouched, and nothing shows in `git log`.
Each job is `scripts/slurm/run_frozen.sh REPO COMMIT COMMAND...`, which unpacks the commit
into node-local temporary space, runs the command there and removes the copy at the end.
Paths to configs and scripts inside the source are therefore relative to the repository
root. Results record the commit (`utils/results_io.source_provenance`), and each submit
script writes its commit and job IDs to `logs/runs/NAME.txt` under the artifacts folder.

```bash
git for-each-ref refs/runs                              # every frozen run
git show COMMIT:ecad-dlm/experiments/configs/x.yaml     # one file of a run
git archive COMMIT | tar -x -C /tmp/run                 # a run's whole source
```

A run name frozen before is refused unless the script resumes on purpose (`--again`,
`--reuse`, `--existing`). Refs under `refs/runs` stay in this clone: `git push` sends
branches only. Runs submitted before 2026-11-25 used copies under `run_snapshots/<run>`;
those copies are kept as `refs/runs/<run>/<time>.legacy`, byte for byte.
`ecad-dlm/` still imports the shared `CacheSchedule` base class and schedule types from `ecad/`.
