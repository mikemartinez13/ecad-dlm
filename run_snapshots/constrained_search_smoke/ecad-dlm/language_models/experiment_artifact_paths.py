"""Store heavy experiment artifacts on HPC scratch, with historical path support."""

import getpass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = Path("/scratch") / getpass.getuser() / "ai4mat/ecad"
DEFAULT_RESULTS_ROOT = DEFAULT_ARTIFACT_ROOT / "results"
LEGACY_RESULTS_ROOT = PROJECT_ROOT / "results"


def resolve_experiment_result_path(path: str | Path) -> Path:
    """Route repository-local result paths in saved configs to HPC scratch."""

    source = Path(path).expanduser()
    legacy_roots = (
        LEGACY_RESULTS_ROOT,
        PROJECT_ROOT / "scratch/results",
        Path("/home/gqs7zr/ai4mat/ecad/results"),
        Path("/home/gqs7zr/ai4mat/ecad/scratch/results"),
        Path("results"),
        Path("scratch/results"),
    )
    for root in legacy_roots:
        try:
            relative = source.relative_to(root)
        except ValueError:
            continue
        return DEFAULT_RESULTS_ROOT / relative
    return source


def legacy_checkpoint_result_path(path: str | Path) -> str:
    """Preserve existing configuration signatures across artifact relocation."""

    source = Path(path)
    for root in (DEFAULT_RESULTS_ROOT, DEFAULT_RESULTS_ROOT.resolve()):
        try:
            relative = source.relative_to(root)
        except ValueError:
            continue
        return str(LEGACY_RESULTS_ROOT / relative)
    return str(path)
