"""
Storage locations shared by every notebook in the pipeline.

All notebooks read and write three folders:

    <OUTPUT_ROOT>/data/      raw and processed datasets, clustering assignments
    <OUTPUT_ROOT>/models/    trained classifiers, experts, gating and production model
    <OUTPUT_ROOT>/results/   figures, tables and benchmark reports

`OUTPUT_ROOT` is resolved in this order:

1. The value passed to `setup_paths()` (the `OUTPUT_ROOT` parameter at the top
   of each notebook).
2. The environment variable `MOE_OUTPUT_ROOT`.
3. The repository root (the folder that contains `src/`).

Relative paths are interpreted relative to the repository root, so
`OUTPUT_ROOT = "outputs/run_01"` works the same in every environment. Paths
under `/content/drive` trigger a Google Drive mount when running in Colab.

Every notebook in a run must use the same `OUTPUT_ROOT`, because each notebook
reads the artifacts written by the previous ones.
"""

import os
import sys
from pathlib import Path

ENV_VAR = "MOE_OUTPUT_ROOT"

# Sub-folders created on first use (relative to OUTPUT_ROOT)
SUBFOLDERS = (
    "data/raw",
    "data/processed",
    "data/clustering",
    "models/classifiers",
    "models/experts",
    "models/gating",
    "models/hyperparameters",
    "models/production/experts",
    "results/benchmark",
)

_DRIVE_MOUNT_POINT = "/content/drive"


def _default_repo_root():
    """Repository root, assuming this file lives in <repo>/src/moe_forecaster/."""
    return Path(__file__).resolve().parents[2]


def _mount_drive_if_needed(path):
    """Mount Google Drive when running in Colab and `path` points inside it."""
    if "google.colab" not in sys.modules:
        return
    if not str(path).startswith(_DRIVE_MOUNT_POINT):
        return
    if Path(_DRIVE_MOUNT_POINT, "MyDrive").exists():
        return
    from google.colab import drive  # only available inside Colab

    drive.mount(_DRIVE_MOUNT_POINT)


def resolve_output_root(output_root=None, repo_root=None):
    """Return the absolute folder that will hold data/, models/ and results/."""
    repo_root = Path(repo_root) if repo_root else _default_repo_root()
    value = output_root or os.environ.get(ENV_VAR) or repo_root
    root = Path(value).expanduser()
    if not root.is_absolute():
        root = repo_root / root
    return root.resolve()


def setup_paths(output_root=None, repo_root=None, verbose=True):
    """
    Resolve the storage folders, create them if needed and return them.

    Parameters
    ----------
    output_root : str or Path, optional
        Where to store data/, models/ and results/. See the module docstring
        for how the default is chosen.
    repo_root : str or Path, optional
        Repository root, used to resolve relative paths. Defaults to the
        folder that contains `src/`.
    verbose : bool
        Print the resolved locations.

    Returns
    -------
    (DATA_DIR, MODELS_DIR, RESULTS_DIR) : tuple of str
        Absolute paths, as strings so they can be used directly in f-strings.
    """
    root = resolve_output_root(output_root, repo_root)
    _mount_drive_if_needed(root)

    for sub in SUBFOLDERS:
        (root / sub).mkdir(parents=True, exist_ok=True)

    data_dir = str(root / "data")
    models_dir = str(root / "models")
    results_dir = str(root / "results")

    if verbose:
        print(f"OUTPUT_ROOT = {root}")
        print(f"  DATA_DIR    = {data_dir}")
        print(f"  MODELS_DIR  = {models_dir}")
        print(f"  RESULTS_DIR = {results_dir}")

    return data_dir, models_dir, results_dir
