from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DEFAULT_REPO_BRANCH = "dev"
DEFAULT_REPO_DIR = "vr-rlft-side-project"
DEFAULT_REPO_URL = "https://github.com/ntiensiit/vr-rlft-side-project.git"


def is_colab_runtime() -> bool:
    """Return whether the current interpreter is running in Google Colab.

    Returns:
        ``True`` when ``google.colab`` can be imported, otherwise ``False``.
    """
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_project_root(start: Path | None = None) -> Path:
    """Locate the repository root from a starting directory.

    Args:
        start: Directory to inspect first. When omitted, ``Path.cwd()`` is used.

    Returns:
        The nearest directory containing ``configs/config.yaml``, or ``start``
        when no marker file is found.
    """
    root = start or Path.cwd()
    if (root / "configs" / "config.yaml").is_file():
        return root
    parent = root.parent
    if (parent / "configs" / "config.yaml").is_file():
        return parent
    return root


def locate_or_clone_project_root() -> Path:
    """Locate a local checkout or clone the repository on Colab.

    Returns:
        Project root path. On Colab, this is ``/content/vr-rlft-side-project``.
    """
    if is_colab_runtime():
        project_root = Path("/content") / DEFAULT_REPO_DIR
        if not project_root.is_dir():
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--branch",
                    DEFAULT_REPO_BRANCH,
                    "--depth",
                    "1",
                    DEFAULT_REPO_URL,
                    str(project_root),
                ],
                check=True,
            )
        return project_root
    return resolve_project_root()


def bootstrap_notebook() -> Path:
    """Prepare import paths and complete notebook environment setup.

    Adds ``notebooks/`` to ``sys.path`` and delegates to
    :func:`notebook_helpers.setup_notebook_environment`.

    Returns:
        Absolute path to the active project root.
    """
    project_root = locate_or_clone_project_root()
    notebooks_dir = str(project_root / "notebooks")
    if notebooks_dir not in sys.path:
        sys.path.insert(0, notebooks_dir)
    from notebook_helpers import setup_notebook_environment

    return setup_notebook_environment(root=project_root)
