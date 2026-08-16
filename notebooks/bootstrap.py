"""Colab/local notebook bootstrap: clone, import paths, and editable install.

This module must not import ``grasping_ai`` or ``notebook_helpers``. Those
modules are loaded only after :func:`bootstrap_notebook` has installed the
package.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    import google.colab as google_colab  # noqa: F401
except ImportError:
    google_colab = None  # pragma: no cover

try:
    from google.colab import drive as colab_drive
except ImportError:
    colab_drive = None  # pragma: no cover

DEFAULT_REPO_BRANCH = "dev"
DEFAULT_REPO_DIR = "vr-rlft-side-project"
DEFAULT_REPO_URL = "https://github.com/ntiensiit/vr-rlft-side-project.git"
DEFAULT_COLAB_ROOT = Path("/content") / DEFAULT_REPO_DIR
DEFAULT_COLAB_DRIVE_MOUNT = Path("/content/drive")
DEFAULT_DRIVE_STORAGE_DIR = DEFAULT_REPO_DIR
MGS_INPUT_DIR_ENV = "MGS_INPUT_DIR"
MGS_OUTPUT_DIR_ENV = "MGS_OUTPUT_DIR"


def is_colab_runtime() -> bool:
    """Return whether the current interpreter is running in Google Colab."""
    return google_colab is not None


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


def _update_git_checkout(project_root: Path, repo_branch: str) -> None:
    """Fetch, check out, and pull ``repo_branch`` in an existing clone."""
    subprocess.run(["git", "-C", str(project_root), "fetch", "origin", repo_branch], check=True)
    subprocess.run(["git", "-C", str(project_root), "checkout", repo_branch], check=True)
    subprocess.run(["git", "-C", str(project_root), "pull", "origin", repo_branch], check=True)


def locate_or_clone_project_root(
    *,
    repo_branch: str = DEFAULT_REPO_BRANCH,
    repo_url: str = DEFAULT_REPO_URL,
    repo_dir: str = DEFAULT_REPO_DIR,
) -> Path:
    """Locate a local checkout or clone the repository on Colab."""
    if is_colab_runtime():
        project_root = Path("/content") / repo_dir
        if not project_root.is_dir():
            subprocess.run(
                ["git", "clone", "--branch", repo_branch, "--depth", "1", repo_url, str(project_root)],
                check=True,
            )
        return project_root
    return resolve_project_root()


def mount_colab_drive(*, force_remount: bool = False) -> Path:
    """Mount Google Drive on Colab and return the ``MyDrive`` directory."""
    if not is_colab_runtime() or colab_drive is None:
        msg = "Google Drive mounting is only supported on Colab."
        raise RuntimeError(msg)
    colab_drive.mount(str(DEFAULT_COLAB_DRIVE_MOUNT), force_remount=force_remount)
    my_drive = DEFAULT_COLAB_DRIVE_MOUNT / "MyDrive"
    if not my_drive.is_dir():
        msg = f"Google Drive MyDrive folder not found: {my_drive}"
        raise FileNotFoundError(msg)
    return my_drive


def drive_process_storage_root(
    storage_dir: str = DEFAULT_DRIVE_STORAGE_DIR,
    *,
    mount: bool = True,
    force_remount: bool = False,
) -> Path | None:
    """Return a Drive-backed root for persistent notebook data and artifacts."""
    if not is_colab_runtime():
        return None
    my_drive = (
        mount_colab_drive(force_remount=force_remount)
        if mount
        else DEFAULT_COLAB_DRIVE_MOUNT / "MyDrive"
    )
    if not my_drive.is_dir():
        msg = f"Google Drive MyDrive folder not found: {my_drive}"
        raise FileNotFoundError(msg)
    storage_root = my_drive / storage_dir
    storage_root.mkdir(parents=True, exist_ok=True)
    return storage_root


def apply_drive_storage_env(
    storage_root: Path,
    *,
    input_subdir: str = "data",
    output_subdir: str = "artifacts",
) -> dict[str, str]:
    """Point Hydra path environment variables at Drive-backed directories."""
    input_dir = storage_root / input_subdir
    output_dir = storage_root / output_subdir
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_values = {
        MGS_INPUT_DIR_ENV: str(input_dir),
        MGS_OUTPUT_DIR_ENV: str(output_dir),
    }
    os.environ.update(env_values)
    return env_values


def configure_python_paths(project_root: Path) -> None:
    """Change into ``project_root`` and prepend ``src``/``scripts``/``notebooks``."""
    os.chdir(project_root)
    os.environ["PYTHONPYCACHEPREFIX"] = str(project_root / ".pycache")
    for relative in ("src", "scripts", "notebooks"):
        path = str(project_root / relative)
        if path not in sys.path:
            sys.path.insert(0, path)


def install_project_editable(project_root: Path) -> None:
    """Install the project in editable mode when ``pip`` is available."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(project_root)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


def setup_notebook_environment(
    *,
    mount_drive: bool = False,
    repo_branch: str = DEFAULT_REPO_BRANCH,
    repo_url: str = DEFAULT_REPO_URL,
    repo_dir: str = DEFAULT_REPO_DIR,
    root: Path | None = None,
) -> Path:
    """Clone or locate the project, install deps, and configure import paths."""
    if root is not None:
        project_root = root
        if is_colab_runtime() and project_root.is_dir():
            _update_git_checkout(project_root, repo_branch)
    elif is_colab_runtime():
        if mount_drive:
            mount_colab_drive(force_remount=False)
        project_root = locate_or_clone_project_root(
            repo_branch=repo_branch,
            repo_url=repo_url,
            repo_dir=repo_dir,
        )
        if (project_root / ".git").is_dir() and any(project_root.iterdir()):
            try:
                _update_git_checkout(project_root, repo_branch)
            except subprocess.CalledProcessError:
                pass
    else:
        project_root = resolve_project_root(root)

    configure_python_paths(project_root)
    install_project_editable(project_root)
    return project_root


def bootstrap_notebook() -> Path:
    """Prepare import paths and complete notebook environment setup.

    Returns:
        Absolute path to the active project root.
    """
    project_root = locate_or_clone_project_root()
    notebooks_dir = str(project_root / "notebooks")
    if notebooks_dir not in sys.path:
        sys.path.insert(0, notebooks_dir)
    return setup_notebook_environment(root=project_root)
