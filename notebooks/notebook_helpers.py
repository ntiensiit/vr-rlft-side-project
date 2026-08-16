from __future__ import annotations

from grasping_ai.config import FlattenedYAMLConfig
from grasping_ai.inference import run_single_object_grasp_inference
from grasping_ai.pipelines import (
    aggregate_evaluation_results,
    evaluate_generated_grasps,
    load_generated_grasps,
    run_simulation_sweep,
    write_evaluation_report,
    write_jsonl_records,
)

import os
import platform
import random
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

def _yaml_config(cfg: Mapping[str, object]) -> FlattenedYAMLConfig:
    return FlattenedYAMLConfig(cfg)

try:
    import google.colab  # noqa: F401
except ImportError:
    google.colab = None  # pragma: no cover

try:
    from google.colab import drive
except ImportError:
    drive = None  # pragma: no cover

try:
    import mlflow
except ImportError:
    mlflow = None  # pragma: no cover

DEFAULT_REPO_BRANCH = "dev"
DEFAULT_REPO_DIR = "vr-rlft-side-project"
DEFAULT_REPO_URL = "https://github.com/ntiensiit/vr-rlft-side-project.git"
DEFAULT_COLAB_DRIVE_MOUNT = Path("/content/drive")
DEFAULT_DRIVE_STORAGE_DIR = DEFAULT_REPO_DIR
MGS_INPUT_DIR_ENV = "MGS_INPUT_DIR"
MGS_OUTPUT_DIR_ENV = "MGS_OUTPUT_DIR"

def is_colab_runtime() -> bool:
    """Return whether the current interpreter is running in Google Colab.

    Returns:
        ``True`` when ``google.colab`` can be imported, otherwise ``False``.
    """
    return google.colab is not None

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

def mount_colab_drive(*, force_remount: bool = False) -> Path:
    """Mount Google Drive on Colab and return the ``MyDrive`` directory.

    Args:
        force_remount: Forwarded to ``google.colab.drive.mount``.

    Returns:
        Path to ``/content/drive/MyDrive``.

    Raises:
        RuntimeError: When called outside Google Colab.
        FileNotFoundError: When the mount succeeds but ``MyDrive`` is missing.
    """
    if not is_colab_runtime():
        raise RuntimeError("Google Drive mounting is only supported on Colab.")
    drive.mount(str(DEFAULT_COLAB_DRIVE_MOUNT), force_remount=force_remount)
    my_drive = DEFAULT_COLAB_DRIVE_MOUNT / "MyDrive"
    if not my_drive.is_dir():
        raise FileNotFoundError(f"Google Drive MyDrive folder not found: {my_drive}")
    return my_drive

def drive_process_storage_root(
    storage_dir: str = DEFAULT_DRIVE_STORAGE_DIR,
    *,
    mount: bool = True,
    force_remount: bool = False,
) -> Path | None:
    """Return a Drive-backed root for persistent notebook data and artifacts.

    Creates ``MyDrive/<storage_dir>`` on Colab so datasets, checkpoints, and
    reports survive runtime disconnects.

    Args:
        storage_dir: Folder name under ``MyDrive`` used for project storage.
        mount: When ``True``, mount Google Drive before resolving the path.
        force_remount: Forwarded to :func:`mount_colab_drive`.

    Returns:
        Drive-backed storage root on Colab, otherwise ``None``.
    """
    if not is_colab_runtime():
        return None
    my_drive = (
        mount_colab_drive(force_remount=force_remount)
        if mount
        else DEFAULT_COLAB_DRIVE_MOUNT / "MyDrive"
    )
    if not my_drive.is_dir():
        raise FileNotFoundError(f"Google Drive MyDrive folder not found: {my_drive}")
    storage_root = my_drive / storage_dir
    storage_root.mkdir(parents=True, exist_ok=True)
    return storage_root

def apply_drive_storage_env(
    storage_root: Path,
    *,
    input_subdir: str = "data",
    output_subdir: str = "artifacts",
) -> dict[str, str]:
    """Point Hydra path environment variables at Drive-backed directories.

    Sets ``MGS_INPUT_DIR`` and ``MGS_OUTPUT_DIR`` so ``configs/data/default.yaml``
    resolves ``paths.input_dir`` and ``paths.output_dir`` under ``storage_root``.

    Args:
        storage_root: Drive-backed project storage root.
        input_subdir: Relative folder for datasets and observations.
        output_subdir: Relative folder for checkpoints, exports, and reports.

    Returns:
        Mapping of the environment variables that were applied.
    """
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

def setup_notebook_drive_storage(
    cfg: Mapping[str, object],
    *,
    force_remount: bool = False,
) -> Path | None:
    """Mount Drive and configure persisted paths when enabled in config.

    Args:
        cfg: Loaded configuration mapping containing ``notebook.mount_drive``.
        force_remount: Forwarded to :func:`mount_colab_drive`.

    Returns:
        Drive-backed storage root when mounting is enabled on Colab, otherwise
        ``None``.
    """
    if not notebook_mount_drive(cfg):
        return None
    storage_root = drive_process_storage_root(
        notebook_drive_storage_dir(cfg),
        force_remount=force_remount,
    )
    if storage_root is not None:
        apply_drive_storage_env(storage_root)
    return storage_root

def setup_notebook_environment(
    *,
    mount_drive: bool = False,
    repo_branch: str = DEFAULT_REPO_BRANCH,
    repo_url: str = DEFAULT_REPO_URL,
    repo_dir: str = DEFAULT_REPO_DIR,
    root: Path | None = None,
) -> Path:
    """Clone or locate the project, install deps, and configure import paths.

    On Colab, clones or updates the repository under ``/content/<repo_dir>``.
    Locally, resolves the checkout, changes into it, prepends ``src``,
    ``scripts``, and ``notebooks`` to ``sys.path``, and installs the package
    in editable mode.

    Args:
        mount_drive: When ``True`` on Colab, mount Google Drive before setup.
        repo_branch: Git branch to clone or update on Colab.
        repo_url: Remote repository URL used for Colab clones.
        repo_dir: Directory name under ``/content`` for Colab checkouts.
        root: Optional pre-resolved project root. When provided, Colab clone is
            skipped and the existing checkout is updated instead.

    Returns:
        Absolute path to the active project root.
    """
    if root is not None:
        project_root = root
        if is_colab_runtime() and project_root.is_dir():
            subprocess.run(["git", "-C", str(project_root), "fetch", "origin", repo_branch], check=True)
            subprocess.run(["git", "-C", str(project_root), "checkout", repo_branch], check=True)
            subprocess.run(["git", "-C", str(project_root), "pull", "origin", repo_branch], check=True)
    elif is_colab_runtime():
        if mount_drive:
            mount_colab_drive(force_remount=False)
        project_root = Path("/content") / repo_dir
        if not project_root.is_dir():
            subprocess.run(
                ["git", "clone", "--branch", repo_branch, "--depth", "1", repo_url, str(project_root)],
                check=True,
            )
        else:
            subprocess.run(["git", "-C", str(project_root), "fetch", "origin", repo_branch], check=True)
            subprocess.run(["git", "-C", str(project_root), "checkout", repo_branch], check=True)
            subprocess.run(["git", "-C", str(project_root), "pull", "origin", repo_branch], check=True)
    else:
        project_root = resolve_project_root(root)

    os.chdir(project_root)
    os.environ["PYTHONPYCACHEPREFIX"] = str(project_root / ".pycache")
    for relative in ("src", "scripts", "notebooks"):
        path = str(project_root / relative)
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return project_root

def config_dir_relative(config_dir: Path, root: Path) -> str:
    """Format a config directory path for ``--config-dir`` CLI usage.

    Args:
        config_dir: Hydra config directory, typically ``<root>/configs``.
        root: Active project root used to compute a relative path.

    Returns:
        Posix-style path to ``config_dir`` relative to ``root``.
    """
    return config_dir.resolve().relative_to(root.resolve()).as_posix()

def config_script_cmd(
    config_dir_arg: str,
    config_name: str,
    script: str,
    *extra: str,
) -> list[str]:
    """Build a subprocess argv list for a Hydra-aware project script.

    Args:
        config_dir_arg: Value passed to ``--config-dir``.
        config_name: Hydra entrypoint basename, e.g. ``training/diffusion``.
        script: Script path relative to the project root.
        *extra: Additional CLI arguments appended after the config flags.

    Returns:
        Argument vector suitable for ``subprocess.run``.
    """
    return [
        sys.executable,
        script,
        "--config-dir",
        config_dir_arg,
        "--config-name",
        config_name,
        *extra,
    ]

def subprocess_env(root: Path) -> dict[str, str]:
    """Build an environment mapping for notebook subprocess calls.

    Args:
        root: Project root used to set ``PYTHONPYCACHEPREFIX``.

    Returns:
        A copy of ``os.environ`` with notebook-friendly defaults applied.
    """
    return {**os.environ, "PYTHONPYCACHEPREFIX": str(root / ".pycache")}

def load_notebook_config(
    config_dir: Path,
    config_name: str,
    *,
    overrides: list[str] | None = None,
    force_remount: bool = False,
) -> DictConfig:
    """Load a named Hydra entrypoint for notebook workflows.

    When ``notebook.mount_drive`` is enabled, mounts Google Drive, configures
    ``MGS_INPUT_DIR`` / ``MGS_OUTPUT_DIR``, and reloads the config so path
    interpolation resolves to Drive-backed storage.

    Args:
        config_dir: Directory containing Hydra YAML entrypoints.
        config_name: Entrypoint basename without ``.yaml``.
        overrides: Optional list of Hydra override strings.
        force_remount: Forwarded to :func:`setup_notebook_drive_storage`.

    Returns:
        Composed configuration for the requested entrypoint.
    """
    yaml_config = FlattenedYAMLConfig.from_hydra(
        config_dir,
        config_name,
        overrides=overrides,
        library_defaults=False,
    )
    cfg = yaml_config.cfg
    if setup_notebook_drive_storage(cfg, force_remount=force_remount) is not None:
        yaml_config = FlattenedYAMLConfig.from_hydra(
            config_dir,
            config_name,
            overrides=overrides,
            library_defaults=False,
        )
        cfg = yaml_config.cfg
    return cfg

def configure_seeds_and_device(cfg: Mapping[str, object]) -> tuple[int, str]:
    """Seed Python, NumPy, and PyTorch and resolve the runtime device.

    Args:
        cfg: Loaded configuration containing ``seed`` and ``device`` keys.

    Returns:
        A ``(seed, device)`` tuple. ``device`` resolves ``auto`` to ``cuda``
        when CUDA is available, otherwise ``cpu``.
    """
    yaml_config = _yaml_config(cfg)
    seed = int(yaml_config.get_path("seed"))
    device_setting = str(yaml_config.get_path("device"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if device_setting == "auto" and torch.cuda.is_available() else device_setting
    if device == "auto":
        device = "cpu"
    return seed, device

def print_runtime_banner(
    *,
    experiment: str,
    root: Path,
    device: str,
    seed: int,
    config_names: Mapping[str, str] | str,
    **details: object,
) -> None:
    """Print a standard runtime summary for notebook configuration cells.

    Args:
        experiment: Notebook experiment name from ``notebook.experiment``.
        root: Active project root.
        device: Resolved runtime device string.
        seed: Configured random seed.
        config_names: Single config name or mapping of label to config name.
        **details: Additional key/value pairs printed after the standard banner.
    """
    print("experiment:", experiment)
    if isinstance(config_names, str):
        print("config_name:", config_names)
    else:
        for label, name in config_names.items():
            print(f"{label}_config:", name)
    print("project_root:", root)
    print("python:", sys.version.split()[0])
    print("platform:", platform.platform())
    print("torch:", torch.__version__)
    print("device:", device)
    print("cuda_available:", torch.cuda.is_available())
    print("seed:", seed)
    for key, value in details.items():
        print(f"{key}:", value)

def dataset_paths(cfg: Mapping[str, object]) -> dict[str, Path]:
    """Extract common dataset path entries from a composed config.

    Args:
        cfg: Loaded configuration mapping.

    Returns:
        Mapping of logical dataset path names to ``pathlib.Path`` objects.
    """
    return {
        "ycb_root": _yaml_config(cfg).value("paths", "ycb_root", value_type=Path),
        "dataset_root": _yaml_config(cfg).value("paths", "dataset_root", value_type=Path),
        "mjcf_root": _yaml_config(cfg).value("paths", "ycb_mjcf", value_type=Path),
        "observations_dir": _yaml_config(cfg).value("paths", "observations", value_type=Path),
        "output_index": _yaml_config(cfg).value("paths", "output_index", value_type=Path),
    }

def ensure_dataset_dirs(paths: Mapping[str, Path]) -> None:
    """Create dataset directories expected by notebook data-prep scripts.

    Args:
        paths: Mapping returned by :func:`dataset_paths`.
    """
    for key in ("mjcf_root", "observations_dir", "dataset_root"):
        paths[key].mkdir(parents=True, exist_ok=True)

def maybe_download_ycb(
    root: Path,
    *,
    ycb_root: Path,
    object_ids: list[str],
    download: bool,
    env: Mapping[str, str] | None = None,
) -> None:
    """Download YCB assets when requested and any required object is missing.

    Args:
        root: Project root used as the subprocess working directory.
        ycb_root: Root directory containing per-object YCB folders.
        object_ids: Object identifiers that must exist under ``ycb_root``.
        download: When ``False``, this function is a no-op.
        env: Optional environment mapping passed to ``subprocess.run``.
    """
    if download and not all((ycb_root / object_id).is_dir() for object_id in object_ids):
        subprocess.run(
            [sys.executable, "scripts/download_ycb_dataset.py"],
            cwd=root,
            env=env,
            check=True,
        )

def run_dataset_preparation(
    root: Path,
    *,
    cfg: Mapping[str, object],
    config_dir_arg: str,
    config_name: str,
    object_ids: list[str],
    download_ycb: bool,
) -> dict[str, Path]:
    """Run the standard notebook dataset preparation script chain.

    Optionally downloads YCB assets, then invokes ``prepare_ycb_mjcf``,
    ``prepare_data``, and ``prepare_observations`` with the supplied Hydra
    entrypoint.

    Args:
        root: Project root used as the subprocess working directory.
        cfg: Loaded configuration mapping.
        config_dir_arg: Value passed to ``--config-dir``.
        config_name: Hydra entrypoint basename without ``.yaml``.
        object_ids: Object identifiers forwarded to ``prepare_data.py``.
        download_ycb: Whether missing YCB assets should be downloaded first.

    Returns:
        Dataset path mapping from :func:`dataset_paths`.
    """
    paths = dataset_paths(cfg)
    ensure_dataset_dirs(paths)
    env = subprocess_env(root)
    maybe_download_ycb(
        root,
        ycb_root=paths["ycb_root"],
        object_ids=object_ids,
        download=download_ycb,
        env=env,
    )
    subprocess.run(
        config_script_cmd(config_dir_arg, config_name, "scripts/prepare_ycb_mjcf.py"),
        cwd=root,
        env=env,
        check=True,
    )
    subprocess.run(
        config_script_cmd(
            config_dir_arg,
            config_name,
            "scripts/prepare_data.py",
            "--mode",
            "synthetic",
            "--required-objects",
            *object_ids,
        ),
        cwd=root,
        env=env,
        check=True,
    )
    subprocess.run(
        config_script_cmd(config_dir_arg, config_name, "scripts/prepare_observations.py"),
        cwd=root,
        env=env,
        check=True,
    )
    return paths

def supervised_hyperparameters(cfg: Mapping[str, object]) -> tuple[float, int, int]:
    """Read supervised training hyperparameters from the composed config.

    Args:
        cfg: Loaded configuration mapping.

    Returns:
        ``(learning_rate, num_epochs, batch_size)``.
    """
    learning_rate = _yaml_config(cfg).value("supervised", "learning_rate", value_type=float, default=1e-3)
    num_epochs = _yaml_config(cfg).value("supervised", "num_epochs", value_type=int, default=3)
    batch_size = _yaml_config(cfg).value("supervised", "batch_size", value_type=int, default=2)
    return learning_rate, num_epochs, batch_size

def build_supervised_train_kwargs(
    cfg: Mapping[str, object],
    *,
    model_key: str,
    dataset_root: Path,
    checkpoint_path: Path,
    device: str,
    seed: int,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
) -> dict[str, object]:
    """Build keyword arguments for supervised training pipelines.

    Args:
        cfg: Loaded configuration mapping.
        model_key: Top-level model namespace, e.g. ``"diffusion"`` or ``"flow"``.
        dataset_root: Prepared dataset directory.
        checkpoint_path: Destination checkpoint file.
        device: Resolved runtime device string.
        seed: Configured random seed.
        learning_rate: Supervised learning rate.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.

    Returns:
        Keyword-argument mapping for ``run_diffusion_training_pipeline`` or
        ``run_flow_training_pipeline``.
    """
    return {
        "dataset_root": dataset_root,
        "checkpoint_path": checkpoint_path,
        "feature_dim": int(_yaml_config(cfg).get_path("architecture", "feature_dim")),
        "hidden_dim": int(_yaml_config(cfg).get_path("architecture", "hidden_dim")),
        "num_layers": int(_yaml_config(cfg).get_path("architecture", "num_layers")),
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "device": device,
        "seed": seed,
        "experiment_log_dir": _yaml_config(cfg).value(model_key, "tensorboard", value_type=Path),
        "pretrained_encoder_path": None,
        "resume_checkpoint_path": None,
        "augment": notebook_augment(cfg),
        "min_grasp_score": float(_yaml_config(cfg).get_path("supervised", "min_grasp_score", default=0.0)),
        "score_repeat_factor": int(_yaml_config(cfg).get_path("supervised", "score_repeat_factor", default=0)),
        "score_repeat_power": float(_yaml_config(cfg).get_path("supervised", "score_repeat_power", default=1.0)),
    }

def run_with_optional_mlflow(
    use_mlflow: bool,
    experiment: str,
    log_params: Mapping[str, object],
    pipeline: Callable[..., object],
    **pipeline_kwargs: object,
) -> None:
    """Run a pipeline directly or inside an MLflow run.

    Args:
        use_mlflow: When ``True``, open an MLflow run and log parameters first.
        experiment: MLflow run name.
        log_params: Parameter mapping written with ``mlflow.log_param``.
        pipeline: Callable training or evaluation entry point.
        **pipeline_kwargs: Keyword arguments forwarded to ``pipeline``.
    """
    if use_mlflow:
        with mlflow.start_run(run_name=experiment):
            for key, value in log_params.items():
                mlflow.log_param(key, value)
            pipeline(**pipeline_kwargs)
        return
    pipeline(**pipeline_kwargs)

def print_checkpoint_summary(checkpoint_path: Path, tensorboard_dir: Path) -> None:
    """Print checkpoint existence, size, and TensorBoard log directory.

    Args:
        checkpoint_path: Model or policy checkpoint file.
        tensorboard_dir: TensorBoard output directory for the run.
    """
    if checkpoint_path.is_file():
        stat = checkpoint_path.stat()
        print(
            {
                "path": str(checkpoint_path),
                "exists": True,
                "size_mb": round(stat.st_size / (1024 * 1024), 3),
            }
        )
    else:
        print({"path": str(checkpoint_path), "exists": False})
    print("tensorboard:", tensorboard_dir)

def simulation_outcome_records(object_id: str, outcomes: list[Mapping[str, object]]) -> list[dict[str, object]]:
    """Convert simulation outcome mappings into JSONL-ready report records.

    Args:
        object_id: Evaluated object identifier.
        outcomes: Raw simulation outcome mappings from ``run_simulation_sweep``.

    Returns:
        List of JSON-serializable report records.
    """
    records: list[dict[str, object]] = []
    for grasp_index, outcome in enumerate(outcomes):
        record: dict[str, object] = {
            "record_type": "grasp_outcome",
            "object_id": object_id,
            "grasp_index": grasp_index,
        }
        for key, value in outcome.items():
            record[key] = value.tolist() if hasattr(value, "tolist") else value
        records.append(record)
    return records

def lift_success_rate(outcomes: list[Mapping[str, object]]) -> float | None:
    """Compute the fraction of simulation outcomes with ``lift_success=True``.

    Args:
        outcomes: Simulation outcome mappings from ``run_simulation_sweep``.

    Returns:
        Success rate in ``[0, 1]``, or ``None`` when ``outcomes`` is empty.
    """
    if not outcomes:
        return None
    return sum(1 for row in outcomes if bool(row.get("lift_success"))) / len(outcomes)

def run_generative_evaluation(
    *,
    method: str,
    cfg: Mapping[str, object],
    checkpoint_path: Path,
    grasp_path: Path,
    object_id: str,
    object_point_cloud: np.ndarray,
    gripper_point_cloud: np.ndarray,
    observation_path: Path,
    reports_dir: Path,
    device: str,
    seed: int,
    friction_coefficient: float,
    lift_height_threshold: float,
    contact_clearance: float,
    wrench_regularization: float,
    filter_collisions: bool,
    num_simulation_steps: int,
    gripper_close_command: np.ndarray,
) -> tuple[dict[str, object], list[Mapping[str, object]], Path, Path]:
    """Run inference, analytical evaluation, and simulation for one generative method.

    Args:
        method: Generative method namespace, e.g. ``"diffusion"`` or ``"flow"``.
        cfg: Loaded configuration mapping for the method.
        checkpoint_path: Model checkpoint used for grasp inference.
        grasp_path: Output path for generated grasp poses.
        object_id: Evaluated object identifier.
        object_point_cloud: Object observation point cloud.
        gripper_point_cloud: Gripper point cloud used for evaluation.
        observation_path: Observation ``.npy`` file passed to inference.
        reports_dir: Directory for simulation JSONL output.
        device: Resolved runtime device string.
        seed: Configured random seed.
        friction_coefficient: Analytical evaluation friction coefficient.
        lift_height_threshold: Lift-success height threshold.
        contact_clearance: Collision clearance threshold.
        wrench_regularization: Force-closure wrench regularization weight.
        filter_collisions: Whether colliding grasps should be filtered out.
        num_simulation_steps: MuJoCo rollout length for simulation evaluation.
        gripper_close_command: Gripper close command vector for simulation.

    Returns:
        ``(aggregated_results, simulation_outcomes, analytical_report_path,
        simulation_report_path)``.
    """
    run_single_object_grasp_inference(
        checkpoint_path=checkpoint_path,
        output_path=grasp_path,
        method=method,
        feature_dim=int(_yaml_config(cfg).get_path("architecture", "feature_dim")),
        num_steps=int(_yaml_config(cfg).get_path(method, "inference_steps")),
        num_grasps=int(_yaml_config(cfg).get_path("architecture", "num_grasps")),
        device=device,
        seed=seed,
        observation_path=observation_path,
    )

    grasps = load_generated_grasps(grasp_path, object_key=object_id)
    results = evaluate_generated_grasps(
        grasp_poses=grasps,
        object_point_cloud=object_point_cloud,
        gripper_point_cloud=gripper_point_cloud,
        contact_set_provider=None,
        friction_coefficient=friction_coefficient,
        lift_height_threshold=lift_height_threshold,
        clearance=contact_clearance,
        wrench_regularization=wrench_regularization,
        contact_path=None,
        filter_collisions=filter_collisions,
    )
    report_path = _yaml_config(cfg).value("evaluation", "analytical_report", value_type=Path)
    aggregated = aggregate_evaluation_results({object_id: results})
    write_evaluation_report(report_path, aggregated, None, per_object_results=None)

    sim_path = reports_dir / f"{method}_simulation_outcomes_{object_id}.jsonl"
    sim_outcomes = run_simulation_sweep(
        grasp_poses=grasps,
        object_id=object_id,
        ycb_root=_yaml_config(cfg).value("paths", "ycb_mjcf", value_type=Path),
        robot_xml_path=_yaml_config(cfg).value("robot", "description", value_type=Path),
        table_xml_path=None,
        num_simulation_steps=num_simulation_steps,
        gripper_close_command=gripper_close_command,
    )
    write_jsonl_records(sim_path, simulation_outcome_records(object_id, sim_outcomes))
    return aggregated, sim_outcomes, report_path, sim_path

def object_ids_from_config(cfg: Mapping[str, object]) -> list[str]:
    """Read configured object identifiers.

    Args:
        cfg: Loaded configuration mapping.

    Returns:
        Values from ``objects.ids``.
    """
    return _yaml_config(cfg).value("objects", "ids", value_type=list[str])

def notebook_experiment(cfg: Mapping[str, object]) -> str:
    """Read the notebook experiment name.

    Args:
        cfg: Loaded configuration mapping.

    Returns:
        Value of ``notebook.experiment``.
    """
    return str(_yaml_config(cfg).get_path("notebook", "experiment"))

def notebook_download_ycb(cfg: Mapping[str, object]) -> bool:
    """Read whether notebooks should download missing YCB assets.

    Args:
        cfg: Loaded configuration mapping.

    Returns:
        Value of ``notebook.download_ycb``.
    """
    return bool(_yaml_config(cfg).get_path("notebook", "download_ycb", default=True))

def notebook_augment(cfg: Mapping[str, object]) -> bool:
    """Read whether supervised notebook training should augment data.

    Args:
        cfg: Loaded configuration mapping.

    Returns:
        Value of ``notebook.augment``.
    """
    return bool(_yaml_config(cfg).get_path("notebook", "augment", default=False))

def notebook_mount_drive(cfg: Mapping[str, object]) -> bool:
    """Read whether notebooks should mount Google Drive for persisted storage.

    Args:
        cfg: Loaded configuration mapping.

    Returns:
        Value of ``notebook.mount_drive``.
    """
    return bool(_yaml_config(cfg).get_path("notebook", "mount_drive", default=False))

def notebook_drive_storage_dir(cfg: Mapping[str, object]) -> str:
    """Read the Drive folder name used for persisted notebook storage.

    Args:
        cfg: Loaded configuration mapping.

    Returns:
        Value of ``notebook.drive_storage_dir``.
    """
    return str(_yaml_config(cfg).get_path("notebook", "drive_storage_dir", default=DEFAULT_DRIVE_STORAGE_DIR))

def notebook_object_index(cfg: Mapping[str, object]) -> int:
    """Read the configured notebook object index.

    Args:
        cfg: Loaded configuration mapping.

    Returns:
        Value of ``notebook.object_index``.
    """
    return int(_yaml_config(cfg).get_path("notebook", "object_index", default=0))

def selected_object_id(cfg: Mapping[str, object]) -> str:
    """Resolve the object identifier selected by ``notebook.object_index``.

    Args:
        cfg: Loaded configuration mapping.

    Returns:
        Object ID at ``objects.ids[notebook.object_index]``.
    """
    object_ids = object_ids_from_config(cfg)
    return object_ids[notebook_object_index(cfg)]

def supervised_mlflow_log_params(
    cfg: Mapping[str, object],
    *,
    device: str,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
) -> dict[str, object]:
    """Build MLflow parameter mappings for supervised notebook training.

    Args:
        cfg: Loaded configuration mapping.
        device: Resolved runtime device string.
        learning_rate: Supervised learning rate.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.

    Returns:
        Parameter mapping suitable for :func:`run_with_optional_mlflow`.
    """
    return {
        "download_ycb": notebook_download_ycb(cfg),
        "augment": notebook_augment(cfg),
        "device": device,
        "experiment": notebook_experiment(cfg),
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
    }
