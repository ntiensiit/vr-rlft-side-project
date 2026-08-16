"""Shared helpers for percent-format research notebooks."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from bootstrap import (
    DEFAULT_DRIVE_STORAGE_DIR,
    apply_drive_storage_env,
    drive_process_storage_root,
)

from grasping_ai.config import FlattenedYAMLConfig
from grasping_ai.inference.grasp_inference_runtime import run_single_object_grasp_inference
from grasping_ai.pipelines import (
    aggregate_evaluation_results,
    evaluate_generated_grasps,
    load_generated_grasps,
    run_simulation_sweep,
    write_evaluation_report,
    write_jsonl_records,
)

try:
    import mlflow
except ImportError:
    mlflow = None  # pragma: no cover


def _yaml_config(cfg: Mapping[str, object] | FlattenedYAMLConfig) -> FlattenedYAMLConfig:
    if isinstance(cfg, FlattenedYAMLConfig):
        return cfg
    return FlattenedYAMLConfig(cfg)


@dataclass(frozen=True)
class NotebookContext:
    """Resolved notebook session: project root, composed config, and runtime."""

    root: Path
    config_name: str
    cfg: DictConfig
    yaml_config: FlattenedYAMLConfig
    seed: int
    device: str
    experiment: str
    object_ids: list[str]


def subprocess_env(root: Path) -> dict[str, str]:
    """Build an environment mapping for notebook subprocess calls."""
    return {**os.environ, "PYTHONPYCACHEPREFIX": str(root / ".pycache")}


def setup_notebook_drive_storage(
    cfg: Mapping[str, object],
    *,
    force_remount: bool = False,
) -> Path | None:
    """Mount Drive and configure persisted paths when enabled in config."""
    if not notebook_mount_drive(cfg):
        return None
    storage_root = drive_process_storage_root(
        notebook_drive_storage_dir(cfg),
        force_remount=force_remount,
    )
    if storage_root is not None:
        apply_drive_storage_env(storage_root)
    return storage_root


def load_notebook_config(
    config_dir: Path,
    config_name: str,
    *,
    overrides: list[str] | None = None,
    force_remount: bool = False,
) -> DictConfig:
    """Load a named Hydra entrypoint for notebook workflows."""
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
    """Seed Python, NumPy, and PyTorch and resolve the runtime device."""
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
    """Print a standard runtime summary for notebook configuration cells."""
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


def object_ids_from_config(cfg: Mapping[str, object]) -> list[str]:
    """Read configured object identifiers."""
    return _yaml_config(cfg).value("objects", "ids", value_type=list[str])


def notebook_experiment(cfg: Mapping[str, object]) -> str:
    """Read the notebook experiment name."""
    return str(_yaml_config(cfg).get_path("notebook", "experiment"))


def notebook_download_ycb(cfg: Mapping[str, object]) -> bool:
    """Read whether notebooks should download missing YCB assets."""
    return bool(_yaml_config(cfg).get_path("notebook", "download_ycb", default=True))


def notebook_augment(cfg: Mapping[str, object]) -> bool:
    """Read whether supervised notebook training should augment data."""
    return bool(_yaml_config(cfg).get_path("notebook", "augment", default=False))


def notebook_mount_drive(cfg: Mapping[str, object]) -> bool:
    """Read whether notebooks should mount Google Drive for persisted storage."""
    return bool(_yaml_config(cfg).get_path("notebook", "mount_drive", default=False))


def notebook_drive_storage_dir(cfg: Mapping[str, object]) -> str:
    """Read the Drive folder name used for persisted notebook storage."""
    return str(
        _yaml_config(cfg).get_path("notebook", "drive_storage_dir", default=DEFAULT_DRIVE_STORAGE_DIR),
    )


def notebook_object_index(cfg: Mapping[str, object]) -> int:
    """Read the configured notebook object index."""
    return int(_yaml_config(cfg).get_path("notebook", "object_index", default=0))


def selected_object_id(cfg: Mapping[str, object]) -> str:
    """Resolve the object identifier selected by ``notebook.object_index``."""
    object_ids = object_ids_from_config(cfg)
    return object_ids[notebook_object_index(cfg)]


def dataset_paths(cfg: Mapping[str, object]) -> dict[str, Path]:
    """Extract common dataset path entries from a composed config."""
    yaml_config = _yaml_config(cfg)
    return {
        "ycb_root": yaml_config.value("paths", "ycb_root", value_type=Path),
        "dataset_root": yaml_config.value("paths", "dataset_root", value_type=Path),
        "mjcf_root": yaml_config.value("paths", "ycb_mjcf", value_type=Path),
        "observations_dir": yaml_config.value("paths", "observations", value_type=Path),
        "output_index": yaml_config.value("paths", "output_index", value_type=Path),
    }


def ensure_dataset_dirs(paths: Mapping[str, Path]) -> None:
    """Create dataset directories expected by notebook data-prep scripts."""
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
    """Download YCB assets when requested and any required object is missing."""
    if download and not all((ycb_root / object_id).is_dir() for object_id in object_ids):
        subprocess.run(
            [sys.executable, "scripts/download_ycb_dataset.py"],
            cwd=root,
            env=env,
            check=True,
        )


def hydra_script_cmd(script: str, *overrides: str) -> list[str]:
    """Build a subprocess argv list for a Hydra ``@hydra.main`` script."""
    return [sys.executable, script, *overrides]


def run_dataset_preparation(
    root: Path,
    *,
    cfg: Mapping[str, object],
    object_ids: list[str],
    download_ycb: bool,
) -> dict[str, Path]:
    """Run the standard notebook dataset preparation script chain."""
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
    for script, extra in (
        ("scripts/prepare_ycb_mjcf.py", ()),
        ("scripts/prepare_data.py", ("prepare.mode=synthetic",)),
        ("scripts/prepare_observations.py", ()),
    ):
        subprocess.run(hydra_script_cmd(script, *extra), cwd=root, env=env, check=True)
    return paths


def supervised_hyperparameters(cfg: Mapping[str, object]) -> tuple[float, int, int]:
    """Read supervised training hyperparameters from the composed config."""
    yaml_config = _yaml_config(cfg)
    learning_rate = yaml_config.value("supervised", "learning_rate", value_type=float, default=1e-3)
    num_epochs = yaml_config.value("supervised", "num_epochs", value_type=int, default=3)
    batch_size = yaml_config.value("supervised", "batch_size", value_type=int, default=2)
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
    """Build keyword arguments for supervised training pipelines."""
    yaml_config = _yaml_config(cfg)
    return {
        "dataset_root": dataset_root,
        "checkpoint_path": checkpoint_path,
        "feature_dim": int(yaml_config.get_path("architecture", "feature_dim")),
        "hidden_dim": int(yaml_config.get_path("architecture", "hidden_dim")),
        "num_layers": int(yaml_config.get_path("architecture", "num_layers")),
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "device": device,
        "seed": seed,
        "experiment_log_dir": yaml_config.value(model_key, "tensorboard", value_type=Path),
        "pretrained_encoder_path": None,
        "resume_checkpoint_path": None,
        "augment": notebook_augment(cfg),
        "min_grasp_score": float(yaml_config.get_path("supervised", "min_grasp_score", default=0.0)),
        "score_repeat_factor": int(yaml_config.get_path("supervised", "score_repeat_factor", default=0)),
        "score_repeat_power": float(yaml_config.get_path("supervised", "score_repeat_power", default=1.0)),
    }


def run_with_optional_mlflow(
    use_mlflow: bool,
    experiment: str,
    log_params: Mapping[str, object],
    pipeline: Callable[..., object],
    **pipeline_kwargs: object,
) -> None:
    """Run a pipeline directly or inside an MLflow run."""
    if use_mlflow:
        if mlflow is None:
            msg = "mlflow is required when tracking is enabled"
            raise RuntimeError(msg)
        with mlflow.start_run(run_name=experiment):
            for key, value in log_params.items():
                mlflow.log_param(key, value)
            pipeline(**pipeline_kwargs)
        return
    pipeline(**pipeline_kwargs)


def print_checkpoint_summary(checkpoint_path: Path, tensorboard_dir: Path) -> None:
    """Print checkpoint existence, size, and TensorBoard log directory."""
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


def require_checkpoint(checkpoint_path: Path, label: str) -> Path:
    """Return ``checkpoint_path`` or raise if the file is missing."""
    exists = checkpoint_path.is_file()
    print(label, {"path": str(checkpoint_path), "exists": exists})
    if not exists:
        msg = f"Missing {label} checkpoint: {checkpoint_path}"
        raise FileNotFoundError(msg)
    return checkpoint_path


def supervised_mlflow_log_params(
    cfg: Mapping[str, object],
    *,
    device: str,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
) -> dict[str, object]:
    """Build MLflow parameter mappings for supervised notebook training."""
    return {
        "download_ycb": notebook_download_ycb(cfg),
        "augment": notebook_augment(cfg),
        "device": device,
        "experiment": notebook_experiment(cfg),
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
    }


def load_notebook_context(
    root: Path,
    config_name: str,
    *,
    overrides: list[str] | None = None,
    config_labels: Mapping[str, str] | str | None = None,
    extra_banner: Mapping[str, object] | None = None,
) -> NotebookContext:
    """Load config, seed the runtime, and print the standard banner."""
    cfg = load_notebook_config(root / "configs", config_name, overrides=overrides)
    yaml_config = FlattenedYAMLConfig(cfg)
    seed, device = configure_seeds_and_device(cfg)
    ctx = NotebookContext(
        root=root,
        config_name=config_name,
        cfg=cfg,
        yaml_config=yaml_config,
        seed=seed,
        device=device,
        experiment=notebook_experiment(cfg),
        object_ids=object_ids_from_config(cfg),
    )
    banner = dict(extra_banner or {})
    if config_name.startswith("training/") and "rl_train" not in config_name:
        learning_rate, num_epochs, batch_size = supervised_hyperparameters(cfg)
        banner.setdefault(
            "supervised",
            {"learning_rate": learning_rate, "num_epochs": num_epochs, "batch_size": batch_size},
        )
    if "rl_train" in config_name:
        banner.setdefault("training_object", selected_object_id(cfg))
        banner.setdefault(
            "rl",
            {
                "learning_rate": yaml_config.value("rl", "learning_rate", value_type=float, default=3e-4),
                "num_updates": yaml_config.value("rl", "num_updates", value_type=int, default=10),
            },
        )
    print_runtime_banner(
        experiment=ctx.experiment,
        config_names=config_labels if config_labels is not None else config_name,
        root=root,
        device=device,
        seed=seed,
        **banner,
    )
    return ctx


def prepare_context_dataset(ctx: NotebookContext) -> dict[str, Path]:
    """Run dataset preparation for a loaded notebook context."""
    paths = run_dataset_preparation(
        ctx.root,
        cfg=ctx.cfg,
        object_ids=ctx.object_ids,
        download_ycb=notebook_download_ycb(ctx.cfg),
    )
    dataset_root = paths["dataset_root"]
    print("dataset_root:", dataset_root)
    print("records:", sorted(p.name for p in dataset_root.glob("*.npz")))
    return paths


def run_supervised_notebook(
    ctx: NotebookContext,
    *,
    method: str,
    pipeline: Callable[..., object],
    dataset_root: Path,
) -> Path:
    """Train a diffusion or flow model and print the checkpoint summary."""
    from grasping_ai import init_mlflow, setup_logging

    setup_logging(module_name=f"train_{method}")
    use_mlflow = init_mlflow(ctx.cfg)
    checkpoint_path = ctx.yaml_config.value(method, "checkpoint", value_type=Path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    learning_rate, num_epochs, batch_size = supervised_hyperparameters(ctx.cfg)
    train_kwargs = build_supervised_train_kwargs(
        ctx.cfg,
        model_key=method,
        dataset_root=dataset_root,
        checkpoint_path=checkpoint_path,
        device=ctx.device,
        seed=ctx.seed,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    run_with_optional_mlflow(
        use_mlflow,
        ctx.experiment,
        supervised_mlflow_log_params(
            ctx.cfg,
            device=ctx.device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
        ),
        pipeline,
        **train_kwargs,
    )
    print_checkpoint_summary(checkpoint_path, ctx.yaml_config.value(method, "tensorboard", value_type=Path))
    return checkpoint_path


def run_rl_training_notebook(ctx: NotebookContext, paths: Mapping[str, Path]) -> Path:
    """Train the PPO policy and print the checkpoint summary."""
    from grasping_ai import setup_logging
    from grasping_ai.pipelines.train_rl import run_rl_training_pipeline

    setup_logging(module_name="train_rl")
    policy_checkpoint = ctx.yaml_config.value("rl", "checkpoint", value_type=Path)
    policy_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    robot_xml = ctx.yaml_config.value("robot", "description", value_type=Path)
    print("robot_xml:", robot_xml)
    print("ycb_mjcf:", paths["mjcf_root"])
    learning_rate = ctx.yaml_config.value("rl", "learning_rate", value_type=float, default=3e-4)
    num_updates = ctx.yaml_config.value("rl", "num_updates", value_type=int, default=10)
    run_rl_training_pipeline(
        robot_xml_path=robot_xml,
        ycb_root=paths["mjcf_root"],
        object_ids=[selected_object_id(ctx.cfg)],
        policy_checkpoint_path=policy_checkpoint,
        observation_dim=int(ctx.yaml_config.get_path("rl", "observation_dim")),
        action_dim=int(ctx.yaml_config.get_path("rl", "action_dim")),
        hidden_dim=int(ctx.yaml_config.get_path("rl", "hidden_dim")),
        learning_rate=learning_rate,
        num_updates=num_updates,
        gamma=float(ctx.yaml_config.get_path("rl", "gamma")),
        device=ctx.device,
        seed=ctx.seed,
        experiment_log_dir=ctx.yaml_config.value("rl", "tensorboard", value_type=Path),
    )
    print_checkpoint_summary(policy_checkpoint, ctx.yaml_config.value("rl", "tensorboard", value_type=Path))
    return policy_checkpoint


def simulation_outcome_records(object_id: str, outcomes: list[Mapping[str, object]]) -> list[dict[str, object]]:
    """Convert simulation outcome mappings into JSONL-ready report records."""
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
    """Compute the fraction of simulation outcomes with ``lift_success=True``."""
    if not outcomes:
        return None
    return sum(1 for row in outcomes if bool(row.get("lift_success"))) / len(outcomes)


def evaluation_runtime_kwargs(ctx: NotebookContext) -> dict[str, object]:
    """Read shared analytical/simulation evaluation settings."""
    close_default = ctx.yaml_config.get_path("robot", "gripper", "close_command") or [0.0]
    return {
        "friction_coefficient": ctx.yaml_config.value(
            "metrics", "friction_coefficient", value_type=float, default=0.5,
        ),
        "lift_height_threshold": ctx.yaml_config.value(
            "metrics", "lift_height_threshold", value_type=float, default=0.05,
        ),
        "contact_clearance": ctx.yaml_config.value(
            "metrics", "collision_clearance", value_type=float, default=0.005,
        ),
        "wrench_regularization": ctx.yaml_config.value(
            "metrics", "wrench_regularization", value_type=float, default=1.0,
        ),
        "filter_collisions": ctx.yaml_config.value(
            "evaluation", "filter_collisions", value_type=bool, default=False,
        ),
        "num_simulation_steps": int(ctx.yaml_config.get_path("num_steps")),
        "gripper_close_command": np.asarray(close_default, dtype=np.float64),
    }


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
    """Run inference, analytical evaluation, and simulation for one generative method."""
    yaml_config = _yaml_config(cfg)
    run_single_object_grasp_inference(
        checkpoint_path=checkpoint_path,
        output_path=grasp_path,
        method=method,
        feature_dim=int(yaml_config.get_path("architecture", "feature_dim")),
        num_steps=int(yaml_config.get_path(method, "inference_steps")),
        num_grasps=int(yaml_config.get_path("architecture", "num_grasps")),
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
    report_path = yaml_config.value("evaluation", "analytical_report", value_type=Path)
    aggregated = aggregate_evaluation_results({object_id: results})
    write_evaluation_report(report_path, aggregated, None, per_object_results=None)

    sim_path = reports_dir / f"{method}_simulation_outcomes_{object_id}.jsonl"
    sim_outcomes = run_simulation_sweep(
        grasp_poses=grasps,
        object_id=object_id,
        ycb_root=yaml_config.value("paths", "ycb_mjcf", value_type=Path),
        robot_xml_path=yaml_config.value("robot", "description", value_type=Path),
        table_xml_path=None,
        num_simulation_steps=num_simulation_steps,
        gripper_close_command=gripper_close_command,
    )
    write_jsonl_records(sim_path, simulation_outcome_records(object_id, sim_outcomes))
    return aggregated, sim_outcomes, report_path, sim_path


def run_generative_eval_notebook(
    ctx: NotebookContext,
    *,
    method: str,
    dataset_paths: Mapping[str, Path],
) -> dict[str, object]:
    """Run generative inference/evaluation and print a JSON summary."""
    from grasping_ai import init_mlflow, setup_logging

    object_id = selected_object_id(ctx.cfg)
    exports_dir = ctx.yaml_config.value("paths", "exports", value_type=Path)
    reports_dir = ctx.yaml_config.value("paths", "reports", value_type=Path)
    exports_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    observations_dir = dataset_paths["observations_dir"]
    observation_path = observations_dir / f"{object_id}.npy"
    eval_kwargs = evaluation_runtime_kwargs(ctx)
    checkpoint_path = require_checkpoint(ctx.yaml_config.value(method, "checkpoint", value_type=Path), method)

    setup_logging(module_name=f"evaluate_{method}")
    use_mlflow = init_mlflow(ctx.cfg)
    aggregated, sim_outcomes, report_path, sim_path = run_generative_evaluation(
        method=method,
        cfg=ctx.cfg,
        checkpoint_path=checkpoint_path,
        grasp_path=exports_dir / f"{method}_grasp_poses_{object_id}.npy",
        object_id=object_id,
        object_point_cloud=np.load(observation_path),
        gripper_point_cloud=np.load(observations_dir / "gripper.npy"),
        observation_path=observation_path,
        reports_dir=reports_dir,
        device=ctx.device,
        seed=ctx.seed,
        **eval_kwargs,
    )
    sim_rate = lift_success_rate(sim_outcomes)
    summary = {
        "object_id": object_id,
        f"{method}_aggregated": aggregated,
        f"{method}_sim_lift_success_rate": sim_rate,
        "reports": {
            f"{method}_analytical": str(report_path),
            f"{method}_simulation": str(sim_path),
        },
    }
    print(json.dumps(summary, indent=2, default=str))
    if use_mlflow:
        if mlflow is None:
            msg = "mlflow is required when tracking is enabled"
            raise RuntimeError(msg)
        with mlflow.start_run(run_name=ctx.experiment):
            mlflow.log_param("object_id", object_id)
            for key, val in aggregated.items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(f"{method}_{key}", val)
            if sim_rate is not None:
                mlflow.log_metric(f"{method}_sim_lift_success_rate", sim_rate)
    return summary


def run_rl_eval_notebook(ctx: NotebookContext) -> dict[str, object]:
    """Roll out the exported RL policy and print a JSON summary."""
    from run_rl_evaluation import run_rl_evaluation_main

    object_id = selected_object_id(ctx.cfg)
    reports_dir = ctx.yaml_config.value("paths", "reports", value_type=Path)
    reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = require_checkpoint(ctx.yaml_config.value("rl", "checkpoint", value_type=Path), "rl")
    output_path = ctx.yaml_config.value("evaluation", "rollout_report", value_type=Path)
    run_rl_evaluation_main(
        policy_checkpoint_path=checkpoint_path,
        robot_xml_path=ctx.yaml_config.value("robot", "description", value_type=Path),
        ycb_root=ctx.yaml_config.value("paths", "ycb_mjcf", value_type=Path),
        object_id=object_id,
        observation_dim=int(ctx.yaml_config.get_path("rl", "observation_dim")),
        action_dim=int(ctx.yaml_config.get_path("rl", "action_dim")),
        output_path=output_path,
        episodes=ctx.yaml_config.value("evaluation", "episodes", value_type=int, default=5),
        max_steps=ctx.yaml_config.value("evaluation", "max_steps", value_type=int, default=100),
        device=ctx.device,
        seed=ctx.seed,
        table_xml_path=None,
        observation_dim_from_env=False,
        action_dim_from_env=False,
        stochastic=ctx.yaml_config.value("evaluation", "stochastic", value_type=bool, default=False),
        exploration_noise=ctx.yaml_config.value("evaluation", "exploration_noise", value_type=float, default=0.1),
    )
    summary = {"object_id": object_id, "reports": {"rl_rollout": str(output_path)}}
    print(json.dumps(summary, indent=2, default=str))
    return summary
