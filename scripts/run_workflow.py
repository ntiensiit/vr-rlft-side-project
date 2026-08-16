"""Orchestrate multi-step grasping workflows."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
from loguru import logger

from grasping_ai.config import SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.data.pointcloud_dataset import resolve_ycb_object_id
from grasping_ai.evaluation.metrics import aggregate_grasp_success_rate
from grasping_ai.pipelines.evaluate import read_jsonl_records
from grasping_ai.sensors.pointcloud_sensor import sample_point_cloud_from_mesh

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass(frozen=True)
class _WorkflowOptions:
    """Flat bundle of the hydra-driven workflow settings."""

    checkpoint_path: Path
    output_dir: Path
    method: str
    feature_dim: int
    num_steps: int
    num_grasps: int
    device: str
    seed: int
    gripper_point_cloud_path: Path
    eval_object_key: str
    rl_rollout_report_name: str
    grasp_candidates_name: str
    simulation_outcomes_name: str
    analytical_evaluation_name: str
    object_point_cloud_name: str
    point_cloud_sample_multiplier: int
    pycache_dir: str
    observation_path: Path | None = None
    ycb_root_raw: Path | None = None
    ycb_root_mjcf: Path | None = None
    object_id: str | None = None
    robot_xml_path: Path | None = None
    observation_dim: int | None = None
    action_dim: int | None = None
    rl_policy_checkpoint_path: Path | None = None
    rl_episodes: int | None = None
    rl_max_steps: int | None = None
    table_xml_path: Path | None = None
    num_simulation_steps: int | None = None
    gripper_close_command: float | None = None
    friction_coefficient: float | None = None
    lift_height_threshold: float | None = None
    contact_clearance: float | None = None
    wrench_regularization: float | None = None
    grasp_pose_format: str | None = None


@dataclass(frozen=True)
class _WorkflowPaths:
    """Derived repo, stage-artifact, and report paths shared by the stages."""

    root: Path
    grasps: Path
    simulation: Path
    evaluation: Path
    rl_rollout: Path


def _validate_workflow_options(options: _WorkflowOptions) -> None:
    """Raise ``ValueError`` when a required scalar workflow setting is missing."""
    for name in (
        "num_simulation_steps",
        "gripper_close_command",
        "friction_coefficient",
        "lift_height_threshold",
        "contact_clearance",
        "wrench_regularization",
        "grasp_pose_format",
    ):
        if getattr(options, name) is None:
            msg = f"{name} is required"
            raise ValueError(msg)


def _build_workflow_env(root: Path, pycache_dir: str) -> dict[str, str]:
    """Build the subprocess environment for workflow child commands."""
    return {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
        "PYTHONPYCACHEPREFIX": str(root / pycache_dir),
    }


def _create_output_paths(options: _WorkflowOptions, root: Path) -> _WorkflowPaths:
    """Create the export/report directories and derive the stage artifact paths."""
    exports_dir = options.output_dir / "exports"
    reports_dir = options.output_dir / "reports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return _WorkflowPaths(
        root=root,
        grasps=exports_dir / options.grasp_candidates_name.format(method=options.method),
        simulation=reports_dir / options.simulation_outcomes_name.format(method=options.method),
        evaluation=reports_dir / options.analytical_evaluation_name.format(method=options.method),
        rl_rollout=reports_dir / options.rl_rollout_report_name,
    )


def _run_grasp_inference_stage(options: _WorkflowOptions, paths: _WorkflowPaths, workflow_env: dict[str, str]) -> None:
    """Run the grasp-generation stage."""
    grasp_inference_cmd: list[str] = [
        sys.executable,
        "scripts/run_grasp_inference.py",
        "--checkpoint",
        str(options.checkpoint_path),
        "--output",
        str(paths.grasps),
        "--method",
        options.method,
        "--feature-dim",
        str(options.feature_dim),
        "--num-steps",
        str(options.num_steps),
        "--num-grasps",
        str(options.num_grasps),
        "--device",
        options.device,
        "--seed",
        str(options.seed),
    ]
    if options.observation_path is not None:
        grasp_inference_cmd += ["--observation", str(options.observation_path)]
    else:
        if options.ycb_root_raw is None or options.object_id is None:
            msg = "Provide --observation or both --ycb-root and --object-id"
            raise ValueError(msg)
        grasp_inference_cmd += [
            "--ycb-root",
            str(options.ycb_root_raw),
            "--object-id",
            options.object_id,
        ]
    logger.info(">>> {}", " ".join(grasp_inference_cmd))
    subprocess.run(  # noqa: S603  # fixed internal CLI, no shell
        grasp_inference_cmd, cwd=paths.root, env=workflow_env, check=True, capture_output=False,
    )


def _run_simulation_stage(options: _WorkflowOptions, paths: _WorkflowPaths, workflow_env: dict[str, str]) -> None:
    """Run the MuJoCo grasp simulation stage when scene inputs are provided."""
    if options.robot_xml_path is not None and options.ycb_root_mjcf is not None and options.object_id is not None:
        sim_cmd: list[str] = [
            sys.executable,
            "scripts/run_simulation.py",
            "--grasps",
            str(paths.grasps),
            "--object-id",
            options.object_id,
            "--ycb-root",
            str(options.ycb_root_mjcf),
            "--robot-xml",
            str(options.robot_xml_path),
            "--output",
            str(paths.simulation),
            "--num-simulation-steps",
            str(options.num_simulation_steps),
            "--gripper-close-command",
            str(options.gripper_close_command),
            "--grasp-pose-format",
            options.grasp_pose_format,
        ]
        if options.table_xml_path is not None:
            sim_cmd += ["--table-xml", str(options.table_xml_path)]
        logger.info(">>> {}", " ".join(sim_cmd))
        subprocess.run(  # noqa: S603  # fixed internal CLI, no shell
            sim_cmd, cwd=paths.root, env=workflow_env, check=True, capture_output=False,
        )
    else:
        logger.info("skipping MuJoCo simulation stage: robot-xml / ycb-mjcf / object-id not fully provided")


def _resolve_object_point_cloud(options: _WorkflowOptions) -> Path:
    """Return the object point cloud, sampling it from the YCB mesh when needed."""
    if options.observation_path is not None:
        return options.observation_path
    if options.ycb_root_raw is not None and options.object_id is not None:
        object_pc_path = options.output_dir / options.object_point_cloud_name
        rng = np.random.default_rng(options.seed)
        mesh_path = resolve_ycb_object_id(options.ycb_root_raw, options.object_id)
        object_pc = sample_point_cloud_from_mesh(
            mesh_path, options.num_grasps * options.point_cloud_sample_multiplier, rng,
        )
        np.save(object_pc_path, object_pc.astype(np.float32))
        return object_pc_path
    msg = "Cannot locate an object point cloud for evaluation"
    raise ValueError(msg)


def _resolve_gripper_point_cloud(gripper_point_cloud_path: Path, root: Path) -> Path:
    """Resolve the gripper point cloud against the repo root and require that it exists."""
    if not gripper_point_cloud_path.is_absolute():
        gripper_point_cloud_path = root / gripper_point_cloud_path
    if not gripper_point_cloud_path.is_file():
        msg = (
            f"Gripper point cloud not found at {gripper_point_cloud_path}; "
            "run scripts/prepare_observations.py first"
        )
        raise FileNotFoundError(msg)
    return gripper_point_cloud_path


def _run_evaluation_stage(
    options: _WorkflowOptions,
    paths: _WorkflowPaths,
    object_pc_path: Path,
    workflow_env: dict[str, str],
) -> None:
    """Run the analytical evaluation stage."""
    gripper_point_cloud_path = _resolve_gripper_point_cloud(options.gripper_point_cloud_path, paths.root)
    eval_cmd: list[str] = [
        sys.executable,
        "scripts/evaluate.py",
        "--grasps",
        str(paths.grasps),
        "--object-id",
        options.eval_object_key,
        "--object-point-cloud",
        str(object_pc_path),
        "--gripper-point-cloud",
        str(gripper_point_cloud_path),
        "--report",
        str(paths.evaluation),
        "--friction-coefficient",
        str(options.friction_coefficient),
        "--lift-height-threshold",
        str(options.lift_height_threshold),
        "--contact-clearance",
        str(options.contact_clearance),
        "--wrench-regularization",
        str(options.wrench_regularization),
    ]
    logger.info(">>> {}", " ".join(eval_cmd))
    subprocess.run(  # noqa: S603  # fixed internal CLI, no shell
        eval_cmd, cwd=paths.root, env=workflow_env, check=True, capture_output=False,
    )


def _run_rl_rollout_stage(options: _WorkflowOptions, paths: _WorkflowPaths, workflow_env: dict[str, str]) -> None:
    """Run the optional RL policy rollout stage."""
    if options.rl_policy_checkpoint_path is None:
        return
    if (
        options.robot_xml_path is None
        or options.ycb_root_mjcf is None
        or options.object_id is None
        or options.observation_dim is None
        or options.action_dim is None
        or options.rl_episodes is None
        or options.rl_max_steps is None
    ):
        msg = ("RL rollout stage requires --robot-xml, --ycb-mjcf, "
            "--object-id, --observation-dim, --action-dim, "
            "--rl-episodes, --rl-max-steps")
        raise ValueError(
            msg,
        )
    rl_cmd: list[str] = [
        sys.executable,
        "scripts/run_rl_evaluation.py",
        "--policy-checkpoint",
        str(options.rl_policy_checkpoint_path),
        "--robot-xml",
        str(options.robot_xml_path),
        "--ycb-root",
        str(options.ycb_root_mjcf),
        "--object-id",
        options.object_id,
        "--observation-dim",
        str(options.observation_dim),
        "--action-dim",
        str(options.action_dim),
        "--output",
        str(paths.rl_rollout),
        "--episodes",
        str(options.rl_episodes),
        "--max-steps",
        str(options.rl_max_steps),
        "--device",
        options.device,
        "--seed",
        str(options.seed),
    ]
    if options.table_xml_path is not None:
        rl_cmd += ["--table-xml", str(options.table_xml_path)]
    logger.info(">>> {}", " ".join(rl_cmd))
    subprocess.run(  # noqa: S603  # fixed internal CLI, no shell
        rl_cmd, cwd=paths.root, env=workflow_env, check=True, capture_output=False,
    )


def _read_analytical_summary(eval_path: Path, summary: dict[str, float]) -> None:
    """Merge analytical evaluation metrics from the report into ``summary``."""
    if not eval_path.is_file():
        return
    try:
        for record in read_jsonl_records(eval_path):
            if record.get("record_type") != "summary":
                continue
            for key in ("success_rate", "collision_free_rate", "force_closure_rate"):
                if key in record:
                    summary[f"analytical_{key}"] = float(record[key])
            if "object_success_rate" in record:
                summary["analytical_object_success_rate"] = float(record["object_success_rate"])
            break
    except (OSError, ValueError):
        pass


def _read_simulation_summary(sim_path: Path, eval_object_key: str, summary: dict[str, float]) -> None:
    """Merge simulated grasp success metrics into ``summary``."""
    if not sim_path.is_file():
        return
    try:
        outcomes = [
            record for record in read_jsonl_records(sim_path) if record.get("record_type") == "grasp_outcome"
        ]
        if outcomes:
            n = len(outcomes)
            successes = sum(1 for outcome in outcomes if bool(outcome.get("success")))
            summary["simulated_success_rate"] = float(successes / n)
            summary["simulated_object_success_rate"] = aggregate_grasp_success_rate({eval_object_key: successes > 0})
    except (OSError, ValueError):
        pass


def _read_rl_summary(rl_path: Path, summary: dict[str, float]) -> None:
    """Merge RL rollout return statistics into ``summary``."""
    if not rl_path.is_file():
        return
    try:
        episodes = [record for record in read_jsonl_records(rl_path) if record.get("record_type") == "episode"]
        if episodes:
            mean_return = sum(float(record["summary"]["return_total"]) for record in episodes) / len(episodes)
            mean_len = sum(float(record["summary"]["length"]) for record in episodes) / len(episodes)
            summary["rl_mean_return"] = float(mean_return)
            summary["rl_mean_length"] = float(mean_len)
    except (OSError, ValueError, KeyError, TypeError):
        pass


def _log_workflow_summary(options: _WorkflowOptions, paths: _WorkflowPaths) -> None:
    """Aggregate the stage report metrics and log them."""
    summary: dict[str, float] = {}
    _read_analytical_summary(paths.evaluation, summary)
    _read_simulation_summary(paths.simulation, options.eval_object_key, summary)
    _read_rl_summary(paths.rl_rollout, summary)

    logger.info("workflow summary:")
    for key, value in summary.items():
        logger.info("  {} = {:.4f}", key, value)


def _run_workflow(options: _WorkflowOptions) -> None:
    """Execute the inference, simulation, evaluation, and RL stages in order."""
    root = Path(__file__).resolve().parents[1]
    _validate_workflow_options(options)
    workflow_env = _build_workflow_env(root, options.pycache_dir)
    paths = _create_output_paths(options, root)
    _run_grasp_inference_stage(options, paths, workflow_env)
    _run_simulation_stage(options, paths, workflow_env)
    object_pc_path = _resolve_object_point_cloud(options)
    _run_evaluation_stage(options, paths, object_pc_path, workflow_env)
    _run_rl_rollout_stage(options, paths, workflow_env)
    _log_workflow_summary(options, paths)


def run_workflow_main(  # noqa: PLR0913  # flat signature mirrors the hydra workflow config keys
    checkpoint_path: Path,
    output_dir: Path,
    method: str,
    feature_dim: int,
    num_steps: int,
    *,
    num_grasps: int,
    device: str,
    seed: int,
    gripper_point_cloud_path: Path,
    eval_object_key: str,
    rl_rollout_report_name: str,
    grasp_candidates_name: str,
    simulation_outcomes_name: str,
    analytical_evaluation_name: str,
    object_point_cloud_name: str,
    point_cloud_sample_multiplier: int,
    pycache_dir: str,
    observation_path: Path | None = None,
    ycb_root_raw: Path | None = None,
    ycb_root_mjcf: Path | None = None,
    object_id: str | None = None,
    robot_xml_path: Path | None = None,
    observation_dim: int | None = None,
    action_dim: int | None = None,
    rl_policy_checkpoint_path: Path | None = None,
    rl_episodes: int | None = None,
    rl_max_steps: int | None = None,
    table_xml_path: Path | None = None,
    num_simulation_steps: int | None = None,
    gripper_close_command: float | None = None,
    friction_coefficient: float | None = None,
    lift_height_threshold: float | None = None,
    contact_clearance: float | None = None,
    wrench_regularization: float | None = None,
    grasp_pose_format: str | None = None,
) -> None:
    """Run the end-to-end runtime workflow on a single object.

    Composes ``scripts/run_grasp_inference.py`` -> ``scripts/run_simulation.py`` ->
    ``scripts/evaluate.py`` (and optionally ``scripts/run_rl_evaluation.py``) on
    one object identity, writing four artifacts under ``output_dir`` and a
    stdout summary.

    Args:
        checkpoint_path: Trained grasp-generation checkpoint (``diffusion_grasp_generator.pt``
            or ``flow_grasp_generator.pt``).
        output_dir: Destination directory for all generated artifacts.
        method: ``"diffusion"`` or ``"flow"``; must match the trained checkpoint.
        feature_dim: Conditioning feature dimension expected by the model.
        num_steps: Diffusion denoising steps or flow integration steps.
        num_grasps: Number of candidate grasps to generate.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.
        seed: Random seed used for both inference and RL rollout.
        gripper_point_cloud_path: Path to the gripper point-cloud ``.npy`` file.
        eval_object_key: Object key recorded in the evaluation artifacts.
        rl_rollout_report_name: File name for the RL rollout report.
        grasp_candidates_name: File-name template for the grasp candidates artifact.
        simulation_outcomes_name: File-name template for the simulation outcomes report.
        analytical_evaluation_name: File-name template for the analytical evaluation report.
        object_point_cloud_name: File name for the sampled object point cloud.
        point_cloud_sample_multiplier: Multiplier applied to ``num_grasps`` when
            sampling the fallback object point cloud.
        pycache_dir: Directory used for ``PYTHONPYCACHEPREFIX``.
        observation_path: Optional path to a per-object point-cloud ``.npy``.
        ycb_root_raw: Optional raw YCB root (used for sampling when
            ``observation_path`` is not given).
        ycb_root_mjcf: Optional MJCF-wrapped YCB root; required for the
            simulation and RL stages.
        object_id: Optional YCB object identifier; required for the mesh
            fallback and for the simulation/RL stages.
        robot_xml_path: Optional robot MJCF; required for the simulation and
            RL stages.
        observation_dim: Optional policy observation dimension; required for
            the RL rollout stage.
        action_dim: Optional policy action dimension; required for the RL
            rollout stage.
        rl_policy_checkpoint_path: Optional RL checkpoint; if supplied, the
            RL rollout stage runs.
        rl_episodes: Episodes for the RL rollout; required if
            ``rl_policy_checkpoint_path`` is supplied.
        rl_max_steps: Max steps per RL episode; required if
            ``rl_policy_checkpoint_path`` is supplied.
        table_xml_path: Optional path to a table/workbench MJCF description.
        num_simulation_steps: Number of physics steps per simulated grasp.
        gripper_close_command: Gripper command used to close the gripper.
        friction_coefficient: Friction coefficient used by force closure.
        lift_height_threshold: Height threshold used by lift success.
        contact_clearance: Clearance threshold used by contact detection.
        wrench_regularization: Wrench regularization used by force closure.
        grasp_pose_format: Coordinate frame format for target grasp poses
            (``"world"`` or ``"object"``).
    """
    options = _WorkflowOptions(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        method=method,
        feature_dim=feature_dim,
        num_steps=num_steps,
        num_grasps=num_grasps,
        device=device,
        seed=seed,
        gripper_point_cloud_path=gripper_point_cloud_path,
        eval_object_key=eval_object_key,
        rl_rollout_report_name=rl_rollout_report_name,
        grasp_candidates_name=grasp_candidates_name,
        simulation_outcomes_name=simulation_outcomes_name,
        analytical_evaluation_name=analytical_evaluation_name,
        object_point_cloud_name=object_point_cloud_name,
        point_cloud_sample_multiplier=point_cloud_sample_multiplier,
        pycache_dir=pycache_dir,
        observation_path=observation_path,
        ycb_root_raw=ycb_root_raw,
        ycb_root_mjcf=ycb_root_mjcf,
        object_id=object_id,
        robot_xml_path=robot_xml_path,
        observation_dim=observation_dim,
        action_dim=action_dim,
        rl_policy_checkpoint_path=rl_policy_checkpoint_path,
        rl_episodes=rl_episodes,
        rl_max_steps=rl_max_steps,
        table_xml_path=table_xml_path,
        num_simulation_steps=num_simulation_steps,
        gripper_close_command=gripper_close_command,
        friction_coefficient=friction_coefficient,
        lift_height_threshold=lift_height_threshold,
        contact_clearance=contact_clearance,
        wrench_regularization=wrench_regularization,
        grasp_pose_format=grasp_pose_format,
    )
    _run_workflow(options)


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/run_workflow")
def main(cfg: DictConfig) -> None:
    """Run the configured end-to-end grasping workflow."""
    yaml_config = FlattenedYAMLConfig(cfg)
    method = str(yaml_config.value("method", "evaluation", "method", value_type=object, script_or=True))
    checkpoint = yaml_config.value("checkpoint", "model", "checkpoint", value_type=Path, script_or=True)
    if checkpoint is None:
        msg = "model.checkpoint path is required"
        raise ValueError(msg)

    output_dir = yaml_config.value("output_dir", "artifacts", "root", value_type=Path, script_or=True, required=True)
    close_default = yaml_config.value("robot", "gripper", "close_command", value_type=list[float]) or [0.0]
    intermediate = yaml_config.get_path("workflow", "intermediate")
    if not isinstance(intermediate, dict):
        msg = "workflow.intermediate must be a mapping"
        raise TypeError(msg)
    rl_report = yaml_config.value(
        "rl_rollout_report", "workflow", "rl_rollout_report", value_type=Path, script_or=True, required=True,
    )

    run_workflow_main(
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        method=method,
        feature_dim=yaml_config.value("architecture", "feature_dim", value_type=int),
        num_steps=yaml_config.value("model", "inference_steps", value_type=int),
        num_grasps=yaml_config.value("architecture", "num_grasps", value_type=int),
        device=str(yaml_config.get("device")),
        seed=yaml_config.value("seed", value_type=int),
        gripper_point_cloud_path=yaml_config.value(
            "gripper_point_cloud", "workflow", "gripper_point_cloud", value_type=Path, script_or=True, required=True,
        ),
        eval_object_key=str(
            yaml_config.value("eval_object_key", "workflow", "eval_object_key", value_type=object, script_or=True),
        ),
        rl_rollout_report_name=rl_report.name,
        grasp_candidates_name=str(intermediate["grasp_candidates"]),
        simulation_outcomes_name=str(intermediate["simulation_outcomes"]),
        analytical_evaluation_name=str(intermediate["analytical_evaluation"]),
        object_point_cloud_name=str(
            yaml_config.value(
                "object_point_cloud", "workflow", "object_point_cloud", value_type=object, script_or=True,
            ),
        ),
        point_cloud_sample_multiplier=yaml_config.value(
            "point_cloud_sample_multiplier",
            "workflow",
            "point_cloud_sample_multiplier",
            value_type=int,
            script_or=True,
        ),
        pycache_dir=str(yaml_config.value("pycache_dir", "workflow", "pycache_dir", value_type=object, script_or=True)),
        observation_path=yaml_config.value("observation", value_type=Path, script_or=True),
        ycb_root_raw=yaml_config.value("ycb_root", "paths", "ycb_root", value_type=Path, script_or=True),
        ycb_root_mjcf=yaml_config.value("ycb_mjcf", "paths", "ycb_mjcf", value_type=Path, script_or=True),
        object_id=yaml_config.value("object_id", value_type=object, script_or=True),
        robot_xml_path=yaml_config.value("robot_xml", "robot", "description", value_type=Path, script_or=True),
        observation_dim=yaml_config.value("observation_dim", value_type=object, script_or=True),
        action_dim=yaml_config.value("action_dim", value_type=object, script_or=True),
        rl_policy_checkpoint_path=yaml_config.value("rl_policy_checkpoint", value_type=Path, script_or=True),
        rl_episodes=yaml_config.value("rl_episodes", value_type=object, script_or=True),
        rl_max_steps=yaml_config.value("rl_max_steps", value_type=object, script_or=True),
        table_xml_path=yaml_config.value("table_xml", "env", "table_xml", value_type=Path, script_or=True),
        num_simulation_steps=yaml_config.value("num_steps", value_type=int),
        gripper_close_command=close_default[0],
        friction_coefficient=yaml_config.value("metrics", "friction_coefficient", value_type=float),
        lift_height_threshold=yaml_config.value("metrics", "lift_height_threshold", value_type=float),
        contact_clearance=yaml_config.value("metrics", "collision_clearance", value_type=float),
        wrench_regularization=yaml_config.value("metrics", "wrench_regularization", value_type=float),
        grasp_pose_format=str(
            yaml_config.value("grasp_pose_format", "workflow", "grasp_pose_format", value_type=object, script_or=True),
        ),
    )

if __name__ == "__main__":
    main()
