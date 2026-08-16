"""Orchestrate multi-step grasping workflows."""

from __future__ import annotations

from grasping_ai.config import SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig

from grasping_ai.data.pointcloud_dataset import resolve_ycb_object_id

from grasping_ai.evaluation.metrics import aggregate_grasp_success_rate

from grasping_ai.pipelines.evaluate import read_jsonl_records

from grasping_ai.sensors.pointcloud_sensor import sample_point_cloud_from_mesh

import os
import subprocess
import sys
from pathlib import Path

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig

def run_workflow_main(
    checkpoint_path: Path,
    output_dir: Path,
    method: str,
    feature_dim: int,
    num_steps: int,
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
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if num_simulation_steps is None:
        raise ValueError("num_simulation_steps is required")
    if gripper_close_command is None:
        raise ValueError("gripper_close_command is required")
    if friction_coefficient is None:
        raise ValueError("friction_coefficient is required")
    if lift_height_threshold is None:
        raise ValueError("lift_height_threshold is required")
    if contact_clearance is None:
        raise ValueError("contact_clearance is required")
    if wrench_regularization is None:
        raise ValueError("wrench_regularization is required")
    if grasp_pose_format is None:
        raise ValueError("grasp_pose_format is required")
    workflow_env = {
        **os.environ,
        "PYTHONPATH": str(src),
        "PYTHONPYCACHEPREFIX": str(root / pycache_dir),
    }

    exports_dir = output_dir / "exports"
    reports_dir = output_dir / "reports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    grasps_path = exports_dir / grasp_candidates_name.format(method=method)
    sim_path = reports_dir / simulation_outcomes_name.format(method=method)
    eval_path = reports_dir / analytical_evaluation_name.format(method=method)

    # Stage 1: generate grasps.
    grasp_inference_cmd: list[str] = [
        sys.executable,
        "scripts/run_grasp_inference.py",
        "--checkpoint",
        str(checkpoint_path),
        "--output",
        str(grasps_path),
        "--method",
        method,
        "--feature-dim",
        str(feature_dim),
        "--num-steps",
        str(num_steps),
        "--num-grasps",
        str(num_grasps),
        "--device",
        device,
        "--seed",
        str(seed),
    ]
    if observation_path is not None:
        grasp_inference_cmd += ["--observation", str(observation_path)]
    else:
        if ycb_root_raw is None or object_id is None:
            raise ValueError("Provide --observation or both --ycb-root and --object-id")
        grasp_inference_cmd += [
            "--ycb-root",
            str(ycb_root_raw),
            "--object-id",
            object_id,
        ]
    logger.info(">>> {}", " ".join(grasp_inference_cmd))
    subprocess.run(grasp_inference_cmd, cwd=root, env=workflow_env, check=True, capture_output=False)

    if robot_xml_path is not None and ycb_root_mjcf is not None and object_id is not None:
        sim_cmd: list[str] = [
            sys.executable,
            "scripts/run_simulation.py",
            "--grasps",
            str(grasps_path),
            "--object-id",
            object_id,
            "--ycb-root",
            str(ycb_root_mjcf),
            "--robot-xml",
            str(robot_xml_path),
            "--output",
            str(sim_path),
            "--num-simulation-steps",
            str(num_simulation_steps),
            "--gripper-close-command",
            str(gripper_close_command),
            "--grasp-pose-format",
            grasp_pose_format,
        ]
        if table_xml_path is not None:
            sim_cmd += ["--table-xml", str(table_xml_path)]
        logger.info(">>> {}", " ".join(sim_cmd))
        subprocess.run(sim_cmd, cwd=root, env=workflow_env, check=True, capture_output=False)
    else:
        logger.info("skipping MuJoCo simulation stage: robot-xml / ycb-mjcf / object-id not fully provided")

    # Stage 3: analytical evaluation.
    if observation_path is not None:
        object_pc_path = observation_path
    elif ycb_root_raw is not None and object_id is not None:
        object_pc_path = output_dir / object_point_cloud_name
        rng = np.random.default_rng(seed)
        mesh_path = resolve_ycb_object_id(ycb_root_raw, object_id)
        object_pc = sample_point_cloud_from_mesh(mesh_path, num_grasps * point_cloud_sample_multiplier, rng)
        np.save(object_pc_path, object_pc.astype(np.float32))
    else:
        raise ValueError("Cannot locate an object point cloud for evaluation")

    if not gripper_point_cloud_path.is_absolute():
        gripper_point_cloud_path = root / gripper_point_cloud_path
    if not gripper_point_cloud_path.is_file():
        msg = (
            f"Gripper point cloud not found at {gripper_point_cloud_path}; "
            "run scripts/prepare_observations.py first"
        )
        raise FileNotFoundError(msg)

    eval_cmd: list[str] = [
        sys.executable,
        "scripts/evaluate.py",
        "--grasps",
        str(grasps_path),
        "--object-id",
        eval_object_key,
        "--object-point-cloud",
        str(object_pc_path),
        "--gripper-point-cloud",
        str(gripper_point_cloud_path),
        "--report",
        str(eval_path),
        "--friction-coefficient",
        str(friction_coefficient),
        "--lift-height-threshold",
        str(lift_height_threshold),
        "--contact-clearance",
        str(contact_clearance),
        "--wrench-regularization",
        str(wrench_regularization),
    ]
    logger.info(">>> {}", " ".join(eval_cmd))
    subprocess.run(eval_cmd, cwd=root, env=workflow_env, check=True, capture_output=False)

    rl_path = reports_dir / rl_rollout_report_name
    if rl_policy_checkpoint_path is not None:
        if (
            robot_xml_path is None
            or ycb_root_mjcf is None
            or object_id is None
            or observation_dim is None
            or action_dim is None
            or rl_episodes is None
            or rl_max_steps is None
        ):
            raise ValueError(
                "RL rollout stage requires --robot-xml, --ycb-mjcf, "
                "--object-id, --observation-dim, --action-dim, "
                "--rl-episodes, --rl-max-steps",
            )
        rl_cmd: list[str] = [
            sys.executable,
            "scripts/run_rl_evaluation.py",
            "--policy-checkpoint",
            str(rl_policy_checkpoint_path),
            "--robot-xml",
            str(robot_xml_path),
            "--ycb-root",
            str(ycb_root_mjcf),
            "--object-id",
            object_id,
            "--observation-dim",
            str(observation_dim),
            "--action-dim",
            str(action_dim),
            "--output",
            str(rl_path),
            "--episodes",
            str(rl_episodes),
            "--max-steps",
            str(rl_max_steps),
            "--device",
            device,
            "--seed",
            str(seed),
        ]
        if table_xml_path is not None:
            rl_cmd += ["--table-xml", str(table_xml_path)]
        logger.info(">>> {}", " ".join(rl_cmd))
        subprocess.run(rl_cmd, cwd=root, env=workflow_env, check=True, capture_output=False)

    summary: dict[str, float] = {}
    if eval_path.is_file():
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
    if sim_path.is_file():
        try:
            outcomes = [
                record for record in read_jsonl_records(sim_path) if record.get("record_type") == "grasp_outcome"
            ]
            if outcomes:
                n = len(outcomes)
                successes = sum(1 for outcome in outcomes if bool(outcome.get("success")))
                summary["simulated_success_rate"] = float(successes / n)
                summary["simulated_object_success_rate"] = aggregate_grasp_success_rate(
                    {eval_object_key: successes > 0}
                )
        except (OSError, ValueError):
            pass
    if rl_path.is_file():
        try:
            episodes = [record for record in read_jsonl_records(rl_path) if record.get("record_type") == "episode"]
            if episodes:
                mean_return = sum(float(record["summary"]["return_total"]) for record in episodes) / len(episodes)
                mean_len = sum(float(record["summary"]["length"]) for record in episodes) / len(episodes)
                summary["rl_mean_return"] = float(mean_return)
                summary["rl_mean_length"] = float(mean_len)
        except (OSError, ValueError, KeyError, TypeError):
            pass

    logger.info("workflow summary:")
    for key, value in summary.items():
        logger.info("  {} = {:.4f}", key, value)

@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/run_workflow")
def main(cfg: DictConfig) -> None:
    yaml_config = FlattenedYAMLConfig(cfg)
    method = str(yaml_config.value("method", "evaluation", "method", value_type=object, script_or=True))
    checkpoint = yaml_config.value("checkpoint", "model", "checkpoint", value_type=Path, script_or=True)
    if checkpoint is None:
        raise ValueError("model.checkpoint path is required")

    output_dir = yaml_config.value("output_dir", "artifacts", "root", value_type=Path, script_or=True, required=True)
    close_default = yaml_config.value("robot", "gripper", "close_command", value_type=list[float]) or [0.0]
    intermediate = yaml_config.get_path("workflow", "intermediate")
    if not isinstance(intermediate, dict):
        msg = "workflow.intermediate must be a mapping"
        raise TypeError(msg)
    rl_report = yaml_config.value(
        "rl_rollout_report", "workflow", "rl_rollout_report", value_type=Path, script_or=True, required=True
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
            "gripper_point_cloud", "workflow", "gripper_point_cloud", value_type=Path, script_or=True, required=True
        ),
        eval_object_key=str(
            yaml_config.value("eval_object_key", "workflow", "eval_object_key", value_type=object, script_or=True)
        ),
        rl_rollout_report_name=rl_report.name,
        grasp_candidates_name=str(intermediate["grasp_candidates"]),
        simulation_outcomes_name=str(intermediate["simulation_outcomes"]),
        analytical_evaluation_name=str(intermediate["analytical_evaluation"]),
        object_point_cloud_name=str(
            yaml_config.value("object_point_cloud", "workflow", "object_point_cloud", value_type=object, script_or=True)
        ),
        point_cloud_sample_multiplier=yaml_config.value(
            "point_cloud_sample_multiplier", "workflow", "point_cloud_sample_multiplier", value_type=int, script_or=True
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
            yaml_config.value("grasp_pose_format", "workflow", "grasp_pose_format", value_type=object, script_or=True)
        ),
    )

if __name__ == "__main__":
    main()
