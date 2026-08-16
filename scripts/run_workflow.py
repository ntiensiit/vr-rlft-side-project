"""Orchestrate multi-step grasping workflows."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from grasping_ai.config.config import (
    SCRIPTS_CONFIG_PATH,
    config_get,
    config_value,
)
from grasping_ai.data.pointcloud_dataset import resolve_ycb_object_id
from grasping_ai.evaluation.metrics import aggregate_grasp_success_rate
from grasping_ai.pipelines.evaluate import read_jsonl_records
from grasping_ai.sensors.pointcloud_sensor import sample_point_cloud_from_mesh


def run_workflow_main(
    checkpoint_path: Path,
    output_dir: Path,
    method: str,
    feature_dim: int,
    num_steps: int,
    num_grasps: int,
    device: str,
    seed: int,
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
    num_simulation_steps: int = 50,
    gripper_close_command: float = 0.02,
    friction_coefficient: float = 0.5,
    lift_height_threshold: float = 0.05,
    contact_clearance: float = 0.005,
    wrench_regularization: float = 1.0,
    grasp_pose_format: str = "world",
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
    workflow_env = {
        **os.environ,
        "PYTHONPATH": str(src),
        "PYTHONPYCACHEPREFIX": str(root / ".pycache"),
    }

    exports_dir = output_dir / "exports"
    reports_dir = output_dir / "reports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    grasps_path = exports_dir / f"{method}_grasp_candidates.npy"
    sim_path = reports_dir / f"{method}_simulation_outcomes.jsonl"
    eval_path = reports_dir / f"{method}_analytical_evaluation_report.jsonl"

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
        object_pc_path = output_dir / "object_point_cloud.npy"
        rng = np.random.default_rng(seed)
        mesh_path = resolve_ycb_object_id(ycb_root_raw, object_id)
        object_pc = sample_point_cloud_from_mesh(mesh_path, num_grasps * 8, rng)
        np.save(object_pc_path, object_pc.astype(np.float32))
    else:
        raise ValueError("Cannot locate an object point cloud for evaluation")

    gripper_pc_path = Path("data/observations/gripper.npy")
    if not gripper_pc_path.is_file():
        msg = (
            f"Gripper point cloud not found at {gripper_pc_path}; run scripts/prepare_observations.py first"
        )
        raise FileNotFoundError(msg)

    eval_cmd: list[str] = [
        sys.executable,
        "scripts/evaluate.py",
        "--grasps",
        str(grasps_path),
        "--object-id",
        "object_0",
        "--object-point-cloud",
        str(object_pc_path),
        "--gripper-point-cloud",
        str(gripper_pc_path),
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

    rl_path = reports_dir / "rl_grasp_rollout_report.jsonl"
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
                summary["simulated_object_success_rate"] = aggregate_grasp_success_rate({"object_0": successes > 0})
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
    method = str(config_value(cfg, "method", "evaluation", "method", value_type=object, script_or=True))
    checkpoint = config_value(cfg, "checkpoint", "model", "checkpoint", value_type=Path, script_or=True)
    if checkpoint is None:
        raise ValueError("model.checkpoint path is required")

    output_dir = config_value(cfg, "output_dir", "artifacts", "root", value_type=Path, script_or=True, required=True)
    close_default = config_value(cfg, "robot", "gripper", "close_command", value_type=list[float]) or [0.0]

    run_workflow_main(
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        method=method,
        feature_dim=config_value(cfg, "architecture", "feature_dim", value_type=int),
        num_steps=config_value(cfg, "model", "inference_steps", value_type=int, default=5),
        num_grasps=config_value(cfg, "architecture", "num_grasps", value_type=int),
        device=str(config_get(cfg, "device")),
        seed=config_value(cfg, "seed", value_type=int),
        observation_path=config_value(cfg, "observation", value_type=Path, script_or=True),
        ycb_root_raw=config_value(cfg, "ycb_root", "paths", "ycb_root", value_type=Path, script_or=True),
        ycb_root_mjcf=config_value(cfg, "ycb_mjcf", "paths", "ycb_mjcf", value_type=Path, script_or=True),
        object_id=config_value(cfg, "object_id", value_type=object, script_or=True),
        robot_xml_path=config_value(cfg, "robot_xml", "robot", "description", value_type=Path, script_or=True),
        observation_dim=config_value(cfg, "observation_dim", value_type=object, script_or=True),
        action_dim=config_value(cfg, "action_dim", value_type=object, script_or=True),
        rl_policy_checkpoint_path=config_value(cfg, "rl_policy_checkpoint", value_type=Path, script_or=True),
        rl_episodes=config_value(cfg, "rl_episodes", value_type=object, script_or=True),
        rl_max_steps=config_value(cfg, "rl_max_steps", value_type=object, script_or=True),
        table_xml_path=config_value(cfg, "table_xml", "env", "table_xml", value_type=Path, script_or=True),
        num_simulation_steps=config_value(cfg, "num_steps", value_type=int),
        gripper_close_command=close_default[0],
        friction_coefficient=config_value(cfg, "metrics", "friction_coefficient", value_type=float),
        lift_height_threshold=config_value(cfg, "metrics", "lift_height_threshold", value_type=float),
        contact_clearance=config_value(cfg, "metrics", "collision_clearance", value_type=float),
        wrench_regularization=config_value(cfg, "metrics", "wrench_regularization", value_type=float),
        grasp_pose_format=str(config_value(cfg, "grasp_pose_format", value_type=object, default="object", script_or=True)),
    )


if __name__ == "__main__":
    main()
