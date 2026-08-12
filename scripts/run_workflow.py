from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from grasping_ai.evaluation.metrics import aggregate_grasp_success_rate

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _venv_python() -> str:
    return sys.executable


def _env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(SRC)}


def _run(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=_env(), check=True, capture_output=False)


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
) -> None:
    """Run the end-to-end runtime workflow on a single object.

    Composes ``scripts/run_grasp_inference.py`` -> ``scripts/run_simulation.py`` ->
    ``scripts/evaluate.py`` (and optionally ``scripts/run_rl_evaluation.py``) on
    one object identity, writing four artifacts under ``output_dir`` and a
    stdout summary.

    Args:
        checkpoint_path: Trained grasp-generation checkpoint (``grasp_generation.pt``
            or ``flow_grasp.pt``).
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
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    grasps_path = output_dir / "generated_grasps.npy"
    sim_path = output_dir / "simulation_outcomes.json"
    eval_path = output_dir / "evaluation_report.json"

    # Stage 1: generate grasps.
    grasp_inference_cmd: list[str] = [
        _venv_python(),
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
            raise ValueError(
                "Provide --observation or both --ycb-root and --object-id"
            )
        grasp_inference_cmd += [
            "--ycb-root",
            str(ycb_root_raw),
            "--object-id",
            object_id,
        ]
    _run(grasp_inference_cmd)

    # Stage 2: simulate in MuJoCo (only if a robot + MJCF-wrapped YCB are given).
    if robot_xml_path is not None and ycb_root_mjcf is not None and object_id is not None:
        sim_cmd: list[str] = [
            _venv_python(),
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
            "world",
        ]
        if table_xml_path is not None:
            sim_cmd += ["--table-xml", str(table_xml_path)]
        _run(sim_cmd)
    else:
        print(
            "skipping MuJoCo simulation stage: "
            "robot-xml / ycb-mjcf / object-id not fully provided"
        )

    # Stage 3: analytical evaluation.
    if observation_path is not None:
        object_pc_path = observation_path
    elif ycb_root_raw is not None and object_id is not None:
        object_pc_path = output_dir / "object_point_cloud.npy"
        import numpy as np

        from grasping_ai.data.pointcloud_dataset import resolve_ycb_object_id
        from grasping_ai.sensors.pointcloud_sensor import sample_point_cloud_from_mesh

        mesh_path = resolve_ycb_object_id(ycb_root_raw, object_id)
        rng = np.random.default_rng(seed)
        object_pc = sample_point_cloud_from_mesh(mesh_path, num_grasps * 8, rng)
        np.save(object_pc_path, object_pc.astype(np.float32))
    else:
        raise ValueError("Cannot locate an object point cloud for evaluation")

    gripper_pc_path = Path("data/observations/gripper.npy")
    if not gripper_pc_path.is_file():
        raise FileNotFoundError(
            f"Gripper point cloud not found at {gripper_pc_path}; "
            "run scripts/prepare_observations.py first"
        )

    eval_cmd: list[str] = [
        _venv_python(),
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
    _run(eval_cmd)

    # Stage 4: optional RL rollout.
    rl_path = output_dir / "rl_evaluation.json"
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
                "--rl-episodes, --rl-max-steps"
            )
        rl_cmd: list[str] = [
            _venv_python(),
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
        _run(rl_cmd)

    # Stage 5: print combined summary.
    _print_summary(eval_path, sim_path, rl_path)


def _print_summary(
    eval_path: Path, sim_path: Path, rl_path: Path
) -> None:
    summary: dict[str, float] = {}
    if eval_path.is_file():
        try:
            with eval_path.open() as fp:
                report = json.load(fp)
            for key in ("success_rate", "collision_free_rate", "force_closure_rate"):
                if key in report:
                    summary[f"analytical_{key}"] = float(report[key])
            if "object_success_rate" in report:
                summary["analytical_object_success_rate"] = float(
                    report["object_success_rate"]
                )
        except (OSError, json.JSONDecodeError):
            pass
    if sim_path.is_file():
        try:
            with sim_path.open() as fp:
                outcomes = json.load(fp)
            if isinstance(outcomes, list) and outcomes:
                n = len(outcomes)
                successes = sum(
                    1 for o in outcomes if isinstance(o, dict) and o.get("success")
                )
                summary["simulated_success_rate"] = float(successes / n)
                summary["simulated_object_success_rate"] = aggregate_grasp_success_rate(
                    {"object_0": successes > 0}
                )
        except (OSError, json.JSONDecodeError):
            pass
    if rl_path.is_file():
        try:
            with rl_path.open() as fp:
                rl_report = json.load(fp)
            episodes = rl_report.get("episodes", []) if isinstance(rl_report, dict) else []
            if episodes:
                mean_return = sum(
                    e["summary"]["return_total"] for e in episodes
                ) / len(episodes)
                mean_len = sum(e["summary"]["length"] for e in episodes) / len(episodes)
                summary["rl_mean_return"] = float(mean_return)
                summary["rl_mean_length"] = float(mean_len)
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    print("workflow summary:")
    for key, value in summary.items():
        print(f"  {key} = {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the end-to-end runtime workflow on a single object"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        type=str,
        choices=["diffusion", "flow"],
        default="diffusion",
    )
    parser.add_argument("--feature-dim", type=int, required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--num-grasps", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--observation", type=Path, default=None)
    parser.add_argument("--ycb-root", type=Path, default=None)
    parser.add_argument("--ycb-mjcf", type=Path, default=None)
    parser.add_argument("--object-id", type=str, default=None)
    parser.add_argument("--robot-xml", type=Path, default=None)
    parser.add_argument("--observation-dim", type=int, default=None)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--rl-policy-checkpoint", type=Path, default=None)
    parser.add_argument("--rl-episodes", type=int, default=None)
    parser.add_argument("--rl-max-steps", type=int, default=None)
    parser.add_argument("--table-xml", type=Path, default=None)
    parser.add_argument("--num-simulation-steps", type=int, default=50)
    parser.add_argument("--gripper-close-command", type=float, default=0.02)
    parser.add_argument("--friction-coefficient", type=float, default=0.5)
    parser.add_argument("--lift-height-threshold", type=float, default=0.05)
    parser.add_argument("--contact-clearance", type=float, default=0.005)
    parser.add_argument("--wrench-regularization", type=float, default=1.0)
    args = parser.parse_args()
    run_workflow_main(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        method=args.method,
        feature_dim=args.feature_dim,
        num_steps=args.num_steps,
        num_grasps=args.num_grasps,
        device=args.device,
        seed=args.seed,
        observation_path=args.observation,
        ycb_root_raw=args.ycb_root,
        ycb_root_mjcf=args.ycb_mjcf,
        object_id=args.object_id,
        robot_xml_path=args.robot_xml,
        observation_dim=args.observation_dim,
        action_dim=args.action_dim,
        rl_policy_checkpoint_path=args.rl_policy_checkpoint,
        rl_episodes=args.rl_episodes,
        rl_max_steps=args.rl_max_steps,
        table_xml_path=args.table_xml,
        num_simulation_steps=args.num_simulation_steps,
        gripper_close_command=args.gripper_close_command,
        friction_coefficient=args.friction_coefficient,
        lift_height_threshold=args.lift_height_threshold,
        contact_clearance=args.contact_clearance,
        wrench_regularization=args.wrench_regularization,
    )
