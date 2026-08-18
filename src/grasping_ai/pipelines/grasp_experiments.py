"""Comparable physical grasp experiments for a single YCB object."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from grasping_ai.data.pointcloud_dataset import load_grasp_sample
from grasping_ai.inference.policy_runner import build_rl_policy_runner, run_policy_step
from grasping_ai.pipelines.evaluate import write_jsonl_records
from grasping_ai.pipelines.simulate_grasp import simulate_grasp
from grasping_ai.pipelines.train_rl import configure_grasp_conditioned_reset
from grasping_ai.robotics.gripper import panda_fingertip_object_contacts
from grasping_ai.simulation.mujoco_env import MuJoCoGraspingEnv, read_body_pose
from grasping_ai.simulation.scene import build_scene_xml
from grasping_ai.simulation.ycb import find_ycb_mjcf, resolve_ycb_object_directory
from grasping_ai.training.checkpoint_io import load_torch_checkpoint

if TYPE_CHECKING:
    from pathlib import Path


def _load_candidate(path: Path, object_id: str, index: int) -> np.ndarray:
    sample = load_grasp_sample(path)
    if sample.get("object_id") != object_id:
        raise ValueError(f"grasp archive object_id {sample.get('object_id')!r} does not match {object_id!r}")
    poses = np.asarray(sample["grasp_poses"], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not 0 <= index < len(poses):  # noqa: PLR2004
        raise ValueError("invalid grasp candidate index or pose array")
    valid = sample.get("sim_validated")
    if valid is not None and not bool(np.asarray(valid)[index]):
        raise ValueError(f"candidate {index} is not simulation-validated")
    if sample.get("grasp_pose_format", "object") != "object":
        raise ValueError("comparison experiments require object-frame grasp poses")
    return poses[index]


def _make_env(robot_xml: Path, ycb_root: Path, table_xml: Path, object_id: str) -> MuJoCoGraspingEnv:
    object_xml = find_ycb_mjcf(resolve_ycb_object_directory(ycb_root, object_id))
    return MuJoCoGraspingEnv(
        build_scene_xml(robot_xml, object_xml, table_xml, object_id),
        object_name=object_id,
        place_object_on_table=True,
        control_mode="normalized_delta",
        task_observations=True,
    )


def _world_grasp(env: MuJoCoGraspingEnv, object_id: str, object_grasp: np.ndarray) -> np.ndarray:
    return read_body_pose(env._state, object_id) @ object_grasp  # noqa: SLF001 - physical scene state is read-only


def _rollout(  # noqa: PLR0913, PLR0917
    env: MuJoCoGraspingEnv, runner: object, obs: np.ndarray, object_id: str, max_steps: int, lift_threshold: float,
) -> dict[str, object]:
    initial_height = float(read_body_pose(env._state, object_id)[2, 3])  # noqa: SLF001
    max_height = initial_height
    max_contact_count = 0.0
    bilateral_steps = 0
    max_bilateral_height = initial_height
    physical_success = False
    success_step: int | None = None
    total_reward = 0.0
    for _step in range(max_steps):
        action = run_policy_step(runner, obs)  # type: ignore[arg-type]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        state = env._state  # noqa: SLF001
        count, bilateral = panda_fingertip_object_contacts(state["model"], state["data"], object_id)
        max_contact_count = max(max_contact_count, float(count))
        bilateral_steps += int(bilateral)
        object_height = float(read_body_pose(state, object_id)[2, 3])
        max_height = max(max_height, object_height)
        if bool(info["bilateral_contact"]):
            max_bilateral_height = max(max_bilateral_height, object_height)
            if float(info["height_gain"]) >= lift_threshold:
                physical_success = True
                success_step = _step
        if terminated or truncated:
            break
    final_state = env._state  # noqa: SLF001
    count, bilateral = panda_fingertip_object_contacts(final_state["model"], final_state["data"], object_id)
    gain = max_height - initial_height
    return {
        "initial_height": initial_height,
        "final_height": float(read_body_pose(final_state, object_id)[2, 3]),
        "max_object_height": max_height,
        "max_bilateral_height": max_bilateral_height,
        "lift_distance": gain,
        "contact_count": max_contact_count,
        "final_contact_count": float(count),
        "bilateral_contact": bool(bilateral),
        "contact_sustained": bilateral_steps > 0,
        "success_step": success_step,
        "episode_return": total_reward,
        "success": physical_success,
    }


def run_grasp_experiments(  # noqa: PLR0913
    *, grasp_file: Path, object_id: str, grasp_index: int, robot_xml: Path, ycb_root: Path,
    table_xml: Path, policy_checkpoint: Path, output: Path, episodes: int, max_steps: int,
    baseline_simulation_steps: int, lift_threshold: float, pregrasp_distance: float, device: str, seed: int,
) -> None:
    """Run pose baseline, RL from reset, and RL from an IK grasp configuration."""
    if not policy_checkpoint.is_file():
        raise FileNotFoundError(
            f"RL policy checkpoint not found: {policy_checkpoint}. "
            "Train an object-specific policy with scripts/train_rl.py first.",
        )
    object_grasp = _load_candidate(grasp_file, object_id, grasp_index)
    baseline_env = _make_env(robot_xml, ycb_root, table_xml, object_id)
    baseline_env.reset(seed=seed)
    world_grasp = _world_grasp(baseline_env, object_id, object_grasp)
    baseline = simulate_grasp(
        world_grasp, object_id, ycb_root, robot_xml, table_xml, baseline_simulation_steps,
        np.array([0.0], dtype=np.float64), lift_height_threshold=lift_threshold,
    )

    checkpoint = load_torch_checkpoint(policy_checkpoint, device)
    records: list[dict[str, object]] = [
        {"record_type": "header", "object_id": object_id, "grasp_file": str(grasp_file)},
    ]
    records.append(
        {"record_type": "experiment", "experiment": "pose_baseline", "candidate_index": grasp_index, **baseline},
    )
    for experiment in ("rl_from_reset", "rl_from_pose"):
        env = _make_env(robot_xml, ycb_root, table_xml, object_id)
        if experiment == "rl_from_pose":
            try:
                configure_grasp_conditioned_reset(
                    env,
                    robot_xml,
                    object_id,
                    grasp_file,
                    grasp_index,
                    pregrasp_distance,
                )
            except ValueError as exc:
                records.append(
                    {
                        "record_type": "experiment",
                        "experiment": experiment,
                        "candidate_index": grasp_index,
                        "success": False,
                        "error": str(exc),
                    },
                )
                continue
        low, high = np.asarray(env.action_space.low), np.asarray(env.action_space.high)
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            runner = build_rl_policy_runner(
                checkpoint, env.observation_space.shape[0], env.action_space.shape[0], device,
                action_low=low, action_high=high, seed=seed + episode,
            )
            result = _rollout(env, runner, obs, object_id, max_steps, lift_threshold)
            records.append(
                {"record_type": "experiment", "experiment": experiment, "episode": episode,
                 "candidate_index": grasp_index, **result},
            )
    write_jsonl_records(output, records)
