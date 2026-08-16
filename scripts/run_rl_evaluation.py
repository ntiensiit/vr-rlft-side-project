"""Evaluate RL policies in simulation."""

from __future__ import annotations

from grasping_ai.config import SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig

from grasping_ai.inference.policy_runner import (
    build_rl_policy_runner,
    load_rl_policy_checkpoint,
    run_policy_step,
)

from grasping_ai.pipelines.evaluate import write_jsonl_records

from grasping_ai.simulation.mujoco_env import MuJoCoGraspingEnv

from grasping_ai.simulation.scene import build_scene_xml

from grasping_ai.simulation.ycb import (
    find_ycb_mjcf,
    resolve_ycb_object_directory,
)

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

def run_rl_evaluation_main(
    policy_checkpoint_path: Path,
    robot_xml_path: Path,
    ycb_root: Path,
    object_id: str,
    observation_dim: int,
    action_dim: int,
    output_path: Path,
    episodes: int,
    max_steps: int,
    device: str,
    seed: int,
    table_xml_path: Path | None = None,
    observation_dim_from_env: bool = False,
    action_dim_from_env: bool = False,
    stochastic: bool = False,
    exploration_noise: float = 0.1,
) -> None:
    """Run deterministic rollouts of the exported RL policy and write a JSON trace."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    if not robot_xml_path.is_file():
        msg = f"Robot XML file not found: {robot_xml_path}"
        raise FileNotFoundError(msg)

    if not ycb_root.is_dir():
        msg = f"YCB root directory not found: {ycb_root}"
        raise FileNotFoundError(msg)
    object_xml_path = find_ycb_mjcf(resolve_ycb_object_directory(ycb_root, object_id))
    env_xml_path = build_scene_xml(robot_xml_path, object_xml_path, table_xml_path, object_id)

    env = MuJoCoGraspingEnv(env_xml_path, object_name=object_id)
    env_obs_dim = env.observation_space.shape[0]
    env_act_dim = env.action_space.shape[0]
    if env_obs_dim is None or env_act_dim is None:
        raise ValueError("Environment space shapes cannot be None")

    if observation_dim_from_env:
        observation_dim = env_obs_dim
    if action_dim_from_env:
        action_dim = env_act_dim

    if observation_dim != env_obs_dim:
        msg = (
            f"observation_dim ({observation_dim}) does not match environment observation dimension ({env_obs_dim})"
        )
        raise ValueError(msg)
    if action_dim != env_act_dim:
        msg = f"action_dim ({action_dim}) does not match environment action dimension ({env_act_dim})"
        raise ValueError(msg)

    action_low = np.asarray(env.action_space.low, dtype=np.float64)
    action_high = np.asarray(env.action_space.high, dtype=np.float64)
    finite_bounds = bool(np.all(np.isfinite(action_low)) and np.all(np.isfinite(action_high)))
    if not finite_bounds:
        action_low = np.full(env_act_dim, -1.0, dtype=np.float64)
        action_high = np.full(env_act_dim, 1.0, dtype=np.float64)

    checkpoint = load_rl_policy_checkpoint(policy_checkpoint_path, device)
    runner = build_rl_policy_runner(
        checkpoint,
        observation_dim,
        action_dim,
        device,
        action_low=action_low,
        action_high=action_high,
        stochastic=stochastic,
        exploration_noise=exploration_noise,
        seed=seed,
    )

    episodes_out: list[dict[str, object]] = []
    for episode_idx in range(episodes):
        obs, _info = env.reset(seed=seed + episode_idx)
        trace: list[dict[str, object]] = []
        return_total = 0.0
        final_terminated = False
        final_truncated = False
        for step_idx in range(max_steps):
            action = run_policy_step(runner, obs)
            action = np.clip(action, action_low, action_high).astype(action.dtype)
            obs, reward, terminated, truncated, info = env.step(action)
            return_total += float(reward)
            trace.append(
                {
                    "step": step_idx,
                    "action": action.tolist(),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in info.items()},
                },
            )
            final_terminated = final_terminated or bool(terminated)
            final_truncated = final_truncated or bool(truncated)
            if terminated or truncated:
                break

        episodes_out.append(
            {
                "episode": episode_idx,
                "summary": {
                    "return_total": return_total,
                    "length": len(trace),
                    "final_terminated": final_terminated,
                    "final_truncated": final_truncated,
                },
                "trace": trace,
            },
        )

    records: list[dict[str, object]] = [
        {
            "record_type": "rollout_header",
            "policy_checkpoint": str(policy_checkpoint_path),
            "robot_xml": str(robot_xml_path),
            "ycb_root": str(ycb_root),
            "object_id": object_id,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
        },
        *({"record_type": "episode", **episode} for episode in episodes_out),
    ]
    write_jsonl_records(output_path, records)

@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/run_rl_evaluation")
def main(cfg: DictConfig) -> None:
    yaml_config = FlattenedYAMLConfig(cfg)
    run_rl_evaluation_main(
        policy_checkpoint_path=yaml_config.value(
            "policy_checkpoint", "rl", "checkpoint", value_type=Path, script_or=True, required=True
        ),
        robot_xml_path=yaml_config.value(
            "robot_xml", "robot", "description", value_type=Path, script_or=True, required=True
        ),
        ycb_root=yaml_config.value("ycb_root", "paths", "ycb_mjcf", value_type=Path, script_or=True, required=True),
        object_id=str(
            yaml_config.value(
                "object_id",
                value_type=object,
                default=yaml_config.value("objects", "ids", value_type=list[str])[0],
                script_or=True,
            )
        ),
        observation_dim=int(
            yaml_config.value("observation_dim", "rl", "observation_dim", value_type=object, script_or=True)
        ),
        action_dim=int(yaml_config.value("action_dim", "rl", "action_dim", value_type=object, script_or=True)),
        output_path=yaml_config.value(
            "output", "evaluation", "rollout_report", value_type=Path, script_or=True, required=True
        ),
        episodes=yaml_config.value("evaluation", "episodes", value_type=int),
        max_steps=yaml_config.value("evaluation", "max_steps", value_type=int),
        device=str(yaml_config.get("device")),
        seed=yaml_config.value("seed", value_type=int),
        table_xml_path=yaml_config.value("table_xml", "env", "table_xml", value_type=Path, script_or=True),
        observation_dim_from_env=yaml_config.value(
            "observation_dim_from_env", value_type=bool, default=False, script_or=True
        ),
        action_dim_from_env=yaml_config.value("action_dim_from_env", value_type=bool, default=False, script_or=True),
        stochastic=yaml_config.value("evaluation", "stochastic", value_type=bool, default=False),
        exploration_noise=yaml_config.value("evaluation", "exploration_noise", value_type=float, default=0.1),
    )

if __name__ == "__main__":
    main()
