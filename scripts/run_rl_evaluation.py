from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grasping_ai.inference.policy_runner import (
    build_rl_policy_runner,
    load_rl_policy_checkpoint,
    run_policy_step,
)
from grasping_ai.simulation.mujoco_env import MuJoCoGraspingEnv
from grasping_ai.simulation.scene import build_scene_xml
from grasping_ai.simulation.ycb import find_ycb_mjcf, resolve_ycb_object_directory


def _resolve_object_mjcf(ycb_root: Path, object_id: str) -> Path:
    """Return the MJCF path for the YCB object under the YCB root.

    The YCB raw directory ships OpenRAVE KinBody descriptions; production
    scripts consume the MJCF wrappers produced by
    ``scripts/prepare_ycb_mjcf.py``. This helper prefers the MJCF wrapper
    when it exists and falls back to the raw directory only if the
    wrapper is missing (in which case ``build_scene_xml`` will reject the
    KinBody and surface a clear ``ValueError``).
    """
    if not ycb_root.is_dir():
        raise FileNotFoundError(f"YCB root directory not found: {ycb_root}")
    object_dir = resolve_ycb_object_directory(ycb_root, object_id)
    return find_ycb_mjcf(object_dir)


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
    """Run deterministic rollouts of the exported RL policy and write a JSON trace.

    Builds the same Gymnasium-compatible ``MuJoCoGraspingEnv`` that
    ``scripts/train_rl.py`` uses (composed robot + YCB object scene,
    ``RewardConfig`` defaults, validated observation/action dims), then runs
    ``episodes`` independent rollouts of up to ``max_steps`` each, clipping
    policy outputs to the environment's actuator bounds.

    ``ycb_root`` must point to a directory whose object entries ship MJCF
    object files (e.g. ``data/processed/ycb_mjcf``). The raw
    ``data/raw/ycb`` directory uses OpenRAVE KinBody XML and is not
    directly consumable by ``build_scene_xml``.

    Args:
        policy_checkpoint_path: Path to the legacy MLP checkpoint produced by
            ``scripts/train_rl.py``.
        robot_xml_path: Path to the robot MJCF description used at training.
        ycb_root: Root directory of the YCB object set (MJCF-wrapped).
        object_id: YCB object identifier; matches the training object.
        observation_dim: Policy observation dimension (must match the env).
        action_dim: Policy action dimension (must match the env).
        output_path: Destination JSON path.
        episodes: Number of independent rollouts to run.
        max_steps: Maximum steps per episode.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.
        seed: Random seed for ``env.reset(seed=...)``.
        table_xml_path: Optional path to a table/workbench MJCF description.
        observation_dim_from_env: If True, ignore ``observation_dim`` and
            read it from the env's ``observation_space.shape``.
        action_dim_from_env: If True, ignore ``action_dim`` and read it from
            the env's ``action_space.shape``.
        stochastic: When True, sample actions with Gaussian exploration noise.
        exploration_noise: Standard deviation used when ``stochastic`` is True.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    if not robot_xml_path.is_file():
        raise FileNotFoundError(f"Robot XML file not found: {robot_xml_path}")

    object_xml_path = _resolve_object_mjcf(ycb_root, object_id)
    env_xml_path = build_scene_xml(
        robot_xml_path, object_xml_path, table_xml_path, object_id
    )

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
        raise ValueError(
            f"observation_dim ({observation_dim}) does not match "
            f"environment observation dimension ({env_obs_dim})"
        )
    if action_dim != env_act_dim:
        raise ValueError(
            f"action_dim ({action_dim}) does not match "
            f"environment action dimension ({env_act_dim})"
        )

    action_low = np.asarray(env.action_space.low, dtype=np.float64)
    action_high = np.asarray(env.action_space.high, dtype=np.float64)
    finite_bounds = bool(
        np.all(np.isfinite(action_low)) and np.all(np.isfinite(action_high))
    )
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
                    "info": {
                        k: (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in info.items()
                    },
                }
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
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "policy_checkpoint": str(policy_checkpoint_path),
                "robot_xml": str(robot_xml_path),
                "ycb_root": str(ycb_root),
                "object_id": object_id,
                "observation_dim": observation_dim,
                "action_dim": action_dim,
                "episodes": episodes_out,
            },
            fp,
            indent=2,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run deterministic rollouts of the exported RL policy"
    )
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, required=True)
    parser.add_argument("--ycb-root", type=Path, required=True)
    parser.add_argument("--object-id", type=str, required=True)
    parser.add_argument("--observation-dim", type=int, required=True)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--table-xml", type=Path, default=None)
    parser.add_argument(
        "--observation-dim-from-env",
        action="store_true",
        help="Ignore --observation-dim and read it from env.observation_space",
    )
    parser.add_argument(
        "--action-dim-from-env",
        action="store_true",
        help="Ignore --action-dim and read it from env.action_space",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions with Gaussian exploration noise",
    )
    parser.add_argument(
        "--exploration-noise",
        type=float,
        default=0.1,
        help="Exploration noise scale when --stochastic is set",
    )
    args = parser.parse_args()
    run_rl_evaluation_main(
        policy_checkpoint_path=args.policy_checkpoint,
        robot_xml_path=args.robot_xml,
        ycb_root=args.ycb_root,
        object_id=args.object_id,
        observation_dim=args.observation_dim,
        action_dim=args.action_dim,
        output_path=args.output,
        episodes=args.episodes,
        max_steps=args.max_steps,
        device=args.device,
        seed=args.seed,
        table_xml_path=args.table_xml,
        observation_dim_from_env=args.observation_dim_from_env,
        action_dim_from_env=args.action_dim_from_env,
        stochastic=args.stochastic,
        exploration_noise=args.exploration_noise,
    )
