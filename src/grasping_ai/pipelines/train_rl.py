"""RL training pipeline for grasp policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from grasping_ai.utils.path_validation import require_path

if TYPE_CHECKING:
    from pathlib import Path


def run_rl_training_pipeline(
    robot_xml_path: Path,
    ycb_root: Path,
    object_ids: list[str],
    policy_checkpoint_path: Path,
    observation_dim: int,
    action_dim: int,
    hidden_dim: int,
    learning_rate: float,
    num_updates: int,
    gamma: float,
    device: str,
    seed: int | None = None,
    experiment_log_dir: Path | None = None,
    n_steps: int = 64,
    batch_size: int = 64,
    n_epochs: int = 1,
    policy_num_layers: int = 2,
) -> None:
    """Run an end-to-end RL training pipeline using MuJoCo as the environment.

    Args:
        robot_xml_path: Path to the robot MJCF description used in training.
        ycb_root: Root directory of the YCB object set.
        object_ids: YCB object identifiers used during training rollouts. The
            environment tracks a single object body, so exactly one object
            identifier must be supplied.
        policy_checkpoint_path: Destination path for the trained policy.
        observation_dim: Dimensionality of the policy observation vector.
        action_dim: Dimensionality of the policy action vector.
        hidden_dim: Hidden width of the policy and value networks.
        learning_rate: Learning rate for the policy optimizer.
        num_updates: Number of policy update steps to perform.
        gamma: Discount factor for return computation.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.
        seed: Optional random seed for reproducible policy initialization.
        experiment_log_dir: Optional path to write TensorBoard experiment events.
        n_steps: PPO rollout length per environment per update.
        batch_size: PPO minibatch size.
        n_epochs: Number of PPO optimization epochs per update.
        policy_num_layers: Hidden-layer count for the exported legacy policy MLP.
    """
    require_path(robot_xml_path, "robot_xml_path")
    if not robot_xml_path.is_file():
        raise FileNotFoundError(f"Robot XML file not found: {robot_xml_path}")
    if object_ids:
        require_path(ycb_root, "ycb_root")
    if object_ids and not ycb_root.is_dir():
        raise FileNotFoundError(f"YCB root directory not found: {ycb_root}")
    if observation_dim <= 0:
        raise ValueError("observation_dim must be positive")
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if num_updates <= 0:
        raise ValueError("num_updates must be positive")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n_epochs <= 0:
        raise ValueError("n_epochs must be positive")
    if policy_num_layers <= 0:
        raise ValueError("policy_num_layers must be positive")
    if len(object_ids) > 1:
        raise ValueError(
            "object_ids must contain at most one object; "
            "the environment tracks a single object body during RL training",
        )

    env_xml_path = robot_xml_path
    object_name: str | None = None
    if object_ids:
        from grasping_ai.simulation.scene import build_scene_xml
        from grasping_ai.simulation.ycb import find_ycb_mjcf, resolve_ycb_object_directory

        object_dir = resolve_ycb_object_directory(ycb_root, object_ids[0])
        object_xml_path = find_ycb_mjcf(object_dir)
        object_name = object_ids[0]
        env_xml_path = build_scene_xml(robot_xml_path, object_xml_path, None, object_name)

    from stable_baselines3 import PPO

    from grasping_ai.simulation.mujoco_env import MuJoCoGraspingEnv, RewardConfig

    try:
        from grasping_ai.config.config import DEFAULT_CONFIG_DIR, load_project_yaml_config

        _cfg = load_project_yaml_config(DEFAULT_CONFIG_DIR)
        reward_config = RewardConfig.load_from_config(_cfg)
    except Exception:
        reward_config = RewardConfig(
            action_cost_weight=0.01,
            survival_bonus=1.0,
            contact_reward=0.5,
            lift_reward_weight=2.0,
            grasp_success_bonus=5.0,
            lift_height_threshold=0.05,
            drop_height_threshold=0.1,
        )
    env = MuJoCoGraspingEnv(env_xml_path, object_name=object_name, reward_config=reward_config)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    if obs_shape is None or act_shape is None:
        raise ValueError("Environment space shapes cannot be None")

    if observation_dim != obs_shape[0]:
        raise ValueError(
            f"observation_dim ({observation_dim}) does not match environment observation dimension ({obs_shape[0]})",
        )
    if action_dim != act_shape[0]:
        raise ValueError(f"action_dim ({action_dim}) does not match environment action dimension ({act_shape[0]})")

    from grasping_ai.models.rl_policy import build_sb3_net_arch

    policy_kwargs = {
        "net_arch": build_sb3_net_arch(hidden_dim, policy_num_layers),
        "activation_fn": torch.nn.Tanh,
    }

    sb3_model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        device=device,
        seed=seed,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(experiment_log_dir) if experiment_log_dir else None,
        verbose=1,
    )

    total_timesteps = num_updates * n_steps
    sb3_model.learn(total_timesteps=total_timesteps)

    from grasping_ai.models.rl_policy import (
        build_policy_network,
        copy_sb3_policy_weights,
        save_rl_policy_checkpoint,
    )

    legacy_policy = build_policy_network(observation_dim, action_dim, hidden_dim, policy_num_layers)
    copy_sb3_policy_weights(sb3_model.policy, legacy_policy)

    save_rl_policy_checkpoint(
        policy=legacy_policy,
        policy_checkpoint_path=policy_checkpoint_path,
        epoch=num_updates,
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=policy_num_layers,
        seed=seed,
    )
