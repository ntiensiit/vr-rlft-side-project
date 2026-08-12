from pathlib import Path
from typing import cast

import torch


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
    """
    if not isinstance(robot_xml_path, Path):
        raise TypeError("robot_xml_path must be a pathlib.Path instance")
    if not robot_xml_path.is_file():
        raise FileNotFoundError(
            f"Robot XML file not found: {robot_xml_path}"
        )
    if object_ids and not isinstance(ycb_root, Path):
        raise TypeError("ycb_root must be a pathlib.Path instance")
    if object_ids and not ycb_root.is_dir():
        raise FileNotFoundError(
            f"YCB root directory not found: {ycb_root}"
        )
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
    if len(object_ids) > 1:
        raise ValueError(
            "object_ids must contain at most one object; the environment "
            "tracks a single object body during RL training"
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
            f"observation_dim ({observation_dim}) does not match "
            f"environment observation dimension ({obs_shape[0]})"
        )
    if action_dim != act_shape[0]:
        raise ValueError(
            f"action_dim ({action_dim}) does not match "
            f"environment action dimension ({act_shape[0]})"
        )

    policy_kwargs = {
        "net_arch": {"pi": [hidden_dim, hidden_dim], "vf": [hidden_dim, hidden_dim]},
        "activation_fn": torch.nn.Tanh,
    }

    sb3_model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=64,
        batch_size=64,
        n_epochs=1,
        gamma=gamma,
        device=device,
        seed=seed,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(experiment_log_dir) if experiment_log_dir else None,
        verbose=1,
    )

    total_timesteps = num_updates * 64
    sb3_model.learn(total_timesteps=total_timesteps)

    from grasping_ai.models.rl_policy import build_policy_network, build_value_network

    legacy_policy = cast(
        torch.nn.Module,
        build_policy_network(observation_dim, action_dim, hidden_dim, 2),
    )
    value_network = cast(
        torch.nn.Module,
        build_value_network(observation_dim, hidden_dim, 2),
    )
    sb3_policy = sb3_model.policy
    sb3_pi = sb3_policy.mlp_extractor.policy_net

    layer0 = cast(torch.nn.Linear, sb3_pi[0])
    layer2 = cast(torch.nn.Linear, sb3_pi[2])
    action_net = cast(torch.nn.Linear, sb3_policy.action_net)
    sb3_vf = sb3_policy.mlp_extractor.value_net
    vf_layer0 = cast(torch.nn.Linear, sb3_vf[0])
    vf_layer2 = cast(torch.nn.Linear, sb3_vf[2])
    vf_out = cast(torch.nn.Linear, sb3_policy.value_net)

    legacy_state = legacy_policy.state_dict()
    legacy_state["0.weight"] = layer0.weight.data.clone()
    legacy_state["0.bias"] = layer0.bias.data.clone()
    legacy_state["2.weight"] = layer2.weight.data.clone()
    legacy_state["2.bias"] = layer2.bias.data.clone()
    legacy_state["4.weight"] = action_net.weight.data.clone()
    legacy_state["4.bias"] = action_net.bias.data.clone()

    value_state = value_network.state_dict()
    value_state["0.weight"] = vf_layer0.weight.data.clone()
    value_state["0.bias"] = vf_layer0.bias.data.clone()
    value_state["2.weight"] = vf_layer2.weight.data.clone()
    value_state["2.bias"] = vf_layer2.bias.data.clone()
    value_state["4.weight"] = vf_out.weight.data.clone()
    value_state["4.bias"] = vf_out.bias.data.clone()
    value_network.load_state_dict(value_state)

    legacy_policy.load_state_dict(legacy_state)

    from grasping_ai.models.rl_policy import save_rl_policy_checkpoint
    save_rl_policy_checkpoint(
        policy=legacy_policy,
        policy_checkpoint_path=policy_checkpoint_path,
        epoch=num_updates,
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        seed=seed,
    )
