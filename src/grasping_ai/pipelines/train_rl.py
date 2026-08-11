from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from grasping_ai.simulation.mujoco_env import ContactReporter, SimulationStep


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
        object_ids: YCB object identifiers used during training rollouts.
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

    object_ids[0] if object_ids else "default"

    from stable_baselines3 import PPO

    from grasping_ai.simulation.mujoco_env import MuJoCoGraspingEnv

    env = MuJoCoGraspingEnv(robot_xml_path)

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

    from grasping_ai.models.rl_policy import build_policy_network

    legacy_policy = cast(
        torch.nn.Module,
        build_policy_network(observation_dim, action_dim, hidden_dim, 2),
    )
    sb3_policy = sb3_model.policy
    sb3_pi = sb3_policy.mlp_extractor.policy_net

    layer0 = cast(torch.nn.Linear, sb3_pi[0])
    layer2 = cast(torch.nn.Linear, sb3_pi[2])
    action_net = cast(torch.nn.Linear, sb3_policy.action_net)

    legacy_state = legacy_policy.state_dict()
    legacy_state["0.weight"] = layer0.weight.data.clone()
    legacy_state["0.bias"] = layer0.bias.data.clone()
    legacy_state["2.weight"] = layer2.weight.data.clone()
    legacy_state["2.bias"] = layer2.bias.data.clone()
    legacy_state["4.weight"] = action_net.weight.data.clone()
    legacy_state["4.bias"] = action_net.bias.data.clone()

    legacy_policy.load_state_dict(legacy_state)

    policy_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dict: dict[str, Any] = {
        "epoch": num_updates,
        "model_state_dict": legacy_policy.state_dict(),
    }
    if seed is not None:
        checkpoint_dict["seed"] = seed
    torch.save(checkpoint_dict, policy_checkpoint_path)


def build_rl_environment(
    robot_xml_path: Path,
    ycb_root: Path,
    object_id: str,
    observation_dim: int,
    action_dim: int,
) -> tuple[object, SimulationStep, ContactReporter]:
    """Construct a closed-loop RL environment over a MuJoCo scene.

    Args:
        robot_xml_path: Path to the robot MJCF description.
        ycb_root: Root directory of the YCB object set.
        object_id: YCB object identifier used as the manipulation target.
        observation_dim: Dimensionality of observations produced by the env.
        action_dim: Dimensionality of actions accepted by the env.

    Returns:
        A tuple ``(env_state, step, contacts)`` providing a stepping interface
        over the constructed RL environment.
    """
    from grasping_ai.simulation.mujoco_env import (
        create_simulation,
        load_mujoco_model,
    )

    model = load_mujoco_model(robot_xml_path)
    state, step, contacts = create_simulation(model)
    return state, step, contacts


def collect_rl_rollout(
    env_state: object,
    policy_runner: object,
    num_steps: int,
) -> list[tuple[object, object, float, object, bool]]:
    """Collect a rollout of environment transitions under the supplied policy.

    Args:
        env_state: Opaque environment state handle.
        policy_runner: Callable returning an action given an observation.
        num_steps: Number of environment steps to collect.

    Returns:
        A list of ``(obs, action, reward, next_obs, done)`` transition tuples.
    """
    from grasping_ai.simulation.mujoco_env import (
        reset_simulation,
    )

    state_dict: dict[str, Any] = env_state  # type: ignore[assignment]
    mj_model: Any = state_dict["model"]
    mj_data: Any = state_dict["data"]

    reset_simulation(env_state)

    transitions: list[tuple[object, object, float, object, bool]] = []
    for _ in range(num_steps):
        qpos = np.array(mj_data.qpos, copy=True)
        qvel = np.array(mj_data.qvel, copy=True)
        obs = np.concatenate([qpos, qvel]).astype(np.float32)

        runner_fn = policy_runner  # type: ignore[assignment]
        action = runner_fn(obs) if callable(runner_fn) else np.zeros(mj_model.nu)
        action = np.asarray(action, dtype=np.float32)

        ctrl_dim = mj_model.nu
        if action.shape[0] > ctrl_dim:
            action = action[:ctrl_dim]
        elif action.shape[0] < ctrl_dim:
            padded = np.zeros(ctrl_dim, dtype=np.float32)
            padded[: action.shape[0]] = action
            action = padded

        mj_data.ctrl[:] = action

        import mujoco  # type: ignore[import-untyped]
        mujoco.mj_step(mj_model, mj_data)

        new_qpos = np.array(mj_data.qpos, copy=True)
        new_qvel = np.array(mj_data.qvel, copy=True)
        next_obs = np.concatenate([new_qpos, new_qvel]).astype(np.float32)

        reward = -float(np.sum(action**2)) * 0.01
        if np.isfinite(next_obs).all():
            reward += 1.0
        reward = float(np.clip(reward, -10.0, 10.0))

        done = not np.isfinite(next_obs).all()
        if done:
            reset_simulation(env_state)

        transitions.append((obs, action, reward, next_obs, done))

    return transitions
