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

    object_id = object_ids[0] if object_ids else "default"

    env_state, _sim_step, _contacts = build_rl_environment(
        robot_xml_path, ycb_root, object_id, observation_dim, action_dim,
    )

    state_dict: dict[str, Any] = env_state  # type: ignore[assignment]
    mj_model: Any = state_dict["model"]
    env_obs_dim = mj_model.nq + mj_model.nv
    env_act_dim = mj_model.nu

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

    from grasping_ai.models.rl_policy import build_policy_network
    from grasping_ai.training.rl_trainer import (
        build_rl_training_step,
        run_rl_training_loop,
    )

    if seed is not None:
        torch.manual_seed(seed)

    policy = build_policy_network(observation_dim, action_dim, hidden_dim, 2)
    policy_module = cast(torch.nn.Module, policy)
    policy_module.to(torch.device(device))
    optimizer = torch.optim.Adam(
        policy_module.parameters(), lr=learning_rate
    )

    update_step = build_rl_training_step(
        policy_module, optimizer, clip_ratio=0.2, entropy_coefficient=0.0,
        device=device, gamma=gamma,
    )

    def rollout_generator():
        """Yield rollouts by stepping the simulation environment."""
        while True:
            rollout = collect_rl_rollout(
                env_state, lambda obs: policy(  # type: ignore[misc]
                    torch.from_numpy(obs).float().unsqueeze(0).to(device)
                ).detach().squeeze(0).cpu().numpy(),
                num_steps=64,
            )
            yield rollout

    rollout_iter = iter(rollout_generator())

    policy_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "robot_xml_path": str(robot_xml_path),
        "ycb_root": str(ycb_root),
        "object_id": object_id,
        "policy_checkpoint_path": str(policy_checkpoint_path),
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "num_updates": num_updates,
        "gamma": gamma,
        "device": device,
        "rollout_step_count": 64,
        "clip_ratio": 0.2,
        "entropy_coefficient": 0.0,
    }
    if seed is not None:
        metadata["seed"] = seed

    run_rl_training_loop(
        update_step, rollout_iter, num_updates,
        policy_checkpoint_path, log_every=10,
        experiment_log_dir=experiment_log_dir,
        metadata=metadata, seed=seed,
    )


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
