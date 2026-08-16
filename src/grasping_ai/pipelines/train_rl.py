"""RL training pipeline for grasp policies."""

from __future__ import annotations

from pathlib import Path

import torch
from stable_baselines3 import PPO

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
from grasping_ai.models.rl_policy import (
    build_policy_network,
    build_sb3_net_arch,
    copy_sb3_policy_weights,
    save_rl_policy_checkpoint,
)
from grasping_ai.simulation import mujoco_env
from grasping_ai.simulation.scene import build_scene_xml
from grasping_ai.simulation.ycb import (
    find_ycb_mjcf,
    resolve_ycb_object_directory,
)
from grasping_ai.utils.path_validation import require_path

ROBOT_XML_PATH = Path(FLATTENED_YAML_CONFIG.get("robot.description", "assets/franka_emika_panda.xml"))
YCB_ROOT = Path(FLATTENED_YAML_CONFIG.get("paths.ycb_root", "data/raw/ycb"))
OBJECT_IDS = tuple(FLATTENED_YAML_CONFIG.get("objects.ids", []))
POLICY_CHECKPOINT_PATH = Path(
    FLATTENED_YAML_CONFIG.get("rl.checkpoint", "artifacts/checkpoints/rl_grasp_policy.pt"),
)
OBSERVATION_DIM = int(FLATTENED_YAML_CONFIG.get("rl.observation_dim", 31))
ACTION_DIM = int(FLATTENED_YAML_CONFIG.get("rl.action_dim", 8))
HIDDEN_DIM = int(FLATTENED_YAML_CONFIG.get("rl.hidden_dim", 32))
LEARNING_RATE = float(FLATTENED_YAML_CONFIG.get("rl.learning_rate", 0.0003))
NUM_UPDATES = int(FLATTENED_YAML_CONFIG.get("rl.num_updates", 10))
GAMMA = float(FLATTENED_YAML_CONFIG.get("rl.gamma", 0.99))
DEVICE = str(FLATTENED_YAML_CONFIG.get("device", "cpu"))
SEED = int(FLATTENED_YAML_CONFIG.get("seed", 42))
N_STEPS = int(FLATTENED_YAML_CONFIG.get("rl.n_steps", 64))
BATCH_SIZE = int(FLATTENED_YAML_CONFIG.get("rl.batch_size", 64))
N_EPOCHS = int(FLATTENED_YAML_CONFIG.get("rl.n_epochs", 1))
POLICY_NUM_LAYERS = int(FLATTENED_YAML_CONFIG.get("rl.policy_num_layers", 2))
VERBOSE = int(FLATTENED_YAML_CONFIG.get("rl.verbose", 1))


def _validate_rl_paths(robot_xml_path: Path, ycb_root: Path, object_ids: list[str]) -> None:
    """Validate RL pipeline filesystem inputs."""
    require_path(robot_xml_path, "robot_xml_path")
    if not robot_xml_path.is_file():
        msg = f"Robot XML file not found: {robot_xml_path}"
        raise FileNotFoundError(msg)
    if object_ids:
        require_path(ycb_root, "ycb_root")
    if object_ids and not ycb_root.is_dir():
        msg = f"YCB root directory not found: {ycb_root}"
        raise FileNotFoundError(msg)


def run_rl_training_pipeline(  # noqa: PLR0915  # end-to-end pipeline orchestration
    robot_xml_path: Path = ROBOT_XML_PATH,
    ycb_root: Path = YCB_ROOT,
    object_ids: tuple[str, ...] = OBJECT_IDS,
    policy_checkpoint_path: Path = POLICY_CHECKPOINT_PATH,
    experiment_log_dir: Path | None = None,
    **options: object,
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
        options: Optional RL hyperparameter overrides keyed by configuration name.
    """
    observation_dim = int(options.pop("observation_dim", OBSERVATION_DIM))
    action_dim = int(options.pop("action_dim", ACTION_DIM))
    hidden_dim = int(options.pop("hidden_dim", HIDDEN_DIM))
    learning_rate = float(options.pop("learning_rate", LEARNING_RATE))
    num_updates = int(options.pop("num_updates", NUM_UPDATES))
    gamma = float(options.pop("gamma", GAMMA))
    device = str(options.pop("device", DEVICE))
    seed = options.pop("seed", SEED)
    n_steps = int(options.pop("n_steps", N_STEPS))
    batch_size = int(options.pop("batch_size", BATCH_SIZE))
    n_epochs = int(options.pop("n_epochs", N_EPOCHS))
    policy_num_layers = int(options.pop("policy_num_layers", POLICY_NUM_LAYERS))
    if options:
        unexpected = ", ".join(sorted(options))
        msg = f"Unexpected RL training options: {unexpected}"
        raise TypeError(msg)

    object_ids = list(object_ids)
    _validate_rl_paths(robot_xml_path, ycb_root, object_ids)
    for name, value in (
        ("observation_dim", observation_dim),
        ("action_dim", action_dim),
        ("hidden_dim", hidden_dim),
        ("learning_rate", learning_rate),
        ("num_updates", num_updates),
        ("n_steps", n_steps),
        ("batch_size", batch_size),
        ("n_epochs", n_epochs),
        ("policy_num_layers", policy_num_layers),
    ):
        if value <= 0:
            msg = f"{name} must be positive"
            raise ValueError(msg)
    if not 0.0 <= gamma <= 1.0:
        msg = "gamma must be in [0, 1]"
        raise ValueError(msg)
    if len(object_ids) > 1:
        msg = (
            "object_ids must contain at most one object; the environment tracks a single object body during RL training"
        )
        raise ValueError(
            msg,
        )

    env_xml_path = robot_xml_path
    object_name: str | None = None
    if object_ids:
        object_dir = resolve_ycb_object_directory(ycb_root, object_ids[0])
        object_xml_path = find_ycb_mjcf(object_dir)
        object_name = object_ids[0]
        env_xml_path = build_scene_xml(robot_xml_path, object_xml_path, None, object_name)

    env = mujoco_env.MuJoCoGraspingEnv(env_xml_path, object_name=object_name)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    if obs_shape is None or act_shape is None:
        msg = "Environment space shapes cannot be None"
        raise ValueError(msg)

    if observation_dim != obs_shape[0]:
        msg = (
            f"observation_dim ({observation_dim}) does not match environment observation dimension ({obs_shape[0]})"
        )
        raise ValueError(
            msg,
        )
    if action_dim != act_shape[0]:
        msg = f"action_dim ({action_dim}) does not match environment action dimension ({act_shape[0]})"
        raise ValueError(msg)

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
        verbose=VERBOSE,
    )

    total_timesteps = num_updates * n_steps
    sb3_model.learn(total_timesteps=total_timesteps)

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
