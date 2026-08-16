"""RL training pipeline for grasp policies."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class _RLHyperparameters:
    """Scalar hyperparameters for PPO training and the exported policy."""

    observation_dim: int
    action_dim: int
    hidden_dim: int
    learning_rate: float
    num_updates: int
    gamma: float
    n_steps: int
    batch_size: int
    n_epochs: int
    policy_num_layers: int


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


def _validate_rl_hyperparameters(hp: _RLHyperparameters) -> None:
    """Validate scalar PPO/policy hyperparameters."""
    positive_scalars: list[tuple[str, float]] = [
        ("observation_dim", hp.observation_dim),
        ("action_dim", hp.action_dim),
        ("hidden_dim", hp.hidden_dim),
        ("learning_rate", hp.learning_rate),
        ("num_updates", hp.num_updates),
    ]
    for name, value in positive_scalars:
        if value <= 0:
            msg = f"{name} must be positive"
            raise ValueError(msg)
    if not 0.0 <= hp.gamma <= 1.0:
        msg = "gamma must be in [0, 1]"
        raise ValueError(msg)
    ppo_scalars: list[tuple[str, int]] = [
        ("n_steps", hp.n_steps),
        ("batch_size", hp.batch_size),
        ("n_epochs", hp.n_epochs),
        ("policy_num_layers", hp.policy_num_layers),
    ]
    for name, value in ppo_scalars:
        if value <= 0:
            msg = f"{name} must be positive"
            raise ValueError(msg)


def run_rl_training_pipeline(  # noqa: PLR0913, PLR0917  # public pipeline API; tests call it positionally
    robot_xml_path: Path = ROBOT_XML_PATH,
    ycb_root: Path = YCB_ROOT,
    object_ids: tuple[str, ...] = OBJECT_IDS,
    policy_checkpoint_path: Path = POLICY_CHECKPOINT_PATH,
    observation_dim: int = OBSERVATION_DIM,
    action_dim: int = ACTION_DIM,
    hidden_dim: int = HIDDEN_DIM,
    learning_rate: float = LEARNING_RATE,
    num_updates: int = NUM_UPDATES,
    gamma: float = GAMMA,
    device: str = DEVICE,
    seed: int | None = SEED,
    experiment_log_dir: Path | None = None,
    n_steps: int = N_STEPS,
    batch_size: int = BATCH_SIZE,
    n_epochs: int = N_EPOCHS,
    policy_num_layers: int = POLICY_NUM_LAYERS,
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
    object_ids = list(object_ids)
    hp = _RLHyperparameters(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        num_updates=num_updates,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        policy_num_layers=policy_num_layers,
    )
    _validate_rl_paths(robot_xml_path, ycb_root, object_ids)
    _validate_rl_hyperparameters(hp)
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

    try:
        reward_config = mujoco_env.RewardConfig.load_from_config()
    except Exception:  # noqa: BLE001  # any config-load failure falls back to default rewards
        reward_config = mujoco_env.RewardConfig()
    env = mujoco_env.MuJoCoGraspingEnv(env_xml_path, object_name=object_name, reward_config=reward_config)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    if obs_shape is None or act_shape is None:
        msg = "Environment space shapes cannot be None"
        raise ValueError(msg)

    if hp.observation_dim != obs_shape[0]:
        msg = (
            f"observation_dim ({hp.observation_dim}) does not match environment observation dimension ({obs_shape[0]})"
        )
        raise ValueError(
            msg,
        )
    if hp.action_dim != act_shape[0]:
        msg = f"action_dim ({hp.action_dim}) does not match environment action dimension ({act_shape[0]})"
        raise ValueError(msg)

    policy_kwargs = {
        "net_arch": build_sb3_net_arch(hp.hidden_dim, hp.policy_num_layers),
        "activation_fn": torch.nn.Tanh,
    }

    sb3_model = PPO(
        "MlpPolicy",
        env,
        learning_rate=hp.learning_rate,
        n_steps=hp.n_steps,
        batch_size=hp.batch_size,
        n_epochs=hp.n_epochs,
        gamma=hp.gamma,
        device=device,
        seed=seed,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(experiment_log_dir) if experiment_log_dir else None,
        verbose=VERBOSE,
    )

    total_timesteps = hp.num_updates * hp.n_steps
    sb3_model.learn(total_timesteps=total_timesteps)

    legacy_policy = build_policy_network(hp.observation_dim, hp.action_dim, hp.hidden_dim, hp.policy_num_layers)
    copy_sb3_policy_weights(sb3_model.policy, legacy_policy)

    save_rl_policy_checkpoint(
        policy=legacy_policy,
        policy_checkpoint_path=policy_checkpoint_path,
        epoch=hp.num_updates,
        observation_dim=hp.observation_dim,
        action_dim=hp.action_dim,
        hidden_dim=hp.hidden_dim,
        num_layers=hp.policy_num_layers,
        seed=seed,
    )
