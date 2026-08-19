"""RL training pipeline for grasp policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
from grasping_ai.data.pointcloud_dataset import load_grasp_sample
from grasping_ai.models.rl_policy import (
    build_policy_network,
    build_sb3_net_arch,
    copy_sb3_policy_weights,
    save_rl_policy_checkpoint,
)
from grasping_ai.perception.geometry import invert_transform
from grasping_ai.robotics.gripper import panda_hand_to_contact_transform
from grasping_ai.robotics.kinematics import build_inverse_kinematics, load_robot_model, solve_inverse_kinematics
from grasping_ai.robotics.transforms import transform_grasp_pose
from grasping_ai.simulation import mujoco_env
from grasping_ai.simulation.scene import build_scene_xml
from grasping_ai.simulation.ycb import (
    find_ycb_mjcf,
    resolve_ycb_object_directory,
)
from grasping_ai.utils.path_validation import require_path

ROBOT_XML_PATH = Path(FLATTENED_YAML_CONFIG.get("robot.description", "assets/franka_emika_panda.xml"))
YCB_ROOT = Path(FLATTENED_YAML_CONFIG.get("paths.ycb_root", "data/raw/ycb"))
_configured_object_ids: list[Any] = FLATTENED_YAML_CONFIG.get("objects.ids", [])
OBJECT_IDS: tuple[str, ...] = tuple(str(item) for item in _configured_object_ids)
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
LOG_STD_INIT = float(FLATTENED_YAML_CONFIG.get("rl.log_std_init", -1.5))
ENTROPY_COEFFICIENT = float(FLATTENED_YAML_CONFIG.get("rl.entropy_coefficient", 0.0))
POLICY_NUM_LAYERS = int(FLATTENED_YAML_CONFIG.get("rl.policy_num_layers", 2))
VERBOSE = int(FLATTENED_YAML_CONFIG.get("rl.verbose", 1))
PREGRASP_DISTANCE = float(FLATTENED_YAML_CONFIG.get("rl.pregrasp_distance", 0.05))
PANDA_QPOS_SIZE = 9


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


def _load_validated_object_grasp(grasp_file: Path, object_id: str, grasp_index: int) -> np.ndarray:
    """Load one simulation-validated object-frame grasp candidate."""
    sample = load_grasp_sample(grasp_file)
    if sample.get("object_id") != object_id:
        raise ValueError(f"grasp archive object_id {sample.get('object_id')!r} does not match {object_id!r}")
    poses = np.asarray(sample["grasp_poses"], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not 0 <= grasp_index < len(poses):  # noqa: PLR2004
        raise ValueError("invalid grasp candidate index or pose array")
    if sample.get("grasp_pose_format", "object") != "object":
        raise ValueError("grasp-conditioned RL requires object-frame grasp poses")
    valid = sample.get("sim_validated")
    if valid is not None and not bool(np.asarray(valid)[grasp_index]):
        raise ValueError(f"candidate {grasp_index} is not simulation-validated")
    return poses[grasp_index]


def configure_grasp_conditioned_reset(  # noqa: PLR0913, PLR0917
    env: mujoco_env.MuJoCoGraspingEnv,
    robot_xml_path: Path,
    object_id: str,
    grasp_file: Path,
    grasp_index: int,
    pregrasp_distance: float,
) -> None:
    """Reset the robot above a valid candidate while leaving the object free.

    The candidate is object-relative. Its world pose is calculated only after
    MuJoCo has placed the object on the table, then IK targets a pose displaced
    along the candidate approach axis. No object coordinates are copied into
    the robot state or updated during rollout.
    """
    if pregrasp_distance <= 0.0:
        raise ValueError("pregrasp_distance must be positive")
    object_grasp = _load_validated_object_grasp(grasp_file, object_id, grasp_index)
    # The grasp candidates are generated and validated from the robot's
    # declared home pose.  MuJoCo otherwise resets an assembled scene to
    # zero qpos, which is a poor IK basin for the side grasps used here.
    robot = load_robot_model(str(robot_xml_path))
    model = robot["model"]
    if int(model.nkey) > 0:
        q_home = np.asarray(model.key_qpos[0, : model.nq], dtype=np.float64)
    else:
        q_home = np.zeros(model.nq, dtype=np.float64)
    env.set_reset_robot_configuration(q_home)
    env.reset()
    state = env._state  # noqa: SLF001 - read-only object pose after physical table placement
    object_world = mujoco_env.read_body_pose(state, object_id)
    contact_world = object_world @ object_grasp
    pregrasp_contact = np.array(contact_world, copy=True)
    pregrasp_contact[:3, 3] -= pregrasp_distance * contact_world[:3, 2]
    hand_target = transform_grasp_pose(pregrasp_contact, invert_transform(panda_hand_to_contact_transform()))

    q0 = np.asarray(state["data"].qpos[: robot["model"].nq], dtype=np.float64)
    ik = build_inverse_kinematics(robot, max_iterations=500, tolerance=1e-3)
    q_pregrasp = solve_inverse_kinematics(ik, hand_target, q0)
    if q_pregrasp.size >= PANDA_QPOS_SIZE:
        q_pregrasp[-2:] = 0.04
    env.set_reset_robot_configuration(q_pregrasp)


def run_rl_training_pipeline(  # noqa: C901, PLR0912, PLR0913, PLR0915, PLR0917  # end-to-end orchestration
    robot_xml_path: Path = ROBOT_XML_PATH,
    ycb_root: Path = YCB_ROOT,
    object_ids: tuple[str, ...] = OBJECT_IDS,
    policy_checkpoint_path: Path = POLICY_CHECKPOINT_PATH,
    experiment_log_dir: Path | None = None,
    table_xml_path: Path | None = None,
    grasp_file: Path | None = None,
    grasp_index: int = 0,
    **options: Any,  # noqa: ANN401
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
        table_xml_path: Optional workbench MJCF included in training scenes.
        grasp_file: Optional archive of simulation-validated object-frame
            candidates used to initialize each episode at a candidate-consistent
            pregrasp configuration.
        grasp_index: Candidate index selected from ``grasp_file``.
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
    log_std_init = float(options.pop("log_std_init", LOG_STD_INIT))
    entropy_coefficient = float(options.pop("entropy_coefficient", ENTROPY_COEFFICIENT))
    policy_num_layers = int(options.pop("policy_num_layers", POLICY_NUM_LAYERS))
    pregrasp_distance = float(options.pop("pregrasp_distance", PREGRASP_DISTANCE))
    if options:
        unexpected = ", ".join(sorted(options))
        msg = f"Unexpected RL training options: {unexpected}"
        raise TypeError(msg)

    object_ids_list = list(object_ids)
    _validate_rl_paths(robot_xml_path, ycb_root, object_ids_list)
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
        ("pregrasp_distance", pregrasp_distance),
    ):
        if value <= 0:
            msg = f"{name} must be positive"
            raise ValueError(msg)
    if not 0.0 <= gamma <= 1.0:
        msg = "gamma must be in [0, 1]"
        raise ValueError(msg)
    if not np.isfinite(log_std_init):
        raise ValueError("log_std_init must be finite")
    if entropy_coefficient < 0.0 or not np.isfinite(entropy_coefficient):
        raise ValueError("entropy_coefficient must be finite and non-negative")
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
        env_xml_path = build_scene_xml(robot_xml_path, object_xml_path, table_xml_path, object_name)

    env = mujoco_env.MuJoCoGraspingEnv(
        env_xml_path,
        object_name=object_name,
        place_object_on_table=table_xml_path is not None,
        control_mode=mujoco_env.CONTROL_MODE,
        task_observations=True,
    )
    if grasp_file is not None:
        if object_name is None:
            raise ValueError("grasp_file requires exactly one object_id")
        configure_grasp_conditioned_reset(
            env,
            robot_xml_path,
            object_name,
            grasp_file,
            grasp_index,
            pregrasp_distance,
        )

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
        "log_std_init": log_std_init,
    }

    sb3_model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=entropy_coefficient,
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
