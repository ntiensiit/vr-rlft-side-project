"""Gymnasium-compatible MuJoCo grasp environment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym
import mujoco  # type: ignore[import-untyped]
import numpy as np

from grasping_ai.config.flattened_yaml_config import (
    FLATTENED_YAML_CONFIG,
    FlattenedYAMLConfig,
)
from grasping_ai.perception.geometry import make_transform
from grasping_ai.utils.path_validation import require_path

if TYPE_CHECKING:
    from pathlib import Path

    from omegaconf import DictConfig

SimulationStep = Callable[[float], None]
ContactReporter = Callable[[], list[dict[str, np.ndarray]]]


@dataclass(frozen=True)
class RewardConfig:
    """Reward scaling and terminal-condition parameters for the RL environment.

    The reward is composed of task-terms whose weights are all configurable:

    * a negative quadratic action cost scaled by ``action_cost_weight``;
    * a ``survival_bonus`` granted while the observation stays finite;
    * a per-step ``contact_reward`` when the tracked object is in contact;
    * a ``lift_reward_weight``-scaled positive term for object height gain;
    * a one-time ``grasp_success_bonus`` when the object is first lifted past
      ``lift_height_threshold``.

    The episode terminates on non-finite observations (when
    ``terminate_on_non_finite``) or when the object drops more than
    ``drop_height_threshold`` below its height at reset.
    """

    action_cost_weight: float = 0.01
    survival_bonus: float = 1.0
    contact_reward: float = 0.0
    lift_reward_weight: float = 0.0
    grasp_success_bonus: float = 0.0
    lift_height_threshold: float = 0.05
    drop_height_threshold: float = 0.1
    terminate_on_non_finite: bool = True

    @classmethod
    def load_from_config(cls, cfg: DictConfig | None = None) -> RewardConfig:
        """Load RewardConfig parameters from a composed Hydra config.

        Args:
            cfg: The project configuration mapping. When ``None``, the global
                flattened YAML config is used.

        Returns:
            A RewardConfig instance populated with configured parameters.
        """
        resolved_cfg = FLATTENED_YAML_CONFIG.cfg if cfg is None else cfg
        yaml_config = FlattenedYAMLConfig(resolved_cfg)
        return cls(
            action_cost_weight=yaml_config.value(
                "rl",
                "reward",
                "action_cost_weight",
                value_type=float,
                default=0.01,
            ),
            survival_bonus=yaml_config.value(
                "rl",
                "reward",
                "survival_bonus",
                value_type=float,
                default=1.0,
            ),
            contact_reward=yaml_config.value(
                "rl",
                "reward",
                "contact_reward",
                value_type=float,
                default=0.0,
            ),
            lift_reward_weight=yaml_config.value(
                "rl",
                "reward",
                "lift_reward_weight",
                value_type=float,
                default=0.0,
            ),
            grasp_success_bonus=yaml_config.value(
                "rl",
                "reward",
                "grasp_success_bonus",
                value_type=float,
                default=0.0,
            ),
            lift_height_threshold=yaml_config.value(
                "rl",
                "reward",
                "lift_height_threshold",
                value_type=float,
                default=0.05,
            ),
            drop_height_threshold=yaml_config.value(
                "rl",
                "reward",
                "drop_height_threshold",
                value_type=float,
                default=0.1,
            ),
            terminate_on_non_finite=yaml_config.value(
                "rl",
                "reward",
                "terminate_on_non_finite",
                value_type=bool,
                default=True,
            ),
        )


def load_mujoco_model(model_xml_path: Path) -> object:
    """Load a MuJoCo simulation model from an XML file.

    Args:
        model_xml_path: Path to the MuJoCo MJCF XML description.

    Returns:
        An opaque simulation model object usable by other simulation helpers.
    """
    require_path(model_xml_path, "model_xml_path")
    if not model_xml_path.is_file():
        msg = f"Model XML file not found at: {model_xml_path}"
        raise FileNotFoundError(msg)

    try:
        mj_model = mujoco.MjModel.from_xml_path(str(model_xml_path))
    except Exception as e:
        msg = f"Failed to load MuJoCo model from XML: {e}"
        raise ValueError(msg) from e

    return {
        "mj_model": mj_model,
        "xml_path": model_xml_path,
    }


def create_simulation(model: object) -> tuple[object, SimulationStep, ContactReporter]:
    """Create a stepping interface over a MuJoCo model.

    Args:
        model: MuJoCo model returned by ``load_mujoco_model``.

    Returns:
        A tuple ``(state, step, contacts)`` where ``state`` is an opaque state
        handle, ``step`` advances the simulation by a time increment, and
        ``contacts`` reports the current contact information.
    """
    if isinstance(model, dict) and "mj_model" in model:
        mj_model = model["mj_model"]
        model_xml_path = model.get("xml_path")
    else:
        mj_model = model
        model_xml_path = None

    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    state: dict[str, Any] = {
        "model": mj_model,
        "data": mj_data,
        "model_xml_path": model_xml_path,
        "attached_xml_paths": [],
    }

    def step(dt: float) -> None:
        if not isinstance(dt, (int, float, np.floating, np.integer)):
            msg = "dt must be a float or integer"
            raise TypeError(msg)
        if dt <= 0:
            msg = "dt must be positive"
            raise ValueError(msg)
        if not np.isfinite(dt):
            msg = "dt must be a finite number"
            raise ValueError(msg)

        current_model: Any = state["model"]
        current_data: Any = state["data"]
        current_model.opt.timestep = dt
        mujoco.mj_step(current_model, current_data)

    def contacts() -> list[dict[str, np.ndarray]]:
        current_model: Any = state["model"]
        current_data: Any = state["data"]
        reports = []
        for i in range(current_data.ncon):
            c = current_data.contact[i]
            body1_id = current_model.geom_bodyid[c.geom1]
            body2_id = current_model.geom_bodyid[c.geom2]
            body1_name = current_model.body(body1_id).name
            body2_name = current_model.body(body2_id).name

            force = np.zeros(6)
            mujoco.mj_contactForce(current_model, current_data, i, force)

            reports.append(
                {
                    "position": np.array(c.pos, copy=True),
                    "normal": np.array(c.frame[:3], copy=True),
                    "force": force,
                    "body_names": np.array([body1_name, body2_name], dtype=object),
                },
            )
        return reports

    return state, step, contacts


def reset_simulation(state: object) -> None:
    """Reset the simulation state to its initial configuration.

    Args:
        state: Opaque state handle returned by ``create_simulation``.
    """
    if not isinstance(state, dict) or "model" not in state or "data" not in state:
        msg = "state must be a simulation state dictionary"
        raise TypeError(msg)

    state_dict = cast("dict[str, Any]", state)
    mujoco.mj_resetData(state_dict["model"], state_dict["data"])
    mujoco.mj_forward(state_dict["model"], state_dict["data"])


def set_actuator_controls(state: object, ctrl: np.ndarray) -> None:
    """Write actuator controls into the simulation state.

    This helper is the single authoritative command path for gripper and
    robot actuation: every pipeline (the Gymnasium environment, grasp
    simulation, and the gripper controller) routes control writes through
    here rather than writing ``mj_data.ctrl`` directly.

    Args:
        state: Opaque state handle returned by ``create_simulation``.
        ctrl: Actuator control vector with shape ``(num_actuators,)``.

    Raises:
        TypeError: If ``state`` or ``ctrl`` have incorrect types.
        ValueError: If ``ctrl`` contains non-finite values or has the wrong
            shape.
    """
    if not isinstance(state, dict) or "model" not in state or "data" not in state:
        msg = "state must be a simulation state dictionary"
        raise TypeError(msg)
    if not isinstance(ctrl, np.ndarray):
        msg = "ctrl must be a numpy array"
        raise TypeError(msg)
    if not np.isfinite(ctrl).all():
        msg = "ctrl must contain only finite values"
        raise ValueError(msg)

    state_dict = cast("dict[str, Any]", state)
    model: Any = state_dict["model"]
    data: Any = state_dict["data"]

    nu: int = model.nu
    if ctrl.shape != (nu,):
        msg = f"ctrl shape {ctrl.shape} does not match model.nu ({nu})"
        raise ValueError(msg)

    data.ctrl[:] = ctrl


def read_joint_positions(state: object) -> np.ndarray:
    """Read the current joint positions from the simulation state.

    Args:
        state: Opaque state handle returned by ``create_simulation``.

    Returns:
        Joint position vector with shape ``(num_joints,)``.
    """
    if not isinstance(state, dict) or "model" not in state or "data" not in state:
        msg = "state must be a simulation state dictionary"
        raise TypeError(msg)

    state_dict = cast("dict[str, Any]", state)
    return np.array(state_dict["data"].qpos, copy=True)


def set_joint_positions(state: object, positions: np.ndarray) -> None:
    """Write joint positions into the simulation state.

    Args:
        state: Opaque state handle returned by ``create_simulation``.
        positions: Joint position vector with shape ``(num_joints,)``.
    """
    if not isinstance(state, dict) or "model" not in state or "data" not in state:
        msg = "state must be a simulation state dictionary"
        raise TypeError(msg)
    if not isinstance(positions, np.ndarray):
        msg = "positions must be a numpy array"
        raise TypeError(msg)
    if not np.isfinite(positions).all():
        msg = "positions must contain only finite values"
        raise ValueError(msg)

    state_dict = cast("dict[str, Any]", state)
    model: Any = state_dict["model"]
    data: Any = state_dict["data"]

    if positions.shape != (model.nq,):
        msg = f"positions shape {positions.shape} does not match model.nq ({model.nq})"
        raise ValueError(msg)

    data.qpos[:] = positions
    mujoco.mj_forward(model, data)


def read_body_pose(state: object, body_name: str) -> np.ndarray:
    """Read the world-frame pose of a named body.

    Args:
        state: Opaque state handle returned by ``create_simulation``.
        body_name: Name of the body whose pose should be read.

    Returns:
        A ``(4, 4)`` transformation matrix representing the body pose.
    """
    if not isinstance(state, dict) or "model" not in state or "data" not in state:
        msg = "state must be a simulation state dictionary"
        raise TypeError(msg)
    if not isinstance(body_name, str):
        msg = "body_name must be a string"
        raise TypeError(msg)

    state_dict = cast("dict[str, Any]", state)
    model: Any = state_dict["model"]
    data: Any = state_dict["data"]

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        msg = f"Body '{body_name}' not found in simulation model"
        raise ValueError(msg)

    return make_transform(data.xmat[body_id].reshape(3, 3), data.xpos[body_id])


class MuJoCoGraspingEnv(gym.Env):
    """Gymnasium-compatible MuJoCo environment for RL policy training.

    Wraps the existing functional MuJoCo simulation primitives
    (``load_mujoco_model``, ``create_simulation``, ``reset_simulation``)
    into a standardized Gymnasium environment with explicit observation
    and action spaces.

    Observations are the concatenation of MuJoCo ``qpos`` and ``qvel``
    vectors. Actions map directly to MuJoCo actuator controls. The reward
    preserves the legacy behavior: negative quadratic action cost plus a
    survival bonus when the observation is finite, clipped to a finite range.

    Args:
        robot_xml_path: Path to the robot MJCF XML description.
        object_name: Optional name of the object body to track for contact,
            lift, and drop rewards. When ``None`` those reward terms are
            disabled.
        reward_config: Reward scaling and terminal-condition parameters. When
            ``None`` the legacy reward behavior is preserved exactly.
    """

    def __init__(
        self,
        robot_xml_path: Path,
        object_name: str | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        """Initialize the environment from a robot MJCF file.

        Args:
            robot_xml_path: Path to the robot MJCF XML description.
            object_name: Optional name of the object body to track for
                contact, lift, and drop rewards.
            reward_config: Reward scaling and terminal-condition parameters.

        Raises:
            FileNotFoundError: If the robot XML file does not exist.
            ValueError: If the MuJoCo model has zero actuators.
        """
        super().__init__()

        require_path(robot_xml_path, "robot_xml_path")
        if object_name is not None and not isinstance(object_name, str):
            msg = "object_name must be a string or None"
            raise TypeError(msg)
        if reward_config is not None and not isinstance(reward_config, RewardConfig):
            msg = "reward_config must be a RewardConfig or None"
            raise TypeError(msg)

        self._object_name = object_name
        if reward_config is None:
            try:
                reward_config = RewardConfig.load_from_config()
            # Deliberate fallback: any config-loading failure uses default rewards.
            except Exception:  # noqa: BLE001
                reward_config = RewardConfig()
        self._reward_config = reward_config
        self._initial_object_height: float | None = None
        self._grasp_success_granted = False

        model_handle = load_mujoco_model(robot_xml_path)
        self._state, self._step_fn, self._contacts_fn = create_simulation(model_handle)

        state_dict: dict[str, Any] = self._state  # type: ignore[assignment]
        mj_model: Any = state_dict["model"]

        nq: int = mj_model.nq
        nv: int = mj_model.nv
        nu: int = mj_model.nu

        if nu == 0:
            msg = "MuJoCo model has zero actuators; the RL environment requires a non-empty action space"
            raise ValueError(msg)

        obs_size = nq + nv
        space_dtype = FLATTENED_YAML_CONFIG.numpy_dtype()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=space_dtype,
        )

        act_low = np.full(nu, -1.0, dtype=space_dtype)
        act_high = np.full(nu, 1.0, dtype=space_dtype)
        if hasattr(mj_model, "actuator_ctrlrange"):
            ctrl_range = np.array(mj_model.actuator_ctrlrange, copy=True)
            if ctrl_range.shape == (nu, 2):
                for i in range(nu):
                    lo, hi = ctrl_range[i]
                    if np.isfinite(lo) and np.isfinite(hi):
                        act_low[i] = float(lo)
                        act_high[i] = float(hi)

        self.action_space = gym.spaces.Box(
            low=act_low,
            high=act_high,
            shape=(nu,),
            dtype=space_dtype,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the simulation to its initial state.

        Args:
            seed: Optional random seed for Gymnasium seeding.
            options: Optional configuration dictionary (unused).

        Returns:
            A tuple ``(observation, info)`` where ``observation`` is the
            initial state vector and ``info`` is an empty dictionary.
        """
        super().reset(seed=seed, options=options)
        reset_simulation(self._state)
        self._grasp_success_granted = False
        if self._object_name is not None:
            self._initial_object_height = self._get_object_height()
        else:
            self._initial_object_height = None
        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the simulation by one step.

        Args:
            action: Action vector applied to MuJoCo actuator controls.

        Returns:
            A Gymnasium step tuple ``(observation, reward, terminated,
            truncated, info)``.

        Raises:
            ValueError: If the action contains non-finite values.
        """
        action = np.asarray(action, dtype=np.float32).flatten()

        if not np.isfinite(action).all():
            msg = "Action must contain only finite values"
            raise ValueError(msg)

        state_dict: dict[str, Any] = self._state  # type: ignore[assignment]
        mj_model: Any = state_dict["model"]
        mj_data: Any = state_dict["data"]
        nu: int = mj_model.nu

        if action.shape[0] > nu:
            action = action[:nu]
        elif action.shape[0] < nu:
            padded = np.zeros(nu, dtype=np.float32)
            padded[: action.shape[0]] = action
            action = padded

        set_actuator_controls(self._state, action)
        mujoco.mj_step(mj_model, mj_data)

        obs = self._get_observation()

        config = self._reward_config
        reward = config.action_cost_weight * -float(np.sum(action**2))
        terminated = bool(config.terminate_on_non_finite and not np.isfinite(obs).all())

        if np.isfinite(obs).all():
            reward += config.survival_bonus
            if self._object_name is not None and self._initial_object_height is not None:
                current_height = self._get_object_height()
                height_gain = current_height - self._initial_object_height

                if self._has_object_contact():
                    reward += config.contact_reward
                if config.lift_reward_weight > 0.0 and height_gain > 0.0:
                    reward += config.lift_reward_weight * height_gain
                if (
                    config.grasp_success_bonus > 0.0
                    and not self._grasp_success_granted
                    and height_gain > config.lift_height_threshold
                ):
                    reward += config.grasp_success_bonus
                    self._grasp_success_granted = True
                if height_gain < -config.drop_height_threshold:
                    terminated = True

        reward = float(np.clip(reward, -10.0, 10.0))
        truncated = False

        return obs, reward, terminated, truncated, {}

    def _get_object_height(self) -> float:
        """Read the current height of the tracked object body.

        Returns:
            The world-frame z-coordinate of the object body.
        """
        pose = read_body_pose(self._state, cast("str", self._object_name))
        return float(pose[2, 3])

    def _has_object_contact(self) -> bool:
        """Report whether the tracked object is currently in contact.

        Returns:
            ``True`` if any contact report involves the object body.
        """
        reports = self._contacts_fn()
        return any(self._object_name in set(contact["body_names"]) for contact in reports)

    def _get_observation(self) -> np.ndarray:
        """Read and concatenate qpos and qvel into a float32 observation.

        Returns:
            Observation vector of shape ``(nq + nv,)``.
        """
        state_dict: dict[str, Any] = self._state  # type: ignore[assignment]
        mj_data: Any = state_dict["data"]
        qpos = np.array(mj_data.qpos, copy=True)
        qvel = np.array(mj_data.qvel, copy=True)
        return np.concatenate([qpos, qvel]).astype(np.float32)
