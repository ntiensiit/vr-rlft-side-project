"""Gymnasium-compatible MuJoCo grasp environment."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym
import mujoco  # type: ignore[import-untyped]
import numpy as np

from grasping_ai.config.flattened_yaml_config import (
    FLATTENED_YAML_CONFIG,
)
from grasping_ai.perception.geometry import make_transform
from grasping_ai.robotics.gripper import panda_fingertip_object_contacts
from grasping_ai.utils.path_validation import require_path

if TYPE_CHECKING:
    from pathlib import Path

REWARD_CLIP_MIN = float(FLATTENED_YAML_CONFIG.get("rl.reward.clip_min", -10.0))
REWARD_CLIP_MAX = float(FLATTENED_YAML_CONFIG.get("rl.reward.clip_max", 10.0))
ACTION_COST_WEIGHT = float(FLATTENED_YAML_CONFIG.get("rl.reward.action_cost_weight", 0.01))
SURVIVAL_BONUS = float(FLATTENED_YAML_CONFIG.get("rl.reward.survival_bonus", 1.0))
CONTACT_REWARD = float(FLATTENED_YAML_CONFIG.get("rl.reward.contact_reward", 0.0))
BILATERAL_CONTACT_REWARD = float(FLATTENED_YAML_CONFIG.get("rl.reward.bilateral_contact_reward", 0.0))
BILATERAL_HOLD_REWARD = float(FLATTENED_YAML_CONFIG.get("rl.reward.bilateral_hold_reward", 0.0))
CONTACT_LOSS_PENALTY = float(FLATTENED_YAML_CONFIG.get("rl.reward.contact_loss_penalty", 0.0))
DISTANCE_PROGRESS_WEIGHT = float(FLATTENED_YAML_CONFIG.get("rl.reward.distance_progress_weight", 0.0))
FINGERTIP_PROGRESS_WEIGHT = float(FLATTENED_YAML_CONFIG.get("rl.reward.fingertip_progress_weight", 0.0))
LIFT_REWARD_WEIGHT = float(FLATTENED_YAML_CONFIG.get("rl.reward.lift_reward_weight", 0.0))
GRASP_SUCCESS_BONUS = float(FLATTENED_YAML_CONFIG.get("rl.reward.grasp_success_bonus", 0.0))
LIFT_HEIGHT_THRESHOLD = float(FLATTENED_YAML_CONFIG.get("rl.reward.lift_height_threshold", 0.05))
DROP_HEIGHT_THRESHOLD = float(FLATTENED_YAML_CONFIG.get("rl.reward.drop_height_threshold", 0.1))
TERMINATE_ON_NON_FINITE = bool(FLATTENED_YAML_CONFIG.get("rl.reward.terminate_on_non_finite", True))
MAX_EPISODE_STEPS = int(FLATTENED_YAML_CONFIG.get("rl.max_episode_steps", 250))
CONTROL_MODE = str(FLATTENED_YAML_CONFIG.get("rl.control.mode", "normalized_delta"))
ARM_DELTA_SCALE = float(FLATTENED_YAML_CONFIG.get("rl.control.arm_delta_scale", 0.025))
GRIPPER_DELTA_SCALE = float(FLATTENED_YAML_CONFIG.get("rl.control.gripper_delta_scale", 25.5))
PANDA_ARM_DOF = 7
PANDA_QPOS_SIZE = 9
PANDA_FINGER_MAX = 0.04
PANDA_GRIPPER_CTRL_MAX = 255.0
TASK_OBSERVATION_SIZE = 8

SimulationStep = Callable[[float], None]
ContactReporter = Callable[[], list[dict[str, np.ndarray]]]


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
        """Advance the MuJoCo simulation by one time step.

        Args:
            dt: Simulation time step in seconds.

        Raises:
            TypeError: If ``dt`` is not a float or integer.
            ValueError: If ``dt`` is non-positive or non-finite.
        """
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
        """Report the contacts currently active in the MuJoCo data.

        Returns:
            A list of contact reports, each containing ``position``, ``normal``,
            ``force``, and the two involved ``body_names``.
        """
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
    """

    def __init__(
        self,
        robot_xml_path: Path,
        object_name: str | None = None,
        *,
        place_object_on_table: bool = False,
        control_mode: str = "direct",
        task_observations: bool = False,
    ) -> None:
        """Initialize the environment from a robot MJCF file.

        Args:
            robot_xml_path: Path to the robot MJCF XML description.
            object_name: Optional name of the object body to track for
                contact, lift, and drop rewards.
            place_object_on_table: Place a free object on the ``table_top``
                support at every reset. This preserves physical dynamics while
                preventing episodes from beginning with a floating object.
            control_mode: ``"direct"`` writes actions as raw actuator
                controls. ``"normalized_delta"`` interprets actions in
                ``[-1, 1]`` as incremental position commands, where zero
                holds the current robot pose.
            task_observations: Append grasp-relevant geometric features to
                raw MuJoCo state: hand-to-object translation, left/right
                fingertip distances, and object height gain.

        Raises:
            FileNotFoundError: If the robot XML file does not exist.
            ValueError: If the MuJoCo model has zero actuators.
        """
        super().__init__()

        require_path(robot_xml_path, "robot_xml_path")
        if object_name is not None and not isinstance(object_name, str):
            msg = "object_name must be a string or None"
            raise TypeError(msg)
        self._object_name = object_name
        self._place_object_on_table = place_object_on_table
        if control_mode not in {"direct", "normalized_delta"}:
            raise ValueError("control_mode must be 'direct' or 'normalized_delta'")
        self._control_mode = control_mode
        self._task_observations = task_observations
        self._control_target: np.ndarray | None = None
        self._initial_object_height: float | None = None
        self._previous_object_height: float | None = None
        self._previous_hand_object_distance: float | None = None
        self._previous_fingertip_object_distance: float | None = None
        self._reset_robot_qpos: np.ndarray | None = None
        self._grasp_success_granted = False
        self._had_object_contact = False
        self._had_bilateral_contact = False
        self._contact_loss_penalized = False
        self._elapsed_steps = 0

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

        obs_size = nq + nv + (TASK_OBSERVATION_SIZE if task_observations else 0)
        space_dtype = FLATTENED_YAML_CONFIG.numpy_dtype()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=space_dtype,
        )

        act_low = np.full(nu, -1.0, dtype=space_dtype)
        act_high = np.full(nu, 1.0, dtype=space_dtype)
        if control_mode == "direct" and hasattr(mj_model, "actuator_ctrlrange"):
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
        self._had_object_contact = False
        self._had_bilateral_contact = False
        self._contact_loss_penalized = False
        self._elapsed_steps = 0
        if self._object_name is not None and self._place_object_on_table:
            # Imported lazily because scene composition imports this module.
            from grasping_ai.simulation.scene import place_freejoint_body_on_surface  # noqa: PLC0415

            state_dict: dict[str, Any] = self._state  # type: ignore[assignment]
            place_freejoint_body_on_surface(state_dict["model"], state_dict["data"], self._object_name)
        if self._reset_robot_qpos is not None:
            self.set_robot_configuration(self._reset_robot_qpos)
        self._initialize_control_target()
        if self._object_name is not None:
            self._initial_object_height = self._get_object_height()
            self._previous_object_height = self._initial_object_height
            self._previous_hand_object_distance = self._hand_object_distance()
            self._previous_fingertip_object_distance = self._max_fingertip_object_distance()
        else:
            self._initial_object_height = None
            self._previous_object_height = None
            self._previous_hand_object_distance = None
            self._previous_fingertip_object_distance = None
        return self._get_observation(), {}

    def set_reset_robot_configuration(self, joint_positions: np.ndarray) -> None:
        """Configure robot qpos restored at the beginning of every episode.

        Only the robot prefix is stored. Object freejoint coordinates remain
        under MuJoCo's reset and contact dynamics, rather than being copied
        from a grasp candidate or artificially moved with the robot.

        Args:
            joint_positions: Robot joint positions with shape ``(N,)`` for
                ``1 <= N <= model.nq``.

        Raises:
            ValueError: If ``joint_positions`` has an invalid shape or contains
                non-finite values.
        """
        joint_positions = np.asarray(joint_positions, dtype=np.float64)
        state_dict: dict[str, Any] = self._state  # type: ignore[assignment]
        mj_model: Any = state_dict["model"]
        if joint_positions.ndim != 1 or joint_positions.size == 0 or joint_positions.size > mj_model.nq:
            msg = f"joint_positions must have shape (N,) for 1 <= N <= {mj_model.nq}"
            raise ValueError(msg)
        if not np.isfinite(joint_positions).all():
            raise ValueError("joint_positions must be finite")
        self._reset_robot_qpos = np.array(joint_positions, copy=True)

    def set_robot_configuration(self, joint_positions: np.ndarray) -> np.ndarray:
        """Set only the robot prefix of ``qpos`` without moving a free object.

        This supports pose-conditioned execution experiments.  In particular,
        the object freejoint remains owned by MuJoCo and is never teleported
        to manufacture a grasp.

        Args:
            joint_positions: Robot joint positions with shape ``(N,)`` for
                ``1 <= N <= model.nq``.

        Returns:
            The resulting observation vector.

        Raises:
            ValueError: If ``joint_positions`` has an invalid shape or contains
                non-finite values.
        """
        joint_positions = np.asarray(joint_positions, dtype=np.float64)
        state_dict: dict[str, Any] = self._state  # type: ignore[assignment]
        mj_model: Any = state_dict["model"]
        mj_data: Any = state_dict["data"]
        if joint_positions.ndim != 1 or joint_positions.size == 0 or joint_positions.size > mj_model.nq:
            msg = f"joint_positions must have shape (N,) for 1 <= N <= {mj_model.nq}"
            raise ValueError(msg)
        if not np.isfinite(joint_positions).all():
            raise ValueError("joint_positions must be finite")
        mj_data.qpos[: joint_positions.size] = joint_positions
        mj_data.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)
        self._initialize_control_target()
        return self._get_observation()

    def _initialize_control_target(self) -> None:
        """Initialize actuator targets from the current physical robot pose.

        Only meaningful in ``normalized_delta`` control mode; a no-op otherwise.
        The computed targets are stored and written to the actuators.
        """
        if self._control_mode != "normalized_delta":
            return
        state_dict: dict[str, Any] = self._state  # type: ignore[assignment]
        model: Any = state_dict["model"]
        data: Any = state_dict["data"]
        target = np.zeros(model.nu, dtype=np.float64)
        arm_count = min(PANDA_ARM_DOF, model.nu, data.qpos.size)
        target[:arm_count] = data.qpos[:arm_count]
        if model.nu > PANDA_ARM_DOF and data.qpos.size >= PANDA_QPOS_SIZE:
            max_width = 2.0 * PANDA_FINGER_MAX
            finger_width = float(np.clip(data.qpos[PANDA_ARM_DOF : PANDA_QPOS_SIZE].sum(), 0.0, max_width))
            target[PANDA_ARM_DOF] = PANDA_GRIPPER_CTRL_MAX * finger_width / max_width
        target = np.clip(target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        self._control_target = target
        set_actuator_controls(self._state, target)

    def _action_to_control(self, action: np.ndarray) -> np.ndarray:
        """Convert a normalized incremental action into actuator targets.

        Args:
            action: Normalized action in ``[-1, 1]`` whose scaled increments
                are applied to the current control target.

        Returns:
            The resulting actuator control vector.
        """
        if self._control_mode == "direct":
            return action
        if self._control_target is None:
            self._initialize_control_target()
        if self._control_target is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("normalized control target was not initialized")
        state_dict: dict[str, Any] = self._state  # type: ignore[assignment]
        model: Any = state_dict["model"]
        delta = np.asarray(action, dtype=np.float64)
        arm_count = min(PANDA_ARM_DOF, model.nu)
        self._control_target[:arm_count] += ARM_DELTA_SCALE * delta[:arm_count]
        if model.nu > PANDA_ARM_DOF:
            self._control_target[PANDA_ARM_DOF] += GRIPPER_DELTA_SCALE * delta[PANDA_ARM_DOF]
        self._control_target[:] = np.clip(
            self._control_target,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )
        return np.array(self._control_target, copy=True)

    def step(  # noqa: C901, PLR0912, PLR0915
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
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

        normalized_action = (
            np.clip(action, self.action_space.low, self.action_space.high)
            if self._control_mode == "normalized_delta"
            else action
        )
        control = self._action_to_control(normalized_action)
        set_actuator_controls(self._state, control)
        mujoco.mj_step(mj_model, mj_data)
        self._elapsed_steps += 1

        obs = self._get_observation()

        # Penalize policy motion, not absolute joint position targets.
        reward = ACTION_COST_WEIGHT * -float(np.sum(normalized_action**2))
        terminated = bool(TERMINATE_ON_NON_FINITE and not np.isfinite(obs).all())
        contact_count = 0
        bilateral_contact = False
        current_height = None
        height_gain = 0.0

        if np.isfinite(obs).all():
            reward += SURVIVAL_BONUS
            if self._object_name is not None and self._initial_object_height is not None:
                current_height = self._get_object_height()
                height_gain = current_height - self._initial_object_height

                contact_count, bilateral_contact = self._fingertip_object_contacts()
                has_object_contact = self._has_object_contact()
                if has_object_contact and not self._had_object_contact:
                    reward += CONTACT_REWARD
                    self._had_object_contact = True
                if bilateral_contact and not self._had_bilateral_contact:
                    reward += BILATERAL_CONTACT_REWARD
                    self._had_bilateral_contact = True
                elif bilateral_contact:
                    reward += BILATERAL_HOLD_REWARD
                elif self._had_bilateral_contact and not self._contact_loss_penalized:
                    reward -= CONTACT_LOSS_PENALTY
                    self._contact_loss_penalized = True
                distance = self._hand_object_distance()
                if self._previous_hand_object_distance is not None:
                    reward += DISTANCE_PROGRESS_WEIGHT * (self._previous_hand_object_distance - distance)
                self._previous_hand_object_distance = distance
                fingertip_distance = self._max_fingertip_object_distance()
                if self._previous_fingertip_object_distance is not None:
                    reward += FINGERTIP_PROGRESS_WEIGHT * (
                        self._previous_fingertip_object_distance - fingertip_distance
                    )
                self._previous_fingertip_object_distance = fingertip_distance
                # Reward signed *progress*, not the absolute height gain on
                # every step. Paying absolute gain makes hovering immediately
                # below the success threshold more profitable than completing
                # the task and terminating the episode.
                height_progress = 0.0
                if self._previous_object_height is not None:
                    height_progress = current_height - self._previous_object_height
                self._previous_object_height = current_height
                # Height alone is not a grasp. The object can rise by bouncing
                # or being pushed, so lift progress only counts under a valid
                # bilateral fingertip grasp.
                if bilateral_contact and LIFT_REWARD_WEIGHT > 0.0:
                    reward += LIFT_REWARD_WEIGHT * height_progress
                if (
                    bilateral_contact
                    and
                    GRASP_SUCCESS_BONUS > 0.0
                    and not self._grasp_success_granted
                    and height_gain > LIFT_HEIGHT_THRESHOLD
                ):
                    reward += GRASP_SUCCESS_BONUS
                    self._grasp_success_granted = True
                    terminated = True
                if height_gain < -DROP_HEIGHT_THRESHOLD:
                    terminated = True

        reward = float(np.clip(reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX))
        truncated = self._elapsed_steps >= MAX_EPISODE_STEPS

        return obs, reward, terminated, truncated, {
            "fingertip_contact_count": int(contact_count),
            "bilateral_contact": bool(bilateral_contact),
            "object_height": None if current_height is None else float(current_height),
            "height_gain": float(height_gain),
            "actuator_control": np.array(control, copy=True),
        }

    def _get_object_height(self) -> float:
        """Read the current height of the tracked object body.

        Returns:
            The world-frame z-coordinate of the object body.
        """
        pose = read_body_pose(self._state, cast("str", self._object_name))
        return float(pose[2, 3])

    def _has_object_contact(self) -> bool:
        """Report whether a Panda fingertip touches the tracked object.

        Returns:
            ``True`` only for gripper-object contact. Object-table support
            contact intentionally does not count as grasp progress.
        """
        contact_count, _bilateral = self._fingertip_object_contacts()
        return contact_count > 0.0

    def _fingertip_object_contacts(self) -> tuple[float, bool]:
        """Return opposed Panda fingertip contact state for the tracked object.

        Returns:
            A tuple ``(contact_count, bilateral_contact)`` where
            ``contact_count`` is the number of fingertips in contact and
            ``bilateral_contact`` indicates opposed fingertip contact. Both are
            zero/false when no object is tracked.
        """
        if self._object_name is None:
            return 0.0, False
        state_dict: dict[str, Any] = self._state  # type: ignore[assignment]
        return panda_fingertip_object_contacts(state_dict["model"], state_dict["data"], self._object_name)

    def _hand_object_distance(self) -> float:
        """Return the world-frame distance from Panda hand to object center.

        Returns:
            The Euclidean distance, or ``0.0`` when no object is tracked.
        """
        if self._object_name is None:
            return 0.0
        hand_pose = read_body_pose(self._state, "hand")
        object_pose = read_body_pose(self._state, self._object_name)
        return float(np.linalg.norm(hand_pose[:3, 3] - object_pose[:3, 3]))

    def _fingertip_object_distances(self) -> tuple[float, float]:
        """Return center distances from each Panda finger to the object.

        Returns:
            A tuple of ``(left, right)`` finger-to-object Euclidean distances,
            or ``(0.0, 0.0)`` when no object is tracked.
        """
        if self._object_name is None:
            return 0.0, 0.0
        object_position = read_body_pose(self._state, self._object_name)[:3, 3]
        left_position = read_body_pose(self._state, "left_finger")[:3, 3]
        right_position = read_body_pose(self._state, "right_finger")[:3, 3]
        return (
            float(np.linalg.norm(left_position - object_position)),
            float(np.linalg.norm(right_position - object_position)),
        )

    def _max_fingertip_object_distance(self) -> float:
        """Return the worse of the two finger-to-object distances.

        Returns:
            The larger of the left/right finger-to-object distances.
        """
        return max(self._fingertip_object_distances())

    def _task_observation(self) -> np.ndarray:
        """Build compact geometric features for candidate-conditioned RL.

        Returns:
            A float32 feature vector of shape ``(TASK_OBSERVATION_SIZE,)``,
            or zeros when no object is tracked.
        """
        if self._object_name is None:
            return np.zeros(TASK_OBSERVATION_SIZE, dtype=np.float32)
        hand_position = read_body_pose(self._state, "hand")[:3, 3]
        object_position = read_body_pose(self._state, self._object_name)[:3, 3]
        left_distance, right_distance = self._fingertip_object_distances()
        contact_count, bilateral_contact = self._fingertip_object_contacts()
        height_gain = 0.0 if self._initial_object_height is None else object_position[2] - self._initial_object_height
        return np.asarray(
            [
                *(object_position - hand_position),
                left_distance,
                right_distance,
                height_gain,
                contact_count,
                float(bilateral_contact),
            ],
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray:
        """Read and concatenate qpos and qvel into a float32 observation.

        Returns:
            Observation vector of shape ``(nq + nv,)``.
        """
        state_dict: dict[str, Any] = self._state  # type: ignore[assignment]
        mj_data: Any = state_dict["data"]
        qpos = np.array(mj_data.qpos, copy=True)
        qvel = np.array(mj_data.qvel, copy=True)
        observation = np.concatenate([qpos, qvel]).astype(np.float32)
        if self._task_observations:
            observation = np.concatenate([observation, self._task_observation()])
        return observation
