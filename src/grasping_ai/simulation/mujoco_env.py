from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import mujoco  # type: ignore[import-untyped]
import numpy as np

SimulationStep = Callable[[float], None]
ContactReporter = Callable[[], list[dict[str, np.ndarray]]]


def load_mujoco_model(model_xml_path: Path) -> object:
    """Load a MuJoCo simulation model from an XML file.

    Args:
        model_xml_path: Path to the MuJoCo MJCF XML description.

    Returns:
        An opaque simulation model object usable by other simulation helpers.
    """
    if not isinstance(model_xml_path, Path):
        raise TypeError("model_xml_path must be a pathlib.Path instance")
    if not model_xml_path.is_file():
        raise FileNotFoundError(f"Model XML file not found at: {model_xml_path}")

    try:
        mj_model = mujoco.MjModel.from_xml_path(str(model_xml_path))
    except Exception as e:
        raise ValueError(f"Failed to load MuJoCo model from XML: {e}") from e

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
            raise TypeError("dt must be a float or integer")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if not np.isfinite(dt):
            raise ValueError("dt must be a finite number")

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

            reports.append({
                "position": np.array(c.pos, copy=True),
                "normal": np.array(c.frame[:3], copy=True),
                "force": force,
                "body_names": np.array([body1_name, body2_name], dtype=object)
            })
        return reports

    return state, step, contacts


def reset_simulation(state: object) -> None:
    """Reset the simulation state to its initial configuration.

    Args:
        state: Opaque state handle returned by ``create_simulation``.
    """
    if not isinstance(state, dict) or "model" not in state or "data" not in state:
        raise TypeError("state must be a simulation state dictionary")

    state_dict = cast(dict[str, Any], state)
    mujoco.mj_resetData(state_dict["model"], state_dict["data"])
    mujoco.mj_forward(state_dict["model"], state_dict["data"])


def read_joint_positions(state: object) -> np.ndarray:
    """Read the current joint positions from the simulation state.

    Args:
        state: Opaque state handle returned by ``create_simulation``.

    Returns:
        Joint position vector with shape ``(num_joints,)``.
    """
    if not isinstance(state, dict) or "model" not in state or "data" not in state:
        raise TypeError("state must be a simulation state dictionary")

    state_dict = cast(dict[str, Any], state)
    return np.array(state_dict["data"].qpos, copy=True)


def set_joint_positions(state: object, positions: np.ndarray) -> None:
    """Write joint positions into the simulation state.

    Args:
        state: Opaque state handle returned by ``create_simulation``.
        positions: Joint position vector with shape ``(num_joints,)``.
    """
    if not isinstance(state, dict) or "model" not in state or "data" not in state:
        raise TypeError("state must be a simulation state dictionary")
    if not isinstance(positions, np.ndarray):
        raise TypeError("positions must be a numpy array")
    if not np.isfinite(positions).all():
        raise ValueError("positions must contain only finite values")

    state_dict = cast(dict[str, Any], state)
    model: Any = state_dict["model"]
    data: Any = state_dict["data"]

    if positions.shape != (model.nq,):
        raise ValueError(
            f"positions shape {positions.shape} does not match model.nq ({model.nq})"
        )

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
        raise TypeError("state must be a simulation state dictionary")
    if not isinstance(body_name, str):
        raise TypeError("body_name must be a string")

    state_dict = cast(dict[str, Any], state)
    model: Any = state_dict["model"]
    data: Any = state_dict["data"]

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body '{body_name}' not found in simulation model")

    pose = np.eye(4)
    pose[:3, :3] = data.xmat[body_id].reshape(3, 3)
    pose[:3, 3] = data.xpos[body_id]
    return pose


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
    """

    def __init__(self, robot_xml_path: Path) -> None:
        """Initialize the environment from a robot MJCF file.

        Args:
            robot_xml_path: Path to the robot MJCF XML description.

        Raises:
            FileNotFoundError: If the robot XML file does not exist.
            ValueError: If the MuJoCo model has zero actuators.
        """
        super().__init__()

        model_handle = load_mujoco_model(robot_xml_path)
        self._state, self._step_fn, self._contacts_fn = create_simulation(
            model_handle
        )

        state_dict: dict[str, Any] = self._state  # type: ignore[assignment]
        mj_model: Any = state_dict["model"]

        nq: int = mj_model.nq
        nv: int = mj_model.nv
        nu: int = mj_model.nu

        if nu == 0:
            raise ValueError(
                "MuJoCo model has zero actuators; "
                "the RL environment requires a non-empty action space"
            )

        obs_size = nq + nv
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

        act_low = np.full(nu, -1.0, dtype=np.float32)
        act_high = np.full(nu, 1.0, dtype=np.float32)
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
            dtype=np.float32,
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
        return self._get_observation(), {}

    def step(
        self, action: np.ndarray
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
            raise ValueError("Action must contain only finite values")

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

        mj_data.ctrl[:] = action
        mujoco.mj_step(mj_model, mj_data)

        obs = self._get_observation()

        reward = -float(np.sum(action ** 2)) * 0.01
        if np.isfinite(obs).all():
            reward += 1.0
        reward = float(np.clip(reward, -10.0, 10.0))

        terminated = not np.isfinite(obs).all()
        truncated = False

        if terminated:
            reset_simulation(self._state)
            obs = self._get_observation()

        return obs, reward, terminated, truncated, {}

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

