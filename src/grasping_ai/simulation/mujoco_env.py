from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

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
