from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco  # type: ignore[import-untyped]
import numpy as np

GripperCommand = np.ndarray


def load_gripper_model(gripper_description_path: str) -> dict[str, object]:
    """Load a gripper description from disk.

    Args:
        gripper_description_path: Path to a gripper description file such as an
            XML/MJCF gripper definition.

    Returns:
        A dictionary describing the gripper kinematic and dynamic parameters.
    """
    if not isinstance(gripper_description_path, str):
        raise TypeError("gripper_description_path must be a string")
    if not gripper_description_path:
        raise ValueError("gripper_description_path must not be empty")

    path = Path(gripper_description_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Gripper description file '{gripper_description_path}' not found"
        )

    try:
        model = mujoco.MjModel.from_xml_path(str(path))
    except Exception as e:
        raise ValueError(f"Failed to load MuJoCo gripper model: {e}") from e

    actuator_names = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        actuator_names.append(name or f"actuator_{i}")

    return {
        "model": model,
        "path": path,
        "actuator_names": actuator_names,
        "nu": model.nu,
        "nq": model.nq,
    }


def build_gripper_controller(gripper_model: dict[str, object]) -> Callable[[GripperCommand], None]:
    """Build a callable gripper controller that issues open/close commands.

    Args:
        gripper_model: Gripper model returned by ``load_gripper_model``.

    Returns:
        A callable accepting a gripper command vector and applying it to the
        underlying simulation environment.
    """
    if not isinstance(gripper_model, dict) or "model" not in gripper_model:
        raise TypeError("gripper_model must be a dictionary returned by load_gripper_model")

    def controller(command: GripperCommand) -> None:
        if not isinstance(command, np.ndarray):
            raise TypeError("command must be a numpy array")
        if not np.isfinite(command).all():
            raise ValueError("command must contain only finite values")

        data: Any = gripper_model.get("data")
        model: Any = gripper_model.get("model")

        if data is None or model is None:
            raise RuntimeError("Simulation data or model is not bound to gripper model")

        expected_nu: Any = gripper_model["nu"]
        if command.shape != (expected_nu,):
            raise ValueError(
                f"command shape {command.shape} does not match gripper actuators ({expected_nu},)"
            )

        active_model = model
        active_data = data

        actuator_names: Any = gripper_model["actuator_names"]
        for name, val in zip(actuator_names, command, strict=False):
            act_id = mujoco.mj_name2id(active_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id != -1:
                active_data.ctrl[act_id] = val

    return controller


def make_open_command(gripper_model: dict[str, object]) -> GripperCommand:
    """Build an "open" gripper command for the supplied gripper model.

    Args:
        gripper_model: Gripper model returned by ``load_gripper_model``.

    Returns:
        A gripper command vector that fully opens the gripper.
    """
    if not isinstance(gripper_model, dict) or "model" not in gripper_model:
        raise TypeError("gripper_model must be a dictionary returned by load_gripper_model")

    if "open_command" in gripper_model:
        return np.array(gripper_model["open_command"], dtype=float)

    model: Any = gripper_model["model"]
    nu: Any = gripper_model["nu"]
    cmd = np.zeros(nu)
    for i in range(nu):
        if model.actuator_ctrllimited[i]:
            cmd[i] = model.actuator_ctrlrange[i, 0]
        else:
            cmd[i] = 0.0
    return cmd


def make_close_command(gripper_model: dict[str, object]) -> GripperCommand:
    """Build a "close" gripper command for the supplied gripper model.

    Args:
        gripper_model: Gripper model returned by ``load_gripper_model``.

    Returns:
        A gripper command vector that closes the gripper.
    """
    if not isinstance(gripper_model, dict) or "model" not in gripper_model:
        raise TypeError("gripper_model must be a dictionary returned by load_gripper_model")

    if "close_command" in gripper_model:
        return np.array(gripper_model["close_command"], dtype=float)

    model: Any = gripper_model["model"]
    nu: Any = gripper_model["nu"]
    cmd = np.zeros(nu)
    for i in range(nu):
        if model.actuator_ctrllimited[i]:
            cmd[i] = model.actuator_ctrlrange[i, 1]
        else:
            cmd[i] = 1.0
    return cmd
