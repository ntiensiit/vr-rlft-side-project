"""Gripper actuator discovery and control helpers."""

from __future__ import annotations

from grasping_ai.perception.geometry import make_transform

from grasping_ai.simulation.mujoco_env import set_actuator_controls

from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco  # type: ignore[import-untyped]
import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

def panda_hand_to_contact_transform() -> np.ndarray:
    """Return the Panda hand-base to contact-center rigid transform.

    Values match ``configs/gripper/franka_emika_panda.yaml`` and the MuJoCo Grasping
    Simulator Panda gripper definition (translation along hand z, wxyz quaternion).

    Returns:
        A ``(4, 4)`` homogeneous transform ``T_hand_contact``.
    """
    import pytransform3d.rotations as pr
    
    position = np.array([0.0, 0.0, -0.102], dtype=np.float64)
    quaternion_wxyz = np.array([0.707106781, 0.0, 0.0, 0.707106781], dtype=np.float64)
    rotation = pr.matrix_from_quaternion(quaternion_wxyz)
    return make_transform(rotation, position)

def panda_width_to_finger_joints(width: float) -> tuple[float, float]:
    """Map a Panda finger opening width in meters to slide-joint targets.

    Args:
        width: Desired opening width in meters.

    Returns:
        A tuple ``(finger_joint1, finger_joint2)`` clipped to Panda slide ranges.
    """
    clamped_width = float(np.clip(width, 0.003, 0.08))
    target_q1 = clamped_width / 2.0
    target_q2 = -0.04 + (clamped_width / 2.0)
    target_q1 = float(np.clip(target_q1, 0.0, 0.04))
    target_q2 = float(np.clip(target_q2, -0.04, 0.0))
    return target_q1, target_q2

def gripper_actuator_indices(mj_model: Any) -> list[int]:
    """Return actuator indices that drive the gripper rather than the arm."""
    indices: list[int] = []
    for i in range(int(mj_model.nu)):
        name = (mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "").lower()
        if any(token in name for token in ("finger", "gripper")):
            indices.append(i)
            continue
        if mj_model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON:
            indices.append(i)
            continue
        if mj_model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            joint_id = int(mj_model.actuator_trnid[i, 0])
            jname = (mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "").lower()
            if "finger" in jname or "gripper" in jname:
                indices.append(i)
    logger.info("Found gripper actuator indices: {}", indices)
    return indices

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
        raise FileNotFoundError(f"Gripper description file '{gripper_description_path}' not found")

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

def build_gripper_controller(gripper_model: dict[str, object]) -> Callable[[np.ndarray], None]:
    """Build a callable gripper controller that issues open/close commands.

    The controller writes through the authoritative actuator-control path
    (``set_actuator_controls``), the same path used by the Gymnasium
    environment and the grasp-simulation pipeline. The gripper model must be
    bound to simulation data via the ``"model"`` and ``"data"`` keys before
    the controller is invoked.

    Args:
        gripper_model: Gripper model returned by ``load_gripper_model``.

    Returns:
        A callable accepting a gripper command vector and applying it to the
        underlying simulation environment.
    """
    if not isinstance(gripper_model, dict) or "model" not in gripper_model:
        raise TypeError("gripper_model must be a dictionary returned by load_gripper_model")

    
    def controller(command: np.ndarray) -> None:
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
            raise ValueError(f"command shape {command.shape} does not match gripper actuators ({expected_nu},)")

        set_actuator_controls({"model": model, "data": data}, command)

    return controller

def make_open_command(gripper_model: dict[str, object]) -> np.ndarray:
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

def make_close_command(gripper_model: dict[str, object]) -> np.ndarray:
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
