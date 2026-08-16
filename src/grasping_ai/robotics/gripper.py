"""Gripper actuator discovery and control helpers."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from typing import Any

import mujoco  # type: ignore[import-untyped]
import numpy as np
import pytransform3d.rotations as pr
from loguru import logger

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
from grasping_ai.perception.geometry import make_transform

GRIPPER_GRID = tuple(
    (axis, tuple(FLATTENED_YAML_CONFIG.get_path("observations", "gripper_grid", axis).items()))
    for axis in ("x", "y", "z")
)
BASE_TO_CONTACT_POSITION = tuple(
    FLATTENED_YAML_CONFIG.get_path("robot", "gripper", "base_to_contact", "position"),
)
BASE_TO_CONTACT_QUATERNION = tuple(
    FLATTENED_YAML_CONFIG.get_path("robot", "gripper", "base_to_contact", "quaternion_wxyz"),
)
MIN_WIDTH_CLAMP = float(FLATTENED_YAML_CONFIG.get("robot.gripper.min_width_clamp"))
MAX_WIDTH = float(FLATTENED_YAML_CONFIG.get("robot.gripper.max_width"))
JOINT_RANGES = tuple(
    (name, tuple(FLATTENED_YAML_CONFIG.get_path("robot", "gripper", "joint_ranges", name)))
    for name in ("finger_joint1", "finger_joint2")
)


def linspace_axis(axis_cfg: object) -> np.ndarray:
    """Build a 1D axis from a ``start``/``stop``/``count`` mapping."""
    if not isinstance(axis_cfg, dict):
        msg = "gripper grid axis must be a mapping with start, stop, and count"
        raise TypeError(msg)
    return np.linspace(float(axis_cfg["start"]), float(axis_cfg["stop"]), int(axis_cfg["count"]))


def gripper_point_cloud_from_grid(grid: Mapping[str, object]) -> np.ndarray:
    """Build a gripper collision point cloud from axis grid specifications."""
    x = linspace_axis(grid["x"])
    y = linspace_axis(grid["y"])
    z = linspace_axis(grid["z"])
    point_cloud = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
    return point_cloud.astype(np.float32)


def default_gripper_point_cloud(
    grid_default: tuple[tuple[str, tuple[tuple[str, object], ...]], ...] = GRIPPER_GRID,
) -> np.ndarray:
    """Return the default analytical gripper collision point cloud."""
    grid = {axis: dict(values) for axis, values in grid_default}
    return gripper_point_cloud_from_grid(grid)


def panda_hand_to_contact_transform(
    position: tuple[float, ...] = BASE_TO_CONTACT_POSITION,
    quaternion_wxyz: tuple[float, ...] = BASE_TO_CONTACT_QUATERNION,
) -> np.ndarray:
    """Return the Panda hand-base to contact-center rigid transform.

    Values match ``configs/gripper/franka_emika_panda.yaml`` and the MuJoCo Grasping
    Simulator Panda gripper definition (translation along hand z, wxyz quaternion).

    Returns:
        A ``(4, 4)`` homogeneous transform ``T_hand_contact``.
    """
    rotation = pr.matrix_from_quaternion(np.asarray(quaternion_wxyz, dtype=np.float64))
    return make_transform(rotation, np.asarray(position, dtype=np.float64))


def panda_width_to_finger_joints(
    width: float,
    min_width_clamp: float = MIN_WIDTH_CLAMP,
    max_width: float = MAX_WIDTH,
    joint_ranges: tuple[tuple[str, tuple[float, ...]], ...] = JOINT_RANGES,
) -> tuple[float, float]:
    """Map a Panda finger opening width in meters to slide-joint targets.

    Args:
        width: Desired opening width in meters.
        min_width_clamp: Minimum allowed opening width.
        max_width: Maximum allowed opening width.
        joint_ranges: Immutable finger-joint range defaults.

    Returns:
        A tuple ``(finger_joint1, finger_joint2)`` clipped to Panda slide ranges.
    """
    clamped_width = float(
        np.clip(
            width,
            min_width_clamp,
            max_width,
        ),
    )
    target_q1 = clamped_width / 2.0
    ranges = dict(joint_ranges)
    finger1_range = ranges["finger_joint1"]
    finger2_range = ranges["finger_joint2"]
    target_q2 = float(finger2_range[0]) + (clamped_width / 2.0)
    finger1_min, finger1_max = (float(value) for value in finger1_range)
    finger2_min, finger2_max = (float(value) for value in finger2_range)
    target_q1 = float(np.clip(target_q1, finger1_min, finger1_max))
    target_q2 = float(np.clip(target_q2, finger2_min, finger2_max))
    return target_q1, target_q2


def gripper_actuator_indices(mj_model: mujoco.MjModel) -> list[int]:
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
        msg = "gripper_description_path must be a string"
        raise TypeError(msg)
    if not gripper_description_path:
        msg = "gripper_description_path must not be empty"
        raise ValueError(msg)

    path = Path(gripper_description_path)
    if not path.is_file():
        msg = f"Gripper description file '{gripper_description_path}' not found"
        raise FileNotFoundError(msg)

    try:
        model = mujoco.MjModel.from_xml_path(str(path))
    except Exception as e:
        msg = f"Failed to load MuJoCo gripper model: {e}"
        raise ValueError(msg) from e

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


def make_open_command(gripper_model: dict[str, object]) -> np.ndarray:
    """Build an "open" gripper command for the supplied gripper model.

    Args:
        gripper_model: Gripper model returned by ``load_gripper_model``.

    Returns:
        A gripper command vector that fully opens the gripper.
    """
    if not isinstance(gripper_model, dict) or "model" not in gripper_model:
        msg = "gripper_model must be a dictionary returned by load_gripper_model"
        raise TypeError(msg)

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
        msg = "gripper_model must be a dictionary returned by load_gripper_model"
        raise TypeError(msg)

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
