"""Inverse kinematics helpers for simulated robots."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco  # type: ignore[import-untyped]
import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from loguru import logger

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
from grasping_ai.perception.geometry import make_transform

SE3_MATRIX_SHAPE = tuple(int(v) for v in FLATTENED_YAML_CONFIG.get("grasp.se3_matrix_shape", [4, 4]))

JointConfiguration = np.ndarray
RigidTransform = np.ndarray
ForwardKinematics = Callable[[JointConfiguration], RigidTransform]

_EE_BODY_CANDIDATES = (
    "end_effector",
    "ee",
    "flange",
    "gripper",
    "hand",
)


def _resolve_end_effector_body_name(model: mujoco.MjModel, robot_model: dict[str, object]) -> str:
    """Return the end-effector body name for FK/IK."""
    ee_body_name: Any = robot_model.get("end_effector_body_name")
    if ee_body_name is None:
        for name in _EE_BODY_CANDIDATES:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id != -1:
                ee_body_name = name
                break
        if ee_body_name is None:
            ee_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody - 1)
    logger.info("Resolved end-effector body name: {}", ee_body_name)
    return str(ee_body_name)


def _se3_pose_error(target_pose: RigidTransform, current_pose: RigidTransform) -> np.ndarray:
    """Compute a 6D SE(3) pose error using ``pytransform3d`` conventions.

    Args:
        target_pose: Desired end-effector transform with shape ``(4, 4)``.
        current_pose: Current end-effector transform with shape ``(4, 4)``.

    Returns:
        Concatenated translation and compact axis-angle rotation error.
    """
    delta = pt.concat(target_pose, pt.invert_transform(current_pose))
    err_pos = delta[:3, 3]
    err_rot = pr.compact_axis_angle_from_matrix(delta[:3, :3])
    return np.hstack((err_pos, err_rot))


def load_robot_model(robot_description_path: str) -> dict[str, object]:
    """Load a robot description from disk.

    Args:
        robot_description_path: Path to a robot description file such as an
            XML/MJCF robot definition.

    Returns:
        A dictionary describing robot kinematic and dynamic parameters.
    """
    if not isinstance(robot_description_path, str):
        msg = "robot_description_path must be a string"
        raise TypeError(msg)
    if not robot_description_path:
        msg = "robot_description_path must not be empty"
        raise ValueError(msg)

    path = Path(robot_description_path)
    if not path.is_file():
        msg = f"Robot description file '{robot_description_path}' not found"
        raise FileNotFoundError(msg)

    try:
        model = mujoco.MjModel.from_xml_path(str(path))
    except Exception as e:
        msg = f"Failed to load MuJoCo robot model: {e}"
        raise ValueError(msg) from e

    return {
        "model": model,
        "path": path,
        "nq": model.nq,
        "nv": model.nv,
    }


def robot_model_nq(robot_model: dict[str, object]) -> int:
    """Return joint count from a ``load_robot_model`` dictionary.

    Args:
        robot_model: Robot model dictionary returned by ``load_robot_model``.

    Returns:
        Number of generalized coordinates ``nq`` for the robot.

    Raises:
        TypeError: If ``robot_model['nq']`` is not an ``int``.
    """
    nq = robot_model.get("nq")
    if not isinstance(nq, int):
        msg = "robot_model['nq'] must be int"
        raise TypeError(msg)
    return nq


def robot_model_mj_model(robot_model: dict[str, object]) -> mujoco.MjModel:
    """Return the MuJoCo model object from a ``load_robot_model`` dictionary.

    Args:
        robot_model: Robot model dictionary returned by ``load_robot_model``.

    Returns:
        MuJoCo ``MjModel`` instance stored under the ``"model"`` key.

    Raises:
        TypeError: If ``robot_model`` does not contain a ``"model"`` entry.
    """
    model = robot_model.get("model")
    if model is None:
        msg = "robot_model must contain 'model'"
        raise TypeError(msg)
    return model


def build_forward_kinematics(robot_model: dict[str, object]) -> ForwardKinematics:
    """Build a callable forward-kinematics function for a robot.

    Args:
        robot_model: Robot model returned by ``load_robot_model``.

    Returns:
        A callable mapping joint configurations to end-effector transforms.
    """
    if not isinstance(robot_model, dict) or "model" not in robot_model:
        msg = "robot_model must be a dictionary returned by load_robot_model"
        raise TypeError(msg)

    model: Any = robot_model["model"]
    ee_body_name = _resolve_end_effector_body_name(model, robot_model)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
    if body_id == -1:
        msg = f"End effector body '{ee_body_name}' not found in robot model"
        raise ValueError(msg)

    local_data = mujoco.MjData(model)

    def fk(joints: JointConfiguration) -> RigidTransform:
        if not isinstance(joints, np.ndarray):
            msg = "joints must be a numpy array"
            raise TypeError(msg)
        if joints.shape != (model.nq,):
            msg = f"joints shape {joints.shape} does not match model.nq ({model.nq})"
            raise ValueError(msg)
        if not np.isfinite(joints).all():
            msg = "joints must contain only finite values"
            raise ValueError(msg)

        local_data.qpos[:] = joints
        mujoco.mj_forward(model, local_data)

        return make_transform(
            local_data.xmat[body_id].reshape(3, 3),
            local_data.xpos[body_id],
        )

    return fk


def _validate_ik_inputs(target_pose: np.ndarray, initial_joints: np.ndarray, model: mujoco.MjModel) -> None:
    """Validate an IK target pose and initial joint configuration."""
    if not isinstance(target_pose, np.ndarray) or target_pose.shape != SE3_MATRIX_SHAPE:
        msg = "target_pose must be a (4, 4) numpy array"
        raise ValueError(msg)
    if not isinstance(initial_joints, np.ndarray) or initial_joints.shape != (model.nq,):
        msg = f"initial_joints shape {initial_joints.shape} does not match model.nq ({model.nq})"
        raise ValueError(msg)
    if not np.isfinite(target_pose).all():
        msg = "target_pose must contain only finite values"
        raise ValueError(msg)
    if not np.isfinite(initial_joints).all():
        msg = "initial_joints must contain only finite values"
        raise ValueError(msg)


def _ee_pose(local_data: mujoco.MjData, body_id: int) -> np.ndarray:
    """Read the current pose of a body from simulation data."""
    curr_pos = local_data.xpos[body_id]
    curr_rot = local_data.xmat[body_id].reshape(3, 3)
    return make_transform(curr_rot, curr_pos)


def _clamp_joint_limits(q: np.ndarray, model: mujoco.MjModel) -> None:
    """Clamp hinge/slide joint positions to their configured ranges in place."""
    for j in range(model.njnt):
        if model.jnt_limited[j] and model.jnt_type[j] in [
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ]:
            qadr = model.jnt_qposadr[j]
            q[qadr] = np.clip(q[qadr], model.jnt_range[j, 0], model.jnt_range[j, 1])


def build_inverse_kinematics(
    robot_model: dict[str, object],
    max_iterations: int,
    tolerance: float,
) -> Callable[..., np.ndarray]:
    """Build a numerical inverse-kinematics solver for a robot.

    Args:
        robot_model: Robot model returned by ``load_robot_model``.
        max_iterations: Maximum number of solver iterations.
        tolerance: Convergence tolerance on the pose error.

    Returns:
        A callable that accepts a target end-effector transform and an
        initial joint configuration and returns a solved joint configuration.
    """
    if not isinstance(robot_model, dict) or "model" not in robot_model:
        msg = "robot_model must be a dictionary returned by load_robot_model"
        raise TypeError(msg)
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        msg = "max_iterations must be a positive integer"
        raise ValueError(msg)
    if not isinstance(tolerance, (int, float)) or tolerance <= 0:
        msg = "tolerance must be a positive float"
        raise ValueError(msg)

    model: Any = robot_model["model"]
    ee_body_name = _resolve_end_effector_body_name(model, robot_model)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
    if body_id == -1:
        msg = f"End effector body '{ee_body_name}' not found in robot model"
        raise ValueError(msg)

    local_data = mujoco.MjData(model)

    def ik(target_pose: RigidTransform, initial_joints: JointConfiguration) -> JointConfiguration:
        _validate_ik_inputs(target_pose, initial_joints, model)

        q = np.copy(initial_joints)
        damping = 0.01

        for _ in range(max_iterations):
            local_data.qpos[:] = q
            mujoco.mj_forward(model, local_data)

            err = _se3_pose_error(target_pose, _ee_pose(local_data, body_id))
            if np.linalg.norm(err) < tolerance:
                break

            jac_pos = np.zeros((3, model.nv))
            jac_rot = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, local_data, jac_pos, jac_rot, body_id)
            jac = np.vstack((jac_pos, jac_rot))

            j_jt = jac @ jac.T
            damped_inv = np.linalg.inv(j_jt + (damping**2) * np.eye(6))
            dq = jac.T @ damped_inv @ err

            q += dq

            _clamp_joint_limits(q, model)

        # Verify final error
        local_data.qpos[:] = q
        mujoco.mj_forward(model, local_data)
        final_err = np.linalg.norm(_se3_pose_error(target_pose, _ee_pose(local_data, body_id)))

        if final_err > tolerance:
            msg = f"IK solver failed to converge within tolerance {tolerance} (final error: {final_err})"
            raise ValueError(msg)

        return q

    return ik


def solve_inverse_kinematics(
    ik_solver: Callable[..., np.ndarray],
    target_pose: RigidTransform,
    initial_joints: JointConfiguration,
) -> JointConfiguration:
    """Run inverse kinematics to find a joint configuration for a target pose.

    Args:
        ik_solver: Solver returned by ``build_inverse_kinematics``.
        target_pose: Desired end-effector pose as a ``(4, 4)`` transform.
        initial_joints: Initial joint configuration used to seed the solver.

    Returns:
        A joint configuration that achieves ``target_pose`` to within tolerance.
    """
    if not callable(ik_solver):
        msg = "ik_solver must be a callable"
        raise TypeError(msg)
    return ik_solver(target_pose, initial_joints)
