"""Rigid-body transform utilities for robotics."""

from __future__ import annotations

import numpy as np

from grasping_ai.perception.geometry import apply_transform, invert_transform
from grasping_ai.utils.constants import GRASP_POSES_NDIM, POINT_CLOUD_NDIM, SE3_MATRIX_SHAPE, SPATIAL_DIM

RigidTransform = np.ndarray


def transform_between_frames(source_to_target: RigidTransform, point_in_source: np.ndarray) -> np.ndarray:
    """Transform a point expressed in a source frame into a target frame.

    Args:
        source_to_target: ``(4, 4)`` transform from the source frame to the
            target frame.
        point_in_source: Points expressed in the source frame.

    Returns:
        Points expressed in the target frame.
    """
    if not isinstance(source_to_target, np.ndarray) or source_to_target.shape != SE3_MATRIX_SHAPE:
        raise ValueError("source_to_target must be a (4, 4) numpy array")
    if not isinstance(point_in_source, np.ndarray):
        raise TypeError("point_in_source must be a numpy array")

    if point_in_source.shape == (3,):
        return apply_transform(point_in_source.reshape(1, 3), source_to_target)[0]
    if len(point_in_source.shape) == POINT_CLOUD_NDIM and point_in_source.shape[1] == SPATIAL_DIM:
        return apply_transform(point_in_source, source_to_target)
    raise ValueError("point_in_source must have shape (3,) or (N, 3)")


def transform_grasp_pose(grasp_to_world: RigidTransform, gripper_to_grasp: RigidTransform) -> RigidTransform:
    """Compose grasp-pose transforms into a single gripper-in-world transform.

    Args:
        grasp_to_world: ``(4, 4)`` transform from the grasp frame to world.
        gripper_to_grasp: ``(4, 4)`` transform from the gripper frame to the
            grasp frame.

    Returns:
        A ``(4, 4)`` transform placing the gripper origin in the world frame.
    """
    if not isinstance(grasp_to_world, np.ndarray) or grasp_to_world.shape != SE3_MATRIX_SHAPE:
        raise ValueError("grasp_to_world must be a (4, 4) numpy array")
    if not isinstance(gripper_to_grasp, np.ndarray) or gripper_to_grasp.shape != SE3_MATRIX_SHAPE:
        raise ValueError("gripper_to_grasp must be a (4, 4) numpy array")

    return grasp_to_world @ gripper_to_grasp


def convert_grasps_to_world_frame(grasps: np.ndarray, object_to_world: RigidTransform) -> np.ndarray:
    """Convert object-frame grasp poses into the world frame.

    Args:
        grasps: Grasp poses expressed in the object frame.
        object_to_world: ``(4, 4)`` transform from object frame to world frame.

    Returns:
        Grasp poses expressed in the world frame.
    """
    if not isinstance(object_to_world, np.ndarray) or object_to_world.shape != SE3_MATRIX_SHAPE:
        raise ValueError("object_to_world must be a (4, 4) numpy array")
    if not isinstance(grasps, np.ndarray):
        raise TypeError("grasps must be a numpy array")

    if grasps.shape == (4, 4):
        return object_to_world @ grasps
    if len(grasps.shape) == GRASP_POSES_NDIM and grasps.shape[1:] == SE3_MATRIX_SHAPE:
        out = np.zeros_like(grasps)
        for i in range(len(grasps)):
            out[i] = object_to_world @ grasps[i]
        return out
    raise ValueError("grasps must have shape (4, 4) or (N, 4, 4)")


def invert_rigid_transform(transform: RigidTransform) -> RigidTransform:
    """Invert a rigid ``(4, 4)`` transformation matrix.

    Args:
        transform: A rigid transformation matrix.

    Returns:
        The inverse rigid transformation.
    """
    if not isinstance(transform, np.ndarray) or transform.shape != SE3_MATRIX_SHAPE:
        raise ValueError("transform must be a (4, 4) numpy array")
    return invert_transform(transform)
