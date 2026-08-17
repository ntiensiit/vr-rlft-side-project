"""Rigid-body transform utilities for robotics."""

from __future__ import annotations

import numpy as np

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
from grasping_ai.perception.geometry import (
    apply_transform,
    invert_transform,
)

GRASP_POSES_NDIM = int(FLATTENED_YAML_CONFIG.get("grasp.poses_ndim", 3))
POINT_CLOUD_NDIM = int(FLATTENED_YAML_CONFIG.get("geometry.point_cloud_ndim", 2))
SE3_MATRIX_SHAPE = tuple(int(v) for v in FLATTENED_YAML_CONFIG.get("grasp.se3_matrix_shape", [4, 4]))
SPATIAL_DIM = int(FLATTENED_YAML_CONFIG.get("geometry.spatial_dim", 3))

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
        msg = "source_to_target must be a (4, 4) numpy array"
        raise ValueError(msg)
    if not isinstance(point_in_source, np.ndarray):
        msg = "point_in_source must be a numpy array"
        raise TypeError(msg)

    if point_in_source.shape == (3,):
        return apply_transform(point_in_source.reshape(1, 3), source_to_target)[0]
    if len(point_in_source.shape) == POINT_CLOUD_NDIM and point_in_source.shape[1] == SPATIAL_DIM:
        return apply_transform(point_in_source, source_to_target)
    msg = "point_in_source must have shape (3,) or (N, 3)"
    raise ValueError(msg)


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
        msg = "grasp_to_world must be a (4, 4) numpy array"
        raise ValueError(msg)
    if not isinstance(gripper_to_grasp, np.ndarray) or gripper_to_grasp.shape != SE3_MATRIX_SHAPE:
        msg = "gripper_to_grasp must be a (4, 4) numpy array"
        raise ValueError(msg)

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
        msg = "object_to_world must be a (4, 4) numpy array"
        raise ValueError(msg)
    if not isinstance(grasps, np.ndarray):
        msg = "grasps must be a numpy array"
        raise TypeError(msg)

    if grasps.shape == (4, 4):
        return object_to_world @ grasps
    if len(grasps.shape) == GRASP_POSES_NDIM and grasps.shape[1:] == SE3_MATRIX_SHAPE:
        out = np.empty(
            grasps.shape,
            dtype=np.result_type(grasps.dtype, object_to_world.dtype),
        )
        for i in range(len(grasps)):
            out[i] = object_to_world @ grasps[i]
        return out
    msg = "grasps must have shape (4, 4) or (N, 4, 4)"
    raise ValueError(msg)


def invert_rigid_transform(transform: RigidTransform) -> RigidTransform:
    """Invert a rigid ``(4, 4)`` transformation matrix.

    Args:
        transform: A rigid transformation matrix.

    Returns:
        The inverse rigid transformation.
    """
    if not isinstance(transform, np.ndarray) or transform.shape != SE3_MATRIX_SHAPE:
        msg = "transform must be a (4, 4) numpy array"
        raise ValueError(msg)
    return invert_transform(transform)
