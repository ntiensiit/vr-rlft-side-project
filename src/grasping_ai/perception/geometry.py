"""SE(3) frame construction and rotation utilities."""

from __future__ import annotations

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG

import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt

POINT_CLOUD_NDIM = int(FLATTENED_YAML_CONFIG.get("geometry.point_cloud_ndim", 2))
SE3_MATRIX_SHAPE = tuple(int(v) for v in FLATTENED_YAML_CONFIG.get("grasp.se3_matrix_shape", [4, 4]))
SPATIAL_DIM = int(FLATTENED_YAML_CONFIG.get("geometry.spatial_dim", 3))

RotationMatrix = np.ndarray
Translation = np.ndarray
Transform4x4 = np.ndarray

def identity_transform() -> Transform4x4:
    """Construct the identity rigid-body transformation.

    Returns:
        A ``(4, 4)`` identity transform matrix.
    """
    return np.eye(4)

def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> RotationMatrix:
    """Build a rotation matrix from a unit axis and an angle in radians.

    Args:
        axis: Unit-length rotation axis with shape ``(3,)``.
        angle: Rotation angle in radians.

    Returns:
        A ``(3, 3)`` rotation matrix representing the rotation.

    Raises:
        TypeError: If inputs have incorrect types.
        ValueError: If inputs have incorrect shapes or axis is not unit-length.
    """
    if not isinstance(axis, np.ndarray):
        raise TypeError("Axis must be a numpy array")
    if axis.shape != (3,):
        raise ValueError("Axis must have shape (3,)")
    if not isinstance(angle, (int, float, np.floating, np.integer)):
        raise TypeError("Angle must be a float or integer")
    norm = np.linalg.norm(axis)
    if not np.allclose(norm, 1.0):
        raise ValueError(f"Axis must be a unit vector (norm: {norm})")

    # Concatenate axis and angle to make axis-angle vector
    a = np.hstack((axis, angle))
    return pr.matrix_from_axis_angle(a)

def rotation_matrix_to_axis_angle(rotation: RotationMatrix) -> tuple[np.ndarray, float]:
    """Recover the axis-angle representation of a rotation matrix.

    Args:
        rotation: A ``(3, 3)`` rotation matrix.

    Returns:
        A tuple ``(axis, angle)`` where ``axis`` has shape ``(3,)`` and
        ``angle`` is expressed in radians.

    Raises:
        TypeError: If input has incorrect type.
        ValueError: If input is not a 3x3 matrix.
    """
    if not isinstance(rotation, np.ndarray):
        raise TypeError("Rotation must be a numpy array")
    if rotation.shape != (3, 3):
        raise ValueError("Rotation must be a (3, 3) matrix")

    a = pr.axis_angle_from_matrix(rotation)
    axis = a[:3]
    angle = a[3]
    return axis, angle

def make_transform(rotation: RotationMatrix, translation: Translation) -> Transform4x4:
    """Assemble a rigid ``(4, 4)`` transformation from rotation and translation.

    Args:
        rotation: A ``(3, 3)`` rotation matrix.
        translation: A ``(3,)`` translation vector.

    Returns:
        A ``(4, 4)`` homogeneous transformation matrix.

    Raises:
        TypeError: If inputs have incorrect types.
        ValueError: If inputs have incorrect shapes.
    """
    if not isinstance(rotation, np.ndarray):
        raise TypeError("Rotation must be a numpy array")
    if not isinstance(translation, np.ndarray):
        raise TypeError("Translation must be a numpy array")
    if rotation.shape != (3, 3):
        raise ValueError("Rotation must have shape (3, 3)")
    if translation.shape != (3,):
        raise ValueError("Translation must have shape (3,)")

    return pt.transform_from(rotation, translation)

def invert_transform(transform: Transform4x4) -> Transform4x4:
    """Compute the inverse of a rigid-body transformation.

    Args:
        transform: A ``(4, 4)`` rigid transformation matrix.

    Returns:
        The inverse transformation matrix.

    Raises:
        TypeError: If input has incorrect type.
        ValueError: If input is not a 4x4 matrix.
    """
    if not isinstance(transform, np.ndarray):
        raise TypeError("Transform must be a numpy array")
    if transform.shape != SE3_MATRIX_SHAPE:
        raise ValueError("Transform must have shape (4, 4)")

    return pt.invert_transform(transform)

def apply_transform(points: np.ndarray, transform: Transform4x4) -> np.ndarray:
    """Apply a rigid transformation to a set of points.

    Args:
        points: Point cloud with shape ``(N, 3)``.
        transform: A ``(4, 4)`` rigid transformation matrix.

    Returns:
        Transformed point cloud with shape ``(N, 3)``.

    Raises:
        TypeError: If inputs have incorrect types.
        ValueError: If inputs have incorrect shapes.
    """
    if not isinstance(points, np.ndarray):
        raise TypeError("Points must be a numpy array")
    if not isinstance(transform, np.ndarray):
        raise TypeError("Transform must be a numpy array")
    if len(points.shape) != POINT_CLOUD_NDIM or points.shape[1] != SPATIAL_DIM:
        raise ValueError("Points must have shape (N, 3)")
    if transform.shape != SE3_MATRIX_SHAPE:
        raise ValueError("Transform must have shape (4, 4)")

    # Convert vectors/points to homogeneous coordinates (N, 4) where last column is 1
    points_hom = pt.vectors_to_points(points)
    transformed_hom = pt.transform(transform, points_hom)
    # Convert back to (N, 3)
    return transformed_hom[:, :3]

def grasp_pose_to_transform(rotation: RotationMatrix, translation: Translation) -> Transform4x4:
    """Convert an SE(3) grasp pose into a 4x4 transformation matrix.

    Args:
        rotation: Gripper orientation as a ``(3, 3)`` rotation matrix.
        translation: Gripper position as a ``(3,)`` translation vector.

    Returns:
        A ``(4, 4)`` homogeneous transformation representing the grasp pose.

    Raises:
        TypeError: If inputs have incorrect types.
        ValueError: If inputs have incorrect shapes.
    """
    return make_transform(rotation, translation)
