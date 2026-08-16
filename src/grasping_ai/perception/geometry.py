"""SE(3) frame construction and rotation utilities."""

from __future__ import annotations

import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG

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
        msg = "Axis must be a numpy array"
        raise TypeError(msg)
    if axis.shape != (3,):
        msg = "Axis must have shape (3,)"
        raise ValueError(msg)
    if not isinstance(angle, (int, float, np.floating, np.integer)):
        msg = "Angle must be a float or integer"
        raise TypeError(msg)
    norm = np.linalg.norm(axis)
    if not np.allclose(norm, 1.0):
        msg = f"Axis must be a unit vector (norm: {norm})"
        raise ValueError(msg)

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
        msg = "Rotation must be a numpy array"
        raise TypeError(msg)
    if rotation.shape != (3, 3):
        msg = "Rotation must be a (3, 3) matrix"
        raise ValueError(msg)

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
        msg = "Rotation must be a numpy array"
        raise TypeError(msg)
    if not isinstance(translation, np.ndarray):
        msg = "Translation must be a numpy array"
        raise TypeError(msg)
    if rotation.shape != (3, 3):
        msg = "Rotation must have shape (3, 3)"
        raise ValueError(msg)
    if translation.shape != (3,):
        msg = "Translation must have shape (3,)"
        raise ValueError(msg)

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
        msg = "Transform must be a numpy array"
        raise TypeError(msg)
    if transform.shape != SE3_MATRIX_SHAPE:
        msg = "Transform must have shape (4, 4)"
        raise ValueError(msg)

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
        msg = "Points must be a numpy array"
        raise TypeError(msg)
    if not isinstance(transform, np.ndarray):
        msg = "Transform must be a numpy array"
        raise TypeError(msg)
    if len(points.shape) != POINT_CLOUD_NDIM or points.shape[1] != SPATIAL_DIM:
        msg = "Points must have shape (N, 3)"
        raise ValueError(msg)
    if transform.shape != SE3_MATRIX_SHAPE:
        msg = "Transform must have shape (4, 4)"
        raise ValueError(msg)

    # Convert vectors/points to homogeneous coordinates (N, 4) where last column is 1
    points_hom = pt.vectors_to_points(points)
    transformed_hom = pt.transform(transform, points_hom)
    # Convert back to (N, 3)
    return transformed_hom[:, :3]


