import numpy as np

RotationMatrix = np.ndarray
Translation = np.ndarray
Transform4x4 = np.ndarray


def identity_transform() -> Transform4x4:
    """Construct the identity rigid-body transformation.

    Returns:
        A ``(4, 4)`` identity transform matrix.
    """
    raise NotImplementedError


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> RotationMatrix:
    """Build a rotation matrix from a unit axis and an angle in radians.

    Args:
        axis: Unit-length rotation axis with shape ``(3,)``.
        angle: Rotation angle in radians.

    Returns:
        A ``(3, 3)`` rotation matrix representing the rotation.
    """
    raise NotImplementedError


def rotation_matrix_to_axis_angle(rotation: RotationMatrix) -> tuple[np.ndarray, float]:
    """Recover the axis-angle representation of a rotation matrix.

    Args:
        rotation: A ``(3, 3)`` rotation matrix.

    Returns:
        A tuple ``(axis, angle)`` where ``axis`` has shape ``(3,)`` and
        ``angle`` is expressed in radians.
    """
    raise NotImplementedError


def make_transform(rotation: RotationMatrix, translation: Translation) -> Transform4x4:
    """Assemble a rigid ``(4, 4)`` transformation from rotation and translation.

    Args:
        rotation: A ``(3, 3)`` rotation matrix.
        translation: A ``(3,)`` translation vector.

    Returns:
        A ``(4, 4)`` homogeneous transformation matrix.
    """
    raise NotImplementedError


def invert_transform(transform: Transform4x4) -> Transform4x4:
    """Compute the inverse of a rigid-body transformation.

    Args:
        transform: A ``(4, 4)`` rigid transformation matrix.

    Returns:
        The inverse transformation matrix.
    """
    raise NotImplementedError


def apply_transform(points: np.ndarray, transform: Transform4x4) -> np.ndarray:
    """Apply a rigid transformation to a set of points.

    Args:
        points: Point cloud with shape ``(N, 3)``.
        transform: A ``(4, 4)`` rigid transformation matrix.

    Returns:
        Transformed point cloud with shape ``(N, 3)``.
    """
    raise NotImplementedError


def grasp_pose_to_transform(rotation: RotationMatrix, translation: Translation) -> Transform4x4:
    """Convert an SE(3) grasp pose into a 4x4 transformation matrix.

    Args:
        rotation: Gripper orientation as a ``(3, 3)`` rotation matrix.
        translation: Gripper position as a ``(3,)`` translation vector.

    Returns:
        A ``(4, 4)`` homogeneous transformation representing the grasp pose.
    """
    raise NotImplementedError
