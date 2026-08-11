from collections.abc import Callable

import numpy as np

RigidTransform = np.ndarray
FrameConversion = Callable[[np.ndarray, np.ndarray], np.ndarray]


def transform_between_frames(
    source_to_target: RigidTransform, point_in_source: np.ndarray
) -> np.ndarray:
    """Transform a point expressed in a source frame into a target frame.

    Args:
        source_to_target: ``(4, 4)`` transform from the source frame to the
            target frame.
        point_in_source: Points expressed in the source frame.

    Returns:
        Points expressed in the target frame.
    """
    raise NotImplementedError


def transform_grasp_pose(
    grasp_to_world: RigidTransform, gripper_to_grasp: RigidTransform
) -> RigidTransform:
    """Compose grasp-pose transforms into a single gripper-in-world transform.

    Args:
        grasp_to_world: ``(4, 4)`` transform from the grasp frame to world.
        gripper_to_grasp: ``(4, 4)`` transform from the gripper frame to the
            grasp frame.

    Returns:
        A ``(4, 4)`` transform placing the gripper origin in the world frame.
    """
    raise NotImplementedError


def convert_grasps_to_world_frame(
    grasps: np.ndarray, object_to_world: RigidTransform
) -> np.ndarray:
    """Convert object-frame grasp poses into the world frame.

    Args:
        grasps: Grasp poses expressed in the object frame.
        object_to_world: ``(4, 4)`` transform from object frame to world frame.

    Returns:
        Grasp poses expressed in the world frame.
    """
    raise NotImplementedError


def invert_rigid_transform(transform: RigidTransform) -> RigidTransform:
    """Invert a rigid ``(4, 4)`` transformation matrix.

    Args:
        transform: A rigid transformation matrix.

    Returns:
        The inverse rigid transformation.
    """
    raise NotImplementedError
