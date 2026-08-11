from collections.abc import Callable

import numpy as np

CollisionChecker = Callable[[np.ndarray], bool]


def build_collision_checker(
    object_point_cloud: np.ndarray,
    gripper_point_cloud: np.ndarray,
    clearance: float,
) -> CollisionChecker:
    """Build a callable that checks for collisions between a gripper and object.

    Args:
        object_point_cloud: Point cloud of the object with shape ``(N, 3)``.
        gripper_point_cloud: Point cloud of the gripper with shape ``(M, 3)``.
        clearance: Minimum allowed distance between gripper and object points.

    Returns:
        A callable that maps a grasp pose to ``True`` when the grasp is
        collision-free and ``False`` otherwise.
    """
    raise NotImplementedError


def check_collision(checker: CollisionChecker, grasp_pose: np.ndarray) -> bool:
    """Evaluate a single grasp pose against the collision checker.

    Args:
        checker: Callable returned by ``build_collision_checker``.
        grasp_pose: Grasp pose represented as a ``(4, 4)`` transformation.

    Returns:
        ``True`` if the grasp is collision-free, otherwise ``False``.
    """
    raise NotImplementedError


def filter_collision_free_grasps(checker: CollisionChecker, grasp_poses: np.ndarray) -> np.ndarray:
    """Filter a set of grasps to only those that are collision-free.

    Args:
        checker: Callable returned by ``build_collision_checker``.
        grasp_poses: Grasp poses with shape ``(K, 4, 4)``.

    Returns:
        The subset of collision-free grasp poses.
    """
    raise NotImplementedError
