from collections.abc import Callable

import numpy as np
from scipy.spatial import KDTree  # type: ignore[import-untyped]

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
    if object_point_cloud.ndim != 2 or object_point_cloud.shape[1] != 3:
        raise ValueError("object_point_cloud must have shape (N, 3)")
    if gripper_point_cloud.ndim != 2 or gripper_point_cloud.shape[1] != 3:
        raise ValueError("gripper_point_cloud must have shape (M, 3)")
    if clearance < 0:
        raise ValueError("clearance must be non-negative")

    tree = KDTree(object_point_cloud)

    def checker(grasp_pose: np.ndarray) -> bool:
        if grasp_pose.shape != (4, 4):
            raise ValueError(
                f"grasp_pose must have shape (4, 4), got {grasp_pose.shape}"
            )
        rot = grasp_pose[:3, :3]
        trans = grasp_pose[:3, 3]
        transformed_gripper = gripper_point_cloud @ rot.T + trans

        dists, _ = tree.query(transformed_gripper)
        return not bool(np.any(dists < clearance))

    return checker


def check_collision(checker: CollisionChecker, grasp_pose: np.ndarray) -> bool:
    """Evaluate a single grasp pose against the collision checker.

    Args:
        checker: Callable returned by ``build_collision_checker``.
        grasp_pose: Grasp pose represented as a ``(4, 4)`` transformation.

    Returns:
        ``True`` if the grasp is collision-free, otherwise ``False``.
    """
    return checker(grasp_pose)


def filter_collision_free_grasps(checker: CollisionChecker, grasp_poses: np.ndarray) -> np.ndarray:
    """Filter a set of grasps to only those that are collision-free.

    Args:
        checker: Callable returned by ``build_collision_checker``.
        grasp_poses: Grasp poses with shape ``(K, 4, 4)``.

    Returns:
        The subset of collision-free grasp poses.
    """
    if grasp_poses.ndim == 2:
        # Single grasp pose shape (4, 4) or flat
        if grasp_poses.shape == (4, 4):
            if checker(grasp_poses):
                return grasp_poses.reshape(1, 4, 4)
            return np.empty((0, 4, 4))
        raise ValueError("grasp_poses must have shape (K, 4, 4) or (4, 4)")

    if grasp_poses.ndim != 3 or grasp_poses.shape[1:] != (4, 4):
        raise ValueError("grasp_poses must have shape (K, 4, 4)")

    free_grasps = []
    for i in range(grasp_poses.shape[0]):
        pose = grasp_poses[i]
        if checker(pose):
            free_grasps.append(pose)

    if not free_grasps:
        return np.empty((0, 4, 4))
    return np.stack(free_grasps, axis=0)
