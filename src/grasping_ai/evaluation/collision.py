"""Point-cloud collision checks for grasp candidates."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
from grasping_ai.perception.pointcloud import build_kdtree
from grasping_ai.robotics.transforms import transform_between_frames

GRASP_POSES_NDIM = int(FLATTENED_YAML_CONFIG.get("grasp.poses_ndim", 3))
NORM_EPS = float(FLATTENED_YAML_CONFIG.get("tolerances.norm_eps", 1e-8))
POINT_CLOUD_NDIM = int(FLATTENED_YAML_CONFIG.get("geometry.point_cloud_ndim", 2))
SE3_MATRIX_SHAPE = tuple(int(v) for v in FLATTENED_YAML_CONFIG.get("grasp.se3_matrix_shape", [4, 4]))
SPATIAL_DIM = int(FLATTENED_YAML_CONFIG.get("geometry.spatial_dim", 3))

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
    if object_point_cloud.ndim != POINT_CLOUD_NDIM or object_point_cloud.shape[1] != SPATIAL_DIM:
        msg = "object_point_cloud must have shape (N, 3)"
        raise ValueError(msg)
    if gripper_point_cloud.ndim != POINT_CLOUD_NDIM or gripper_point_cloud.shape[1] != SPATIAL_DIM:
        msg = "gripper_point_cloud must have shape (M, 3)"
        raise ValueError(msg)
    if clearance < 0:
        msg = "clearance must be non-negative"
        raise ValueError(msg)

    tree = build_kdtree(object_point_cloud)

    def checker(grasp_pose: np.ndarray) -> bool:
        """Check if a given grasp pose is collision-free.

        Args:
            grasp_pose: Grasp pose represented as a (4, 4) transformation.

        Returns:
            True if the grasp is collision-free, False otherwise.
        """
        if grasp_pose.shape != SE3_MATRIX_SHAPE:
            msg = f"grasp_pose must have shape (4, 4), got {grasp_pose.shape}"
            raise ValueError(msg)
        transformed_gripper = transform_between_frames(grasp_pose, gripper_point_cloud)

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
    if grasp_poses.ndim == POINT_CLOUD_NDIM:
        # Single grasp pose shape (4, 4) or flat
        if grasp_poses.shape == (4, 4):
            if checker(grasp_poses):
                return grasp_poses.reshape(1, 4, 4)
            return np.empty((0, 4, 4))
        msg = "grasp_poses must have shape (K, 4, 4) or (4, 4)"
        raise ValueError(msg)

    if grasp_poses.ndim != GRASP_POSES_NDIM or grasp_poses.shape[1:] != SE3_MATRIX_SHAPE:
        msg = "grasp_poses must have shape (K, 4, 4)"
        raise ValueError(msg)

    free_grasps = []
    for i in range(grasp_poses.shape[0]):
        pose = grasp_poses[i]
        if checker(pose):
            free_grasps.append(pose)

    if not free_grasps:
        return np.empty((0, 4, 4))
    return np.stack(free_grasps, axis=0)


def _validate_contact_inputs(
    object_point_cloud: np.ndarray,
    gripper_point_cloud: np.ndarray,
    grasp_pose: np.ndarray,
    contact_clearance: float,
) -> None:
    """Validate inputs for ``generate_analytical_contacts``."""
    if (
        not isinstance(object_point_cloud, np.ndarray)
        or object_point_cloud.ndim != POINT_CLOUD_NDIM
        or object_point_cloud.shape[1] != SPATIAL_DIM
    ):
        msg = "object_point_cloud must have shape (N, 3)"
        raise ValueError(msg)
    if (
        not isinstance(gripper_point_cloud, np.ndarray)
        or gripper_point_cloud.ndim != POINT_CLOUD_NDIM
        or gripper_point_cloud.shape[1] != SPATIAL_DIM
    ):
        msg = "gripper_point_cloud must have shape (M, 3)"
        raise ValueError(msg)
    if not isinstance(grasp_pose, np.ndarray) or grasp_pose.shape != SE3_MATRIX_SHAPE:
        msg = "grasp_pose must have shape (4, 4)"
        raise ValueError(msg)
    if not np.isfinite(object_point_cloud).all():
        msg = "object_point_cloud must contain only finite values"
        raise ValueError(msg)
    if not np.isfinite(gripper_point_cloud).all():
        msg = "gripper_point_cloud must contain only finite values"
        raise ValueError(msg)
    if not np.isfinite(grasp_pose).all():
        msg = "grasp_pose must contain only finite values"
        raise ValueError(msg)
    if contact_clearance < 0 or not np.isfinite(contact_clearance):
        msg = "contact_clearance must be non-negative and finite"
        raise ValueError(msg)


def generate_analytical_contacts(
    object_point_cloud: np.ndarray,
    gripper_point_cloud: np.ndarray,
    grasp_pose: np.ndarray,
    contact_clearance: float,
) -> list[dict[str, np.ndarray]]:
    """Generate analytical contact records between a gripper and an object.

    Transforms the gripper points using the grasp pose, identifies points
    within the contact clearance of the object point cloud, and computes
    contact positions and inward normals.

    Args:
        object_point_cloud: Point cloud of the object with shape ``(N, 3)``.
        gripper_point_cloud: Point cloud of the gripper with shape ``(M, 3)``.
        grasp_pose: Grasp pose represented as a ``(4, 4)`` transformation.
        contact_clearance: Maximum allowed distance for a contact to be detected.

    Returns:
        A list of contact records. Each record is a dictionary mapping ``"position"``
        to a ``(3,)`` position array and ``"normal"`` to an inward unit normal.
    """
    _validate_contact_inputs(object_point_cloud, gripper_point_cloud, grasp_pose, contact_clearance)

    if object_point_cloud.shape[0] == 0 or gripper_point_cloud.shape[0] == 0:
        return []

    transformed_gripper = transform_between_frames(grasp_pose, gripper_point_cloud)

    tree = build_kdtree(object_point_cloud)
    dists, idxs = tree.query(transformed_gripper)

    contacts = []
    # Using absolute tolerance to avoid edge issues with floating point
    for i, dist in enumerate(dists):
        if dist <= contact_clearance + NORM_EPS:
            obj_pt = object_point_cloud[idxs[i]]
            grip_pt = transformed_gripper[i]

            diff = obj_pt - grip_pt
            diff_norm = np.linalg.norm(diff)
            normal = diff / diff_norm if diff_norm > NORM_EPS else np.array([0.0, 0.0, 1.0])

            contacts.append(
                {
                    "position": obj_pt,
                    "normal": normal,
                },
            )

    return contacts
