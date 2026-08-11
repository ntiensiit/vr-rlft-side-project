from collections.abc import Callable

import numpy as np
import pytransform3d.transformations as pt

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
    if not isinstance(source_to_target, np.ndarray) or source_to_target.shape != (4, 4):
        raise ValueError("source_to_target must be a (4, 4) numpy array")
    if not isinstance(point_in_source, np.ndarray):
        raise TypeError("point_in_source must be a numpy array")

    if point_in_source.shape == (3,):
        pt_hom = np.append(point_in_source, 1.0)
        return (source_to_target @ pt_hom)[:3]
    if len(point_in_source.shape) == 2 and point_in_source.shape[1] == 3:
        n = point_in_source.shape[0]
        pts_hom = np.hstack((point_in_source, np.ones((n, 1))))
        return (source_to_target @ pts_hom.T).T[:, :3]
    raise ValueError("point_in_source must have shape (3,) or (N, 3)")


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
    if not isinstance(grasp_to_world, np.ndarray) or grasp_to_world.shape != (4, 4):
        raise ValueError("grasp_to_world must be a (4, 4) numpy array")
    if not isinstance(gripper_to_grasp, np.ndarray) or gripper_to_grasp.shape != (4, 4):
        raise ValueError("gripper_to_grasp must be a (4, 4) numpy array")

    return grasp_to_world @ gripper_to_grasp


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
    if not isinstance(object_to_world, np.ndarray) or object_to_world.shape != (4, 4):
        raise ValueError("object_to_world must be a (4, 4) numpy array")
    if not isinstance(grasps, np.ndarray):
        raise TypeError("grasps must be a numpy array")

    if grasps.shape == (4, 4):
        return object_to_world @ grasps
    if len(grasps.shape) == 3 and grasps.shape[1:] == (4, 4):
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
    if not isinstance(transform, np.ndarray) or transform.shape != (4, 4):
        raise ValueError("transform must be a (4, 4) numpy array")
    return pt.invert_transform(transform)
