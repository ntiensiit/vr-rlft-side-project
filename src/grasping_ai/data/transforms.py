"""Dataset transforms for point clouds and grasp poses."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
from grasping_ai.perception.geometry import (
    apply_transform,
    make_transform,
    rotation_matrix_from_axis_angle,
    rotation_matrix_to_axis_angle,
)
from grasping_ai.robotics.transforms import transform_grasp_pose
from grasping_ai.utils.path_validation import require_path

GRASP_POSES_NDIM = int(FLATTENED_YAML_CONFIG.get("grasp.poses_ndim", 3))
POINT_CLOUD_NDIM = int(FLATTENED_YAML_CONFIG.get("geometry.point_cloud_ndim", 2))
SE3_MATRIX_SHAPE = tuple(int(v) for v in FLATTENED_YAML_CONFIG.get("grasp.se3_matrix_shape", [4, 4]))
SPATIAL_DIM = int(FLATTENED_YAML_CONFIG.get("geometry.spatial_dim", 3))

if TYPE_CHECKING:
    from pathlib import Path

SampleTransform = Callable[
    [np.ndarray, np.ndarray | None, np.ndarray | None],
    tuple[np.ndarray, np.ndarray | None, np.ndarray | None],
]


def _validate_points(points: np.ndarray) -> None:
    """Validate a point-cloud array for sample transforms."""
    if not isinstance(points, np.ndarray):
        msg = "points must be a numpy array"
        raise TypeError(msg)
    if points.ndim != POINT_CLOUD_NDIM or points.shape[1] != SPATIAL_DIM:
        msg = f"points shape must be (N, 3), got {points.shape}"
        raise ValueError(msg)
    if not np.isfinite(points).all():
        msg = "points must contain only finite values"
        raise ValueError(msg)


def _validate_grasp_poses(grasp_poses: np.ndarray) -> None:
    """Validate a grasp-pose array for sample transforms."""
    if not isinstance(grasp_poses, np.ndarray):
        msg = "grasp_poses must be a numpy array"
        raise TypeError(msg)
    if grasp_poses.ndim != GRASP_POSES_NDIM or grasp_poses.shape[1:] != SE3_MATRIX_SHAPE:
        msg = f"grasp_poses must have shape (M, 4, 4), got {grasp_poses.shape}"
        raise ValueError(msg)
    if not np.isfinite(grasp_poses).all():
        msg = "grasp_poses must contain only finite values"
        raise ValueError(msg)


def _validate_scores(scores: np.ndarray) -> None:
    """Validate a score array for sample transforms."""
    if not isinstance(scores, np.ndarray):
        msg = "scores must be a numpy array"
        raise TypeError(msg)
    if not np.isfinite(scores).all():
        msg = "scores must contain only finite values"
        raise ValueError(msg)


def make_random_rotation_jitter(rng: np.random.Generator) -> SampleTransform:
    """Build a transform that applies a random SO(3) rotation to a sample.

    Args:
        rng: NumPy random generator used for sampling rotations.

    Returns:
        A callable transform operating on point cloud, grasp poses, and scores.
    """
    if not isinstance(rng, np.random.Generator):
        msg = "rng must be a numpy random Generator"
        raise TypeError(msg)

    def transform(
        points: np.ndarray,
        grasp_poses: np.ndarray | None,
        scores: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Apply random rotation jitter to the point cloud and grasps.

        Args:
            points: Array of point cloud coordinates.
            grasp_poses: Optional array of SE(3) grasp poses.
            scores: Optional array of grasp scores.

        Returns:
            A tuple of (rotated points, rotated grasp poses, scores).
        """
        _validate_points(points)

        q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        d = np.linalg.det(q)
        q[:, 0] *= d
        axis, angle = rotation_matrix_to_axis_angle(q)
        rot_mat = rotation_matrix_from_axis_angle(axis, angle)

        points_rot = apply_transform(points, make_transform(rot_mat, np.zeros(3)))

        grasp_poses_rot = None
        if grasp_poses is not None:
            _validate_grasp_poses(grasp_poses)
            t_rot = make_transform(rot_mat, np.zeros(3))
            grasp_poses_rot = np.zeros_like(grasp_poses)
            for i in range(len(grasp_poses)):
                grasp_poses_rot[i] = transform_grasp_pose(t_rot, grasp_poses[i])

        if scores is not None:
            _validate_scores(scores)

        return points_rot, grasp_poses_rot, scores

    return transform


def make_translation_jitter(rng: np.random.Generator, scale: float) -> SampleTransform:
    """Build a transform that applies a small random translation to a sample.

    Args:
        rng: NumPy random generator.
        scale: Maximum magnitude of the translation offset.

    Returns:
        A callable transform operating on point cloud, grasp poses, and scores.
    """
    if not isinstance(rng, np.random.Generator):
        msg = "rng must be a numpy random Generator"
        raise TypeError(msg)
    if not isinstance(scale, (int, float)) or scale <= 0:
        msg = "scale must be a positive float"
        raise ValueError(msg)
    if not np.isfinite(scale):
        msg = "scale must be finite"
        raise ValueError(msg)

    def transform(
        points: np.ndarray,
        grasp_poses: np.ndarray | None,
        scores: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Apply translation jitter to the point cloud and grasps.

        Args:
            points: Array of point cloud coordinates.
            grasp_poses: Optional array of SE(3) grasp poses.
            scores: Optional array of grasp scores.

        Returns:
            A tuple of (translated points, translated grasp poses, scores).
        """
        _validate_points(points)

        t = rng.uniform(-scale, scale, size=3)
        points_trans = points + t

        grasp_poses_trans = None
        if grasp_poses is not None:
            _validate_grasp_poses(grasp_poses)
            grasp_poses_trans = np.copy(grasp_poses)
            grasp_poses_trans[:, :3, 3] += t

        if scores is not None:
            _validate_scores(scores)

        return points_trans, grasp_poses_trans, scores

    return transform


def compose_transforms(*transforms: SampleTransform) -> SampleTransform:
    """Compose multiple sample transforms into a single callable.

    Args:
        transforms: Ordered sample transforms to apply sequentially.

    Returns:
        A callable that applies each transform in order to a sample.
    """

    def composed(
        points: np.ndarray,
        grasp_poses: np.ndarray | None,
        scores: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Apply composed transformations to the sample.

        Args:
            points: Array of point cloud coordinates.
            grasp_poses: Optional array of SE(3) grasp poses.
            scores: Optional array of grasp scores.

        Returns:
            A tuple of (transformed points, transformed grasp poses, transformed scores).
        """
        for t in transforms:
            if not callable(t):
                msg = "All transforms in compose_transforms must be callable"
                raise TypeError(msg)
            points, grasp_poses, scores = t(points, grasp_poses, scores)
        return points, grasp_poses, scores

    return composed


def save_grasp_dataset_index(dataset_root: Path, entries: list[dict[str, str]], filename: str = "index.json") -> None:
    """Persist a dataset index file describing available records.

    Args:
        dataset_root: Root directory under which the index file is written.
        entries: List of metadata entries describing dataset records.
        filename: Name of the index file written under ``dataset_root``.
    """
    require_path(dataset_root, "dataset_root")
    if not isinstance(entries, list):
        msg = "entries must be a list of dictionaries"
        raise TypeError(msg)
    for entry in entries:
        if not isinstance(entry, dict):
            msg = "All elements in entries must be dictionaries"
            raise TypeError(msg)

    dataset_root.mkdir(parents=True, exist_ok=True)
    index_path = dataset_root / filename
    try:
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(entries, f, indent=4)
    except Exception as e:
        msg = f"Failed to write dataset index: {e}"
        raise ValueError(msg) from e
