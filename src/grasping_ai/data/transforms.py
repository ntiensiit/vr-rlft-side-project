import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

SampleTransform = Callable[
    [np.ndarray, np.ndarray | None, np.ndarray | None],
    tuple[np.ndarray, np.ndarray | None, np.ndarray | None],
]


def make_random_rotation_jitter(rng: np.random.Generator) -> SampleTransform:
    """Build a transform that applies a random SO(3) rotation to a sample.

    Args:
        rng: NumPy random generator used for sampling rotations.

    Returns:
        A callable transform operating on point cloud, grasp poses, and scores.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy random Generator")

    def transform(
        points: np.ndarray,
        grasp_poses: np.ndarray | None,
        scores: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        if not isinstance(points, np.ndarray):
            raise TypeError("points must be a numpy array")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points shape must be (N, 3), got {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("points must contain only finite values")

        q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        d = np.linalg.det(q)
        q[:, 0] *= d
        rot_mat = q

        points_rot = points @ rot_mat.T

        grasp_poses_rot = None
        if grasp_poses is not None:
            if not isinstance(grasp_poses, np.ndarray):
                raise TypeError("grasp_poses must be a numpy array")
            if grasp_poses.ndim != 3 or grasp_poses.shape[1:] != (4, 4):
                raise ValueError(f"grasp_poses must have shape (M, 4, 4), got {grasp_poses.shape}")
            if not np.isfinite(grasp_poses).all():
                raise ValueError("grasp_poses must contain only finite values")
            t_rot = np.eye(4)
            t_rot[:3, :3] = rot_mat
            grasp_poses_rot = np.zeros_like(grasp_poses)
            for i in range(len(grasp_poses)):
                grasp_poses_rot[i] = t_rot @ grasp_poses[i]

        if scores is not None:
            if not isinstance(scores, np.ndarray):
                raise TypeError("scores must be a numpy array")
            if not np.isfinite(scores).all():
                raise ValueError("scores must contain only finite values")

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
        raise TypeError("rng must be a numpy random Generator")
    if not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError("scale must be a positive float")
    if not np.isfinite(scale):
        raise ValueError("scale must be finite")

    def transform(
        points: np.ndarray,
        grasp_poses: np.ndarray | None,
        scores: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        if not isinstance(points, np.ndarray):
            raise TypeError("points must be a numpy array")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points shape must be (N, 3), got {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("points must contain only finite values")

        t = rng.uniform(-scale, scale, size=3)
        points_trans = points + t

        grasp_poses_trans = None
        if grasp_poses is not None:
            if not isinstance(grasp_poses, np.ndarray):
                raise TypeError("grasp_poses must be a numpy array")
            if grasp_poses.ndim != 3 or grasp_poses.shape[1:] != (4, 4):
                raise ValueError(f"grasp_poses must have shape (M, 4, 4), got {grasp_poses.shape}")
            if not np.isfinite(grasp_poses).all():
                raise ValueError("grasp_poses must contain only finite values")
            grasp_poses_trans = np.copy(grasp_poses)
            grasp_poses_trans[:, :3, 3] += t

        if scores is not None:
            if not isinstance(scores, np.ndarray):
                raise TypeError("scores must be a numpy array")
            if not np.isfinite(scores).all():
                raise ValueError("scores must contain only finite values")

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
        for t in transforms:
            if not callable(t):
                raise TypeError("All transforms in compose_transforms must be callable")
            points, grasp_poses, scores = t(points, grasp_poses, scores)
        return points, grasp_poses, scores

    return composed


def save_grasp_dataset_index(
    dataset_root: Path, entries: list[dict[str, str]], filename: str = "index.json"
) -> None:
    """Persist a dataset index file describing available records.

    Args:
        dataset_root: Root directory under which the index file is written.
        entries: List of metadata entries describing dataset records.
        filename: Name of the index file written under ``dataset_root``.
    """
    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be a pathlib.Path instance")
    if not isinstance(entries, list):
        raise TypeError("entries must be a list of dictionaries")
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("All elements in entries must be dictionaries")

    dataset_root.mkdir(parents=True, exist_ok=True)
    index_path = dataset_root / filename
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=4)
    except Exception as e:
        raise ValueError(f"Failed to write dataset index: {e}") from e
