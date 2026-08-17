"""Point-cloud filtering, normals, and downsampling."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.spatial  # type: ignore[import-untyped]

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG

MIN_NORMAL_NEIGHBORHOOD = int(FLATTENED_YAML_CONFIG.get("geometry.min_normal_neighborhood", 3))
NORM_EPS = float(FLATTENED_YAML_CONFIG.get("tolerances.norm_eps", 1e-8))
POINT_CLOUD_NDIM = int(FLATTENED_YAML_CONFIG.get("geometry.point_cloud_ndim", 2))
SPATIAL_DIM = int(FLATTENED_YAML_CONFIG.get("geometry.spatial_dim", 3))

PointCloud = np.ndarray
FeatureExtractor = Callable[[np.ndarray], np.ndarray]


def sample_point_cloud(points: np.ndarray, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a fixed number of points from a point cloud.

    Args:
        points: Input point cloud with shape ``(N, 3)``.
        num_samples: Target number of points in the sampled output.
        rng: Random generator used to draw samples.

    Returns:
        A point cloud with exactly ``num_samples`` points.
    """
    if not isinstance(points, np.ndarray):
        msg = "points must be a numpy array"
        raise TypeError(msg)
    if points.ndim != POINT_CLOUD_NDIM or points.shape[1] != SPATIAL_DIM:
        msg = f"points shape must be (N, 3), got {points.shape}"
        raise ValueError(msg)
    if points.shape[0] == 0:
        msg = "points must not be empty"
        raise ValueError(msg)
    if not isinstance(num_samples, int) or num_samples <= 0:
        msg = "num_samples must be a positive integer"
        raise ValueError(msg)
    if not isinstance(rng, np.random.Generator):
        msg = "rng must be a numpy random Generator"
        raise TypeError(msg)
    if not np.isfinite(points).all():
        msg = "points must contain only finite values"
        raise ValueError(msg)

    n = points.shape[0]
    replace = n < num_samples
    indices = rng.choice(n, size=num_samples, replace=replace)
    return points[indices]


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """Center and scale a point cloud to a unit canonical frame.

    Args:
        points: Input point cloud with shape ``(N, 3)``.

    Returns:
        A normalized point cloud centered at the origin with unit scale.
    """
    if not isinstance(points, np.ndarray):
        msg = "points must be a numpy array"
        raise TypeError(msg)
    if points.ndim != POINT_CLOUD_NDIM or points.shape[1] != SPATIAL_DIM:
        msg = f"points shape must be (N, 3), got {points.shape}"
        raise ValueError(msg)
    if points.shape[0] == 0:
        msg = "points must not be empty"
        raise ValueError(msg)
    if not np.isfinite(points).all():
        msg = "points must contain only finite values"
        raise ValueError(msg)

    centroid = np.mean(points, axis=0)
    centered = points - centroid
    max_distance = np.max(np.linalg.norm(centered, axis=1))
    if max_distance > 0:
        return centered / max_distance
    return centered


def farthest_point_sampling(points: np.ndarray, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Select ``num_samples`` points using farthest point sampling.

    Args:
        points: Input point cloud with shape ``(N, 3)``.
        num_samples: Number of points to select.
        rng: Random generator used to break ties during sampling.

    Returns:
        Indices into ``points`` representing the farthest point sample.
    """
    if not isinstance(points, np.ndarray):
        msg = "points must be a numpy array"
        raise TypeError(msg)
    if points.ndim != POINT_CLOUD_NDIM or points.shape[1] != SPATIAL_DIM:
        msg = f"points shape must be (N, 3), got {points.shape}"
        raise ValueError(msg)
    if points.shape[0] == 0:
        msg = "points must not be empty"
        raise ValueError(msg)
    if not isinstance(num_samples, int) or num_samples <= 0:
        msg = "num_samples must be a positive integer"
        raise ValueError(msg)
    if not isinstance(rng, np.random.Generator):
        msg = "rng must be a numpy random Generator"
        raise TypeError(msg)
    if not np.isfinite(points).all():
        msg = "points must contain only finite values"
        raise ValueError(msg)

    n = points.shape[0]
    selected_indices = np.zeros(num_samples, dtype=int)

    first_idx = rng.choice(n)
    selected_indices[0] = first_idx

    if num_samples > 1:
        min_dists = np.full(n, np.inf)
        curr_idx = first_idx

        num_unique = min(n, num_samples)
        for i in range(1, num_unique):
            diff = points - points[curr_idx]
            dists = np.sum(diff**2, axis=1)
            min_dists = np.minimum(min_dists, dists)
            curr_idx = int(np.argmax(min_dists))
            selected_indices[i] = curr_idx

        if num_samples > n:
            selected_indices[n:] = rng.choice(n, size=num_samples - n, replace=True)

    return selected_indices


def estimate_point_cloud_normals(points: np.ndarray, neighborhood_size: int) -> np.ndarray:
    """Estimate per-point normals from local neighborhoods.

    Args:
        points: Input point cloud with shape ``(N, 3)``.
        neighborhood_size: Number of neighbors used to fit local tangent planes.

    Returns:
        Per-point normal vectors with shape ``(N, 3)``.
    """
    if not isinstance(points, np.ndarray):
        msg = "points must be a numpy array"
        raise TypeError(msg)
    if points.ndim != POINT_CLOUD_NDIM or points.shape[1] != SPATIAL_DIM:
        msg = f"points shape must be (N, 3), got {points.shape}"
        raise ValueError(msg)
    if points.shape[0] == 0:
        msg = "points must not be empty"
        raise ValueError(msg)
    if not isinstance(neighborhood_size, int) or neighborhood_size <= 0:
        msg = "neighborhood_size must be a positive integer"
        raise ValueError(msg)
    if not np.isfinite(points).all():
        msg = "points must contain only finite values"
        raise ValueError(msg)

    n = points.shape[0]
    k = min(neighborhood_size, n)
    if k < MIN_NORMAL_NEIGHBORHOOD:
        return np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))

    kdtree = scipy.spatial.KDTree(points)
    _, neighbor_indices = kdtree.query(points, k=k)

    normals = np.zeros_like(points)
    cloud_center = np.median(points, axis=0)
    for i in range(n):
        idx = neighbor_indices[i]
        neighbors = points[idx]
        mean = np.mean(neighbors, axis=0)
        centered = neighbors - mean
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        if np.sum(np.abs(eigenvalues)) < NORM_EPS:
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal = eigenvectors[:, 0]
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
        # PCA determines an unoriented axis: ``normal`` and ``-normal`` are
        # equally valid eigenvectors.  Antipodal grasp generation, however,
        # requires consistently outward-facing normals.  Orient each local
        # normal away from the robust cloud center so its sign is stable
        # across LAPACK implementations and random samples.
        if np.dot(normal, points[i] - cloud_center) < 0.0:
            normal = -normal
        normals[i] = normal

    return normals


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample a point cloud using a regular voxel grid.

    Args:
        points: Input point cloud with shape ``(N, 3)``.
        voxel_size: Edge length of each voxel cell.

    Returns:
        The downsampled point cloud.
    """
    if not isinstance(points, np.ndarray):
        msg = "points must be a numpy array"
        raise TypeError(msg)
    if points.ndim != POINT_CLOUD_NDIM or points.shape[1] != SPATIAL_DIM:
        msg = f"points shape must be (N, 3), got {points.shape}"
        raise ValueError(msg)
    if not isinstance(voxel_size, (int, float)) or voxel_size <= 0:
        msg = "voxel_size must be a positive float"
        raise ValueError(msg)
    if not np.isfinite(points).all():
        msg = "points must contain only finite values"
        raise ValueError(msg)

    voxel_indices = np.floor(points / voxel_size).astype(int)
    _, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)

    num_voxels = len(np.unique(inverse_indices))
    downsampled = np.zeros((num_voxels, 3))
    counts = np.zeros(num_voxels)

    np.add.at(downsampled, inverse_indices, points)
    np.add.at(counts, inverse_indices, 1.0)

    return downsampled / counts[:, np.newaxis]


def build_kdtree(points: np.ndarray) -> scipy.spatial.KDTree:
    """Build a spatial index over a point cloud for neighbor queries.

    Args:
        points: Input point cloud with shape ``(N, 3)``.

    Returns:
        A scipy KD-tree over ``points`` usable by other perception functions.
    """
    if not isinstance(points, np.ndarray):
        msg = "points must be a numpy array"
        raise TypeError(msg)
    if points.ndim != POINT_CLOUD_NDIM or points.shape[1] != SPATIAL_DIM:
        msg = f"points shape must be (N, 3), got {points.shape}"
        raise ValueError(msg)
    if not np.isfinite(points).all():
        msg = "points must contain only finite values"
        raise ValueError(msg)

    return scipy.spatial.KDTree(points)
