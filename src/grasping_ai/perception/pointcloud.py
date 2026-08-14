from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.spatial  # type: ignore[import-untyped]

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
        raise TypeError("points must be a numpy array")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points shape must be (N, 3), got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError("points must not be empty")
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy random Generator")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite values")

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
        raise TypeError("points must be a numpy array")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points shape must be (N, 3), got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError("points must not be empty")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite values")

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
        raise TypeError("points must be a numpy array")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points shape must be (N, 3), got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError("points must not be empty")
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy random Generator")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite values")

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
        raise TypeError("points must be a numpy array")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points shape must be (N, 3), got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError("points must not be empty")
    if not isinstance(neighborhood_size, int) or neighborhood_size <= 0:
        raise ValueError("neighborhood_size must be a positive integer")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite values")

    n = points.shape[0]
    k = min(neighborhood_size, n)
    if k < 3:
        return np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))

    kdtree = scipy.spatial.KDTree(points)
    _, neighbor_indices = kdtree.query(points, k=k)

    normals = np.zeros_like(points)
    for i in range(n):
        idx = neighbor_indices[i]
        neighbors = points[idx]
        mean = np.mean(neighbors, axis=0)
        centered = neighbors - mean
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        if np.sum(np.abs(eigenvalues)) < 1e-8:
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal = eigenvectors[:, 0]
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
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
        raise TypeError("points must be a numpy array")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points shape must be (N, 3), got {points.shape}")
    if not isinstance(voxel_size, (int, float)) or voxel_size <= 0:
        raise ValueError("voxel_size must be a positive float")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite values")

    voxel_indices = np.floor(points / voxel_size).astype(int)
    _, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)

    num_voxels = len(np.unique(inverse_indices))
    downsampled = np.zeros((num_voxels, 3))
    counts = np.zeros(num_voxels)

    np.add.at(downsampled, inverse_indices, points)
    np.add.at(counts, inverse_indices, 1.0)

    return downsampled / counts[:, np.newaxis]


def build_kdtree(points: np.ndarray) -> Any:
    """Build a spatial index over a point cloud for neighbor queries.

    Args:
        points: Input point cloud with shape ``(N, 3)``.

    Returns:
        An opaque spatial-index object usable by other perception functions.
    """
    if not isinstance(points, np.ndarray):
        raise TypeError("points must be a numpy array")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points shape must be (N, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite values")

    return scipy.spatial.KDTree(points)
