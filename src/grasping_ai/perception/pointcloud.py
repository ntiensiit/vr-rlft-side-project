from collections.abc import Callable

import numpy as np

PointCloud = np.ndarray
FeatureExtractor = Callable[[np.ndarray], np.ndarray]


def sample_point_cloud(
    points: np.ndarray, num_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample a fixed number of points from a point cloud.

    Args:
        points: Input point cloud with shape ``(N, 3)``.
        num_samples: Target number of points in the sampled output.
        rng: Random generator used to draw samples.

    Returns:
        A point cloud with exactly ``num_samples`` points.
    """
    raise NotImplementedError


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """Center and scale a point cloud to a unit canonical frame.

    Args:
        points: Input point cloud with shape ``(N, 3)``.

    Returns:
        A normalized point cloud centered at the origin with unit scale.
    """
    raise NotImplementedError


def farthest_point_sampling(
    points: np.ndarray, num_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Select ``num_samples`` points using farthest point sampling.

    Args:
        points: Input point cloud with shape ``(N, 3)``.
        num_samples: Number of points to select.
        rng: Random generator used to break ties during sampling.

    Returns:
        Indices into ``points`` representing the farthest point sample.
    """
    raise NotImplementedError


def estimate_point_cloud_normals(points: np.ndarray, neighborhood_size: int) -> np.ndarray:
    """Estimate per-point normals from local neighborhoods.

    Args:
        points: Input point cloud with shape ``(N, 3)``.
        neighborhood_size: Number of neighbors used to fit local tangent planes.

    Returns:
        Per-point normal vectors with shape ``(N, 3)``.
    """
    raise NotImplementedError


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample a point cloud using a regular voxel grid.

    Args:
        points: Input point cloud with shape ``(N, 3)``.
        voxel_size: Edge length of each voxel cell.

    Returns:
        The downsampled point cloud.
    """
    raise NotImplementedError


def build_kdtree(points: np.ndarray) -> object:
    """Build a spatial index over a point cloud for neighbor queries.

    Args:
        points: Input point cloud with shape ``(N, 3)``.

    Returns:
        An opaque spatial-index object usable by other perception functions.
    """
    raise NotImplementedError
