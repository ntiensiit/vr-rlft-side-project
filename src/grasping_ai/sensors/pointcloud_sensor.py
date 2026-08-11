from collections.abc import Iterator
from pathlib import Path

import numpy as np

PointCloudBatch = np.ndarray


def acquire_point_cloud_from_observation(observation_path: Path) -> np.ndarray:
    """Load a single point cloud observation from a sensor data file.

    Args:
        observation_path: Path to a serialized sensor observation.

    Returns:
        A point cloud with shape ``(N, 3)``.

    Raises:
        FileNotFoundError: If ``observation_path`` does not exist.
    """
    if not isinstance(observation_path, Path):
        raise TypeError("observation_path must be a pathlib.Path instance")
    if not observation_path.exists():
        raise FileNotFoundError(f"Observation path '{observation_path}' does not exist")
    try:
        pts = np.load(observation_path)
    except Exception as e:
        raise ValueError(f"Failed to load observation: {e}") from e
    if not isinstance(pts, np.ndarray) or pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("Invalid observation shape: expected (N, 3)")
    if not np.isfinite(pts).all():
        raise ValueError("Observation point cloud contains non-finite values")
    return pts


def acquire_point_cloud_stream(observation_paths: list[Path]) -> Iterator[np.ndarray]:
    """Yield point clouds from a sequence of sensor observation files.

    Args:
        observation_paths: Ordered list of observation file paths.

    Yields:
        Point clouds as ``(N, 3)`` numpy arrays.
    """
    raise NotImplementedError


def merge_point_clouds(clouds: list[np.ndarray]) -> np.ndarray:
    """Merge multiple point clouds into a single observation.

    Args:
        clouds: List of point clouds, each with shape ``(N_i, 3)``.

    Returns:
        A combined point cloud with shape ``(sum(N_i), 3)``.
    """
    raise NotImplementedError


def sample_point_cloud_from_mesh(
    mesh_path: Path, num_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample a point cloud from a mesh resource on disk.

    Args:
        mesh_path: Path to a mesh file representing the object surface.
        num_samples: Number of points to sample.
        rng: Random generator used by the sampler.

    Returns:
        A point cloud with shape ``(num_samples, 3)``.
    """
    raise NotImplementedError
