"""Point-cloud sensor abstraction for environments."""

from __future__ import annotations

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG

from grasping_ai.utils.path_validation import require_path

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]
from loguru import logger

POINT_CLOUD_NDIM = int(FLATTENED_YAML_CONFIG.get("geometry.point_cloud_ndim", 2))
SPATIAL_DIM = int(FLATTENED_YAML_CONFIG.get("geometry.spatial_dim", 3))

if TYPE_CHECKING:
    from collections.abc import Iterator

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
    require_path(observation_path, "observation_path")
    if not observation_path.exists():
        raise FileNotFoundError(f"Observation path '{observation_path}' does not exist")
    try:
        pts = np.load(observation_path)
    except Exception as e:
        raise ValueError(f"Failed to load observation: {e}") from e
    if not isinstance(pts, np.ndarray) or pts.ndim != POINT_CLOUD_NDIM or pts.shape[1] != SPATIAL_DIM:
        raise ValueError("Invalid observation shape: expected (N, 3)")
    if not np.isfinite(pts).all():
        raise ValueError("Observation point cloud contains non-finite values")

    logger.info("Acquired point cloud observation from: {} (shape={})", observation_path, pts.shape)
    return pts

def acquire_point_cloud_stream(observation_paths: list[Path]) -> Iterator[np.ndarray]:
    """Yield point clouds from a sequence of sensor observation files.

    Args:
        observation_paths: Ordered list of observation file paths.

    Yields:
        Point clouds as ``(N, 3)`` numpy arrays.

    Raises:
        TypeError: If ``observation_paths`` is not a list of ``pathlib.Path``
            instances.
    """
    if not isinstance(observation_paths, list) or not all(isinstance(path, Path) for path in observation_paths):
        raise TypeError("observation_paths must be a list of pathlib.Path instances")
    for path in observation_paths:
        yield acquire_point_cloud_from_observation(path)

def merge_point_clouds(clouds: list[np.ndarray]) -> np.ndarray:
    """Merge multiple point clouds into a single observation.

    Args:
        clouds: List of point clouds, each with shape ``(N_i, 3)``.

    Returns:
        A combined point cloud with shape ``(sum(N_i), 3)``.

    Raises:
        TypeError: If ``clouds`` is not a list of numpy arrays.
        ValueError: If any cloud is not a 2D array with three columns.
    """
    if not isinstance(clouds, list) or not all(isinstance(cloud, np.ndarray) for cloud in clouds):
        raise TypeError("clouds must be a list of numpy arrays")
    for cloud in clouds:
        if cloud.ndim != POINT_CLOUD_NDIM or cloud.shape[1] != SPATIAL_DIM:
            raise ValueError("each cloud must have shape (N, 3)")
    if not clouds:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(clouds, axis=0).astype(np.float32)

def sample_point_cloud_from_mesh(mesh_path: Path, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a point cloud from a mesh resource on disk.

    Args:
        mesh_path: Path to a mesh file representing the object surface.
        num_samples: Number of points to sample.
        rng: Random generator used by the sampler.

    Returns:
        A point cloud with shape ``(num_samples, 3)``.
    """
    require_path(mesh_path, "mesh_path")
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file '{mesh_path}' does not exist")
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy random Generator")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Mesh file '{mesh_path}' is empty or invalid")

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if len(triangles) == 0:
        raise ValueError("Mesh has no triangles to sample from")

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    # Compute triangle areas using cross products
    cross_prod = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross_prod, axis=1)

    total_area = np.sum(areas)
    probabilities = np.ones(len(triangles)) / len(triangles) if total_area <= 0 else areas / total_area

    # Sample triangle indices
    sampled_indices = rng.choice(len(triangles), size=num_samples, p=probabilities)

    # Sample barycentric coordinates
    r1 = rng.random(num_samples)
    r2 = rng.random(num_samples)
    sqrt_r1 = np.sqrt(r1)

    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2

    p0 = v0[sampled_indices]
    p1 = v1[sampled_indices]
    p2 = v2[sampled_indices]

    sampled_points = u[:, np.newaxis] * p0 + v[:, np.newaxis] * p1 + w[:, np.newaxis] * p2

    return sampled_points.astype(np.float32)
