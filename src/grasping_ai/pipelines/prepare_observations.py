"""Build observation tensors for training pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from grasping_ai.data.pointcloud_dataset import resolve_ycb_object_id
from grasping_ai.perception.pointcloud import normalize_point_cloud
from grasping_ai.robotics.gripper import gripper_point_cloud_from_grid
from grasping_ai.sensors.pointcloud_sensor import merge_point_clouds, sample_point_cloud_from_mesh
from grasping_ai.simulation.ycb import list_ycb_objects

if TYPE_CHECKING:
    from pathlib import Path


def make_observations(  # noqa: PLR0913, PLR0917  # CLI helper; scripts call it positionally
    ycb_root: Path,
    output_dir: Path,
    num_samples: int,
    seed: int,
    merged_objects_name: str,
    merged_objects_normalized_name: str,
    gripper_name: str,
    gripper_grid: dict[str, object],
) -> None:
    """Sample per-object point clouds and a simple gripper finger cloud."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    object_clouds: list[np.ndarray] = []
    for object_id in list_ycb_objects(ycb_root):
        pts = sample_point_cloud_from_mesh(resolve_ycb_object_id(ycb_root, object_id), num_samples, rng)
        np.save(output_dir / f"{object_id}.npy", pts)
        object_clouds.append(pts)

    merged = merge_point_clouds(object_clouds)
    np.save(output_dir / merged_objects_name, merged)
    np.save(
        output_dir / merged_objects_normalized_name,
        normalize_point_cloud(merged).astype(np.float32),
    )
    np.save(output_dir / gripper_name, gripper_point_cloud_from_grid(gripper_grid))
