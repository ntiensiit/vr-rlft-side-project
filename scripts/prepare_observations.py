"""Build observation tensors for training pipelines."""

from __future__ import annotations

from pathlib import Path

import hydra
import open3d as _open3d  # noqa: F401
import numpy as np
from omegaconf import DictConfig

from grasping_ai.config.config import SCRIPTS_CONFIG_PATH, config_get, config_value
from grasping_ai.data.pointcloud_dataset import resolve_ycb_object_id
from grasping_ai.perception.pointcloud import normalize_point_cloud
from grasping_ai.sensors.pointcloud_sensor import (
    merge_point_clouds,
    sample_point_cloud_from_mesh,
)
from grasping_ai.simulation.ycb import list_ycb_objects


def _linspace_axis(axis_cfg: object) -> np.ndarray:
    if not isinstance(axis_cfg, dict):
        msg = "observations.gripper_grid axis must be a mapping with start, stop, and count"
        raise TypeError(msg)
    return np.linspace(float(axis_cfg["start"]), float(axis_cfg["stop"]), int(axis_cfg["count"]))


def make_observations(
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

    x = _linspace_axis(gripper_grid["x"])
    y = _linspace_axis(gripper_grid["y"])
    z = _linspace_axis(gripper_grid["z"])
    grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
    np.save(output_dir / gripper_name, grid.astype(np.float32))


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/prepare_observations")
def main(cfg: DictConfig) -> None:
    output_cfg = config_get(cfg, "observations", "output")
    gripper_grid = config_get(cfg, "observations", "gripper_grid")
    if not isinstance(output_cfg, dict) or not isinstance(gripper_grid, dict):
        msg = "observations.output and observations.gripper_grid must be mappings"
        raise TypeError(msg)
    make_observations(
        config_value(cfg, "paths", "ycb_root", value_type=Path, required=True),
        config_value(cfg, "paths", "observations", value_type=Path, required=True),
        config_value(cfg, "observations", "num_samples", value_type=int),
        config_value(cfg, "observations", "seed", value_type=int),
        str(output_cfg["merged_objects"]),
        str(output_cfg["merged_objects_normalized"]),
        str(output_cfg["gripper"]),
        gripper_grid,
    )


if __name__ == "__main__":
    main()
