"""Build observation tensors for training pipelines."""

from __future__ import annotations

from pathlib import Path

import hydra
import open3d as _open3d  # noqa: F401
import numpy as np
from omegaconf import DictConfig

from grasping_ai.config.config import SCRIPTS_CONFIG_PATH, config_value
from grasping_ai.data.pointcloud_dataset import resolve_ycb_object_id
from grasping_ai.perception.pointcloud import normalize_point_cloud
from grasping_ai.sensors.pointcloud_sensor import (
    merge_point_clouds,
    sample_point_cloud_from_mesh,
)
from grasping_ai.simulation.ycb import list_ycb_objects


def make_observations(ycb_root: Path, output_dir: Path, num_samples: int, seed: int) -> None:
    """Sample per-object point clouds and a simple gripper finger cloud."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    object_clouds: list[np.ndarray] = []
    for object_id in list_ycb_objects(ycb_root):
        pts = sample_point_cloud_from_mesh(resolve_ycb_object_id(ycb_root, object_id), num_samples, rng)
        np.save(output_dir / f"{object_id}.npy", pts)
        object_clouds.append(pts)

    merged = merge_point_clouds(object_clouds)
    np.save(output_dir / "merged_objects.npy", merged)
    np.save(
        output_dir / "merged_objects_normalized.npy",
        normalize_point_cloud(merged).astype(np.float32),
    )

    x = np.linspace(-0.03, 0.03, 4)
    y = np.linspace(-0.02, 0.02, 3)
    z = np.linspace(-0.04, 0.04, 5)
    grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
    np.save(output_dir / "gripper.npy", grid.astype(np.float32))


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/prepare_observations")
def main(cfg: DictConfig) -> None:
    make_observations(
        config_value(cfg, "paths", "ycb_root", value_type=Path, required=True),
        config_value(cfg, "paths", "observations", value_type=Path, required=True),
        config_value(cfg, "observations", "num_samples", value_type=int),
        config_value(cfg, "observations", "seed", value_type=int),
    )


if __name__ == "__main__":
    main()
