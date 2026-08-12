# ruff: noqa: I001
import argparse
from pathlib import Path

import open3d as _open3d  # noqa: F401
import numpy as np

from grasping_ai.data.pointcloud_dataset import resolve_ycb_object_id
from grasping_ai.perception.pointcloud import normalize_point_cloud
from grasping_ai.sensors.pointcloud_sensor import (
    merge_point_clouds,
    sample_point_cloud_from_mesh,
)
from grasping_ai.simulation.ycb import list_ycb_objects


def make_observations(ycb_root: Path, output_dir: Path, num_samples: int, seed: int) -> None:
    """Sample per-object point clouds and a simple gripper finger cloud.

    Args:
        ycb_root: Root directory of the raw YCB object set.
        output_dir: Destination directory for the .npy observation files.
        num_samples: Number of points to sample per object.
        seed: Random seed for reproducible sampling.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    object_clouds: list[np.ndarray] = []
    for object_id in list_ycb_objects(ycb_root):
        pts = sample_point_cloud_from_mesh(
            resolve_ycb_object_id(ycb_root, object_id), num_samples, rng
        )
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample per-object observation clouds")
    parser.add_argument("--ycb-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    make_observations(args.ycb_root, args.output_dir, args.num_samples, args.seed)
