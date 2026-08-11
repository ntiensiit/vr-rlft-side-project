import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np

from grasping_ai.data.pointcloud_dataset import (
    discover_dataset_files,
    generate_analytical_grasps,
    resolve_ycb_object_id,
)
from grasping_ai.data.transforms import save_grasp_dataset_index
from grasping_ai.perception.pointcloud import estimate_point_cloud_normals
from grasping_ai.sensors.pointcloud_sensor import sample_point_cloud_from_mesh
from grasping_ai.simulation.ycb import list_ycb_objects


def prepare_data_index(dataset_root: Path, output_index_path: Path) -> None:
    """Discover dataset files and write a dataset index file.

    Args:
        dataset_root: Root directory containing raw dataset records.
        output_index_path: Destination path for the generated index file.
    """
    records = discover_dataset_files(dataset_root)
    entries = [{"path": str(record)} for record in records]
    save_grasp_dataset_index(output_index_path.parent, entries)


def generate_synthetic_dataset(
    ycb_root: Path,
    output_dir: Path,
    num_samples: int,
    num_grasps: int,
    gripper_width: float,
    seed: int,
) -> None:
    """Generate synthetic grasp dataset from YCB meshes.

    Args:
        ycb_root: Path to YCB raw data directory.
        output_dir: Output directory where .npy files will be saved.
        num_samples: Number of points to sample per mesh.
        num_grasps: Number of analytical grasps to generate.
        gripper_width: Maximum gripper width.
        seed: Random seed for reproducibility.
    """
    if not ycb_root.is_dir():
        raise FileNotFoundError(f"YCB root directory '{ycb_root}' does not exist.")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    object_names = list_ycb_objects(ycb_root)
    for name in object_names:
        try:
            mesh_path = resolve_ycb_object_id(ycb_root, name)
            if not mesh_path.is_file():
                # Try finding any .obj or .ply file in case directory was returned
                candidates = list(mesh_path.rglob("*.obj")) + list(mesh_path.rglob("*.ply"))
                if candidates:
                    mesh_path = candidates[0]
                else:
                    print(f"Skipping {name}: no mesh files found.")
                    continue

            # Sample point cloud
            points = sample_point_cloud_from_mesh(mesh_path, num_samples, rng)

            # Estimate normals
            normals = estimate_point_cloud_normals(points, neighborhood_size=30)

            # Generate grasps
            grasps = generate_analytical_grasps(points, normals, num_grasps, gripper_width, rng)

            # Create sample dict
            sample = {
                "point_cloud": points.astype(np.float32),
                "grasp_poses": grasps.astype(np.float32),
                "scores": None,
                "object_id": name,
            }

            output_file = output_dir / f"{name}.npy"
            np.save(output_file, cast(Any, sample))

        except Exception as e:
            print(f"Failed to generate synthetic data for '{name}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare grasp dataset or generate synthetic data")
    parser.add_argument("--mode", type=str, choices=["index", "synthetic"], default="index")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-index", type=Path, required=True)

    # Synthetic arguments
    parser.add_argument("--ycb-root", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--num-grasps", type=int, default=64)
    parser.add_argument("--gripper-width", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Resolve output directory for synthetic or indexing
    target_dir = args.output_dir if args.output_dir is not None else args.dataset_root

    if args.mode == "synthetic":
        if args.ycb_root is None:
            parser.error("--ycb-root is required when --mode is synthetic")
        if target_dir is None:
            parser.error("--output-dir or --dataset-root is required when --mode is synthetic")

        generate_synthetic_dataset(
            ycb_root=args.ycb_root,
            output_dir=target_dir,
            num_samples=args.num_samples,
            num_grasps=args.num_grasps,
            gripper_width=args.gripper_width,
            seed=args.seed,
        )
        # Automatically index the generated files
        prepare_data_index(target_dir, args.output_index)
    else:
        if target_dir is None:
            parser.error("--dataset-root or --output-dir is required when --mode is index")
        prepare_data_index(target_dir, args.output_index)
