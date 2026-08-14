# ruff: noqa: I001
import argparse
from pathlib import Path
from typing import Any, cast

import open3d as _open3d  # noqa: F401
import numpy as np

from grasping_ai.data.pointcloud_dataset import (
    discover_dataset_files,
    generate_analytical_grasps,
    resolve_ycb_object_id,
)
from grasping_ai.data.transforms import save_grasp_dataset_index
from grasping_ai.perception.pointcloud import (
    estimate_point_cloud_normals,
    farthest_point_sampling,
    sample_point_cloud,
    voxel_downsample,
)
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
    save_grasp_dataset_index(output_index_path.parent, entries, output_index_path.name)


def generate_synthetic_dataset(
    ycb_root: Path,
    output_dir: Path,
    num_samples: int,
    num_grasps: int,
    gripper_width: float,
    seed: int,
    required_objects: list[str] | None = None,
) -> None:
    """Generate synthetic grasp dataset from YCB meshes.

    Args:
        ycb_root: Path to YCB raw data directory.
        output_dir: Output directory where .npy files will be saved.
        num_samples: Number of points to sample per mesh.
        num_grasps: Number of analytical grasps to generate.
        gripper_width: Maximum gripper width.
        seed: Random seed for reproducibility.
        required_objects: Optional list of object identifiers whose generation
            must succeed. If any listed object fails to produce a record (no
            strict or relaxed grasps, missing mesh, exception), the function
            raises. Objects not in this list are still skipped with a printed
            message when they fail.
    """
    if not ycb_root.is_dir():
        raise FileNotFoundError(f"YCB root directory '{ycb_root}' does not exist.")

    required_set = set(required_objects or [])
    failures: list[str] = []

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
                    if name in required_set:
                        failures.append(f"{name}: no mesh files found")
                    continue

            # Sample point cloud, then refine with perception helpers
            oversample_count = max(num_samples, min(num_samples * 2, num_samples + 128))
            raw_points = sample_point_cloud_from_mesh(mesh_path, oversample_count, rng)
            if raw_points.shape[0] >= num_samples:
                fps_indices = farthest_point_sampling(raw_points, num_samples, rng)
                points = raw_points[fps_indices]
            else:
                points = sample_point_cloud(raw_points, num_samples, rng)
            points = voxel_downsample(points, voxel_size=1e-5)
            if points.shape[0] != num_samples:
                points = sample_point_cloud(points, num_samples, rng)

            # Estimate normals
            normals = estimate_point_cloud_normals(points, neighborhood_size=30)

            # Generate grasps
            grasps = generate_analytical_grasps(points, normals, num_grasps, gripper_width, rng)
            if grasps.shape[0] == 0:
                # Fallback: retry with the relaxed antipodal search so that
                # objects producing no strict grasps are not saved unusable.
                grasps = generate_analytical_grasps(
                    points,
                    normals,
                    num_grasps,
                    gripper_width,
                    rng,
                    allow_relaxed=True,
                )
            if grasps.shape[0] == 0:
                print(f"Skipping {name}: no valid grasps found.")
                if name in required_set:
                    failures.append(f"{name}: no valid grasps")
                continue

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
            if name in required_set:
                failures.append(f"{name}: {e}")

    if failures:
        joined = "; ".join(failures)
        raise RuntimeError(f"Required YCB objects failed to generate synthetic data: {joined}")


if __name__ == "__main__":
    from grasping_ai.config.yaml_loader import (
        config_get,
        config_path,
        config_str_list,
        load_project_yaml_config,
        parse_config_dir_from_argv,
    )

    config_dir = parse_config_dir_from_argv()
    cfg = load_project_yaml_config(config_dir, "base", "data")
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-dir", type=Path, default=config_dir)
    parser = argparse.ArgumentParser(
        description="Prepare grasp dataset or generate synthetic data",
        parents=[pre_parser],
    )
    parser.add_argument("--mode", type=str, choices=["index", "synthetic"], default="index")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=config_path(cfg, "paths", "dataset_root"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--output-index",
        type=Path,
        default=config_path(cfg, "paths", "output_index"),
    )

    parser.add_argument(
        "--ycb-root",
        type=Path,
        default=config_path(cfg, "paths", "ycb_root"),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=int(config_get(cfg, "synthetic", "num_samples")),
    )
    parser.add_argument(
        "--num-grasps",
        type=int,
        default=int(config_get(cfg, "synthetic", "num_grasps")),
    )
    parser.add_argument(
        "--gripper-width",
        type=float,
        default=float(config_get(cfg, "synthetic", "gripper_width")),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(config_get(cfg, "synthetic", "seed")),
    )
    parser.add_argument(
        "--required-objects",
        type=str,
        nargs="+",
        default=config_str_list(cfg, "objects", "ids"),
        help=(
            "Space-separated list of YCB object identifiers whose generation "
            "must succeed; missing objects cause a non-zero exit code. "
            "Objects not in this list are skipped with a printed warning."
        ),
    )

    args = parser.parse_args()
    if args.output_index is None:
        parser.error(
            "--output-index is required (set in configs/data.yaml paths.output_index "
            "or pass explicitly)"
        )

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
            required_objects=args.required_objects,
        )
        # Automatically index the generated files
        prepare_data_index(target_dir, args.output_index)
    else:
        if target_dir is None:
            parser.error("--dataset-root or --output-dir is required when --mode is index")
        prepare_data_index(target_dir, args.output_index)
