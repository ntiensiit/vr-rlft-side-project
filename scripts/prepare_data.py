# ruff: noqa: I001
import argparse
import json
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
from grasping_ai.evaluation.collision import generate_analytical_contacts
from grasping_ai.evaluation.force_closure import compute_grasp_quality
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
    oversample_factor: int = 2,
    oversample_extra: int = 256,
    neighborhood_size: int = 30,
    voxel_size: float = 1e-5,
    strict_antipodal_dot: float = 0.5,
    strict_alignment_dot: float = 0.5,
    relaxed_antipodal_dot: float = 0.3,
    allow_relaxed: bool = True,
    search_multiplier: int = 50,
    candidate_multiplier: int = 3,
    min_grasp_translation: float = 0.01,
    min_grasp_rotation: float = 0.2,
    min_quality_score: float = 0.0,
    friction_coefficient: float = 0.5,
    collision_clearance: float = 0.005,
    sim_validate: bool = False,
    mjcf_root: Path | None = None,
    robot_xml: Path | None = None,
    num_simulation_steps: int = 500,
    gripper_close_command: np.ndarray | None = None,
    lift_height_threshold: float = 0.05,
    max_linear_velocity: float = 0.05,
    max_angular_velocity: float = 0.1,
    quality_report_path: Path | None = None,
    sim_object_position: np.ndarray | None = None,
    sim_validate_require_lift: bool = False,
    sim_validate_require_ik: bool = True,
    sim_validate_min_contacts: float = 1.0,
    sim_validate_fallback_analytical: bool = True,
    table_xml: Path | None = None,
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
        oversample_factor: Multiplier applied to ``num_samples`` before FPS.
        oversample_extra: Additional mesh samples added before FPS refinement.
        neighborhood_size: Neighborhood size for normal estimation.
        voxel_size: Voxel size for point cloud downsampling.
        strict_antipodal_dot: Strict antipodal normal opposition threshold.
        strict_alignment_dot: Strict action-line alignment threshold.
        relaxed_antipodal_dot: Relaxed antipodal normal opposition threshold.
        allow_relaxed: Whether to retry with relaxed antipodal constraints.
        search_multiplier: Random search attempts per requested grasp candidate.
        candidate_multiplier: Oversampling factor before quality filtering.
        min_grasp_translation: Minimum translation distance between kept grasps.
        min_grasp_rotation: Minimum rotation distance in radians between kept grasps.
        min_quality_score: Minimum analytical grasp-quality score to keep.
        friction_coefficient: Friction coefficient for force-closure scoring.
        collision_clearance: Clearance used for collision and contact checks.
        sim_validate: When ``True``, keep only MuJoCo lift-success grasps.
        mjcf_root: YCB MJCF root required when ``sim_validate`` is enabled.
        robot_xml: Robot MJCF path required when ``sim_validate`` is enabled.
        num_simulation_steps: Physics steps per grasp when sim validating.
        gripper_close_command: Gripper close command for sim validation.
        lift_height_threshold: Lift threshold for sim validation.
        max_linear_velocity: Stability linear velocity limit for sim validation.
        max_angular_velocity: Stability angular velocity limit for sim validation.
        quality_report_path: Optional JSON path for per-object quality metrics.
        sim_object_position: Object-frame origin in world coordinates used when
            converting mesh-local grasps for MuJoCo validation.
        sim_validate_require_lift: Require lift success during sim validation.
        sim_validate_require_ik: Require converged IK during sim validation.
        sim_validate_min_contacts: Minimum MuJoCo contact count for sim validation.
        sim_validate_fallback_analytical: Keep analytical labels when sim validation
            filters out every candidate grasp.
        table_xml: Optional table MJCF path used during sim validation.

    Raises:
        FileNotFoundError: If ``ycb_root`` or required sim assets are missing.
        RuntimeError: If any required object fails generation.
        ValueError: If sim validation is enabled without required sim paths.
    """
    if not ycb_root.is_dir():
        raise FileNotFoundError(f"YCB root directory '{ycb_root}' does not exist.")
    if sim_validate and (mjcf_root is None or robot_xml is None):
        raise ValueError("sim_validate requires mjcf_root and robot_xml")
    if sim_validate and gripper_close_command is None:
        raise ValueError("sim_validate requires gripper_close_command")
    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be positive")
    if search_multiplier <= 0:
        raise ValueError("search_multiplier must be positive")

    required_set = set(required_objects or [])
    failures: list[str] = []
    quality_records: list[dict[str, object]] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    from grasping_ai.perception.geometry import make_transform
    from grasping_ai.robotics.transforms import convert_grasps_to_world_frame

    object_position = (
        sim_object_position if sim_object_position is not None else np.array([0.5, 0.0, 0.1], dtype=np.float64)
    )
    if object_position.shape != (3,):
        raise ValueError("sim_object_position must have shape (3,)")
    object_to_world = make_transform(np.eye(3, dtype=np.float64), object_position)

    x = np.linspace(-0.03, 0.03, 4)
    y = np.linspace(-0.02, 0.02, 3)
    z = np.linspace(-0.04, 0.04, 5)
    gripper_point_cloud = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
    gripper_point_cloud = gripper_point_cloud.astype(np.float32)

    object_names = list_ycb_objects(ycb_root)
    for name in object_names:
        try:
            mesh_path = resolve_ycb_object_id(ycb_root, name)
            if not mesh_path.is_file():
                candidates = list(mesh_path.rglob("*.obj")) + list(mesh_path.rglob("*.ply"))
                if candidates:
                    mesh_path = candidates[0]
                else:
                    print(f"Skipping {name}: no mesh files found.")
                    if name in required_set:
                        failures.append(f"{name}: no mesh files found")
                    continue

            oversample_count = max(
                num_samples,
                num_samples * oversample_factor,
                num_samples + oversample_extra,
            )
            raw_points = sample_point_cloud_from_mesh(mesh_path, oversample_count, rng)
            if raw_points.shape[0] >= num_samples:
                fps_indices = farthest_point_sampling(raw_points, num_samples, rng)
                points = raw_points[fps_indices]
            else:
                points = sample_point_cloud(raw_points, num_samples, rng)
            points = voxel_downsample(points, voxel_size=voxel_size)
            if points.shape[0] != num_samples:
                points = sample_point_cloud(points, num_samples, rng)

            normals = estimate_point_cloud_normals(points, neighborhood_size=neighborhood_size)

            candidate_count = num_grasps * candidate_multiplier
            grasp_kwargs = {
                "strict_antipodal_dot": strict_antipodal_dot,
                "strict_alignment_dot": strict_alignment_dot,
                "search_multiplier": search_multiplier,
            }
            grasps = generate_analytical_grasps(
                points,
                normals,
                candidate_count,
                gripper_width,
                rng,
                **grasp_kwargs,
            )
            grasp_source = "strict"
            if grasps.shape[0] == 0 and allow_relaxed:
                grasps = generate_analytical_grasps(
                    points,
                    normals,
                    candidate_count,
                    gripper_width,
                    rng,
                    allow_relaxed=True,
                    relaxed_antipodal_dot=relaxed_antipodal_dot,
                    **grasp_kwargs,
                )
                grasp_source = "relaxed"

            if grasps.shape[0] == 0:
                print(f"Skipping {name}: no valid grasps found.")
                if name in required_set:
                    failures.append(f"{name}: no valid grasps")
                quality_records.append(
                    {
                        "object_id": name,
                        "source": "none",
                        "candidates": 0,
                        "contact_scored": 0,
                        "scored": 0,
                        "kept": 0,
                        "sim_pass": 0,
                        "mean_score": 0.0,
                    }
                )
                continue

            scored_grasps: list[tuple[np.ndarray, float]] = []
            for pose in grasps:
                contacts = generate_analytical_contacts(
                    points,
                    gripper_point_cloud,
                    pose,
                    contact_clearance=collision_clearance,
                )
                if len(contacts) < 2:
                    continue
                score = compute_grasp_quality(contacts, friction_coefficient)
                if score >= min_quality_score:
                    scored_grasps.append((pose, score))
            scored_grasps.sort(key=lambda item: item[1], reverse=True)
            contact_scored_count = len(scored_grasps)
            analytical_scored = list(scored_grasps)
            sim_pass_count = 0
            label_source = grasp_source

            if sim_validate and mjcf_root is not None and robot_xml is not None:
                from grasping_ai.pipelines.simulate_grasp import simulate_grasp

                sim_filtered: list[tuple[np.ndarray, float]] = []
                close_command = cast(np.ndarray, gripper_close_command)
                table_xml_path = table_xml if table_xml is not None and table_xml.is_file() else None
                for pose, score in analytical_scored:
                    world_pose = convert_grasps_to_world_frame(
                        pose.reshape(1, 4, 4),
                        object_to_world,
                    )[0]
                    outcome = simulate_grasp(
                        world_pose,
                        name,
                        mjcf_root,
                        robot_xml,
                        table_xml_path=table_xml_path,
                        num_simulation_steps=num_simulation_steps,
                        gripper_close_command=close_command,
                        lift_height_threshold=lift_height_threshold,
                        max_linear_velocity=max_linear_velocity,
                        max_angular_velocity=max_angular_velocity,
                        grasp_width=gripper_width,
                        quiet=True,
                    )
                    fk_error = float(outcome.get("fk_position_error", float("inf")))
                    contact_count = float(outcome.get("contact_count", 0.0))
                    ik_ok = np.isfinite(fk_error) if sim_validate_require_ik else True
                    contact_ok = contact_count >= sim_validate_min_contacts
                    lift_ok = True
                    if sim_validate_require_lift:
                        initial_height = float(outcome.get("initial_height", 0.0))
                        final_height = float(outcome.get("final_height", 0.0))
                        lift_ok = (final_height - initial_height) >= lift_height_threshold
                    if ik_ok and contact_ok and lift_ok:
                        sim_filtered.append((pose, score))
                sim_pass_count = len(sim_filtered)
                if sim_filtered:
                    scored_grasps = sim_filtered
                    label_source = f"{grasp_source}+sim"
                elif sim_validate_fallback_analytical and analytical_scored:
                    print(f"{name}: no sim-validated grasps; keeping {len(analytical_scored)} analytical labels.")
                    scored_grasps = analytical_scored
                    label_source = f"{grasp_source}+sim_fallback"
                else:
                    scored_grasps = []
                    label_source = f"{grasp_source}+sim_none"

            kept_poses: list[np.ndarray] = []
            kept_scores: list[float] = []
            for pose, score in scored_grasps:
                if len(kept_poses) >= num_grasps:
                    break
                translation = pose[:3, 3]
                rotation = pose[:3, :3]
                too_close = False
                for kept_pose in kept_poses:
                    kept_translation = kept_pose[:3, 3]
                    if np.linalg.norm(translation - kept_translation) >= min_grasp_translation:
                        continue
                    kept_rotation = kept_pose[:3, :3]
                    trace_value = float(np.trace(kept_rotation.T @ rotation))
                    angle = float(np.arccos(np.clip((trace_value - 1.0) / 2.0, -1.0, 1.0)))
                    if angle < min_grasp_rotation:
                        too_close = True
                        break
                if not too_close:
                    kept_poses.append(pose)
                    kept_scores.append(score)

            if not kept_poses:
                print(f"Skipping {name}: no grasps passed quality filters.")
                if name in required_set:
                    failures.append(f"{name}: no grasps passed quality filters")
                quality_records.append(
                    {
                        "object_id": name,
                        "source": grasp_source,
                        "candidates": int(grasps.shape[0]),
                        "contact_scored": contact_scored_count,
                        "scored": len(scored_grasps),
                        "kept": 0,
                        "sim_pass": len(scored_grasps) if sim_validate else 0,
                        "mean_score": 0.0,
                    }
                )
                continue

            mean_score = float(np.mean(kept_scores))
            print(
                f"{name}: source={label_source}, kept={len(kept_poses)}/{num_grasps}, "
                f"mean_score={mean_score:.4f}, sim_pass={sim_pass_count}"
            )
            quality_records.append(
                {
                    "object_id": name,
                    "source": label_source,
                    "candidates": int(grasps.shape[0]),
                    "contact_scored": contact_scored_count,
                    "scored": len(scored_grasps),
                    "kept": len(kept_poses),
                    "sim_pass": sim_pass_count,
                    "mean_score": mean_score,
                }
            )

            sample = {
                "point_cloud": points.astype(np.float32),
                "grasp_poses": np.stack(kept_poses, axis=0).astype(np.float32),
                "scores": np.asarray(kept_scores, dtype=np.float32),
                "object_id": name,
            }

            output_file = output_dir / f"{name}.npy"
            np.save(output_file, cast(Any, sample))

        except Exception as e:
            print(f"Failed to generate synthetic data for '{name}': {e}")
            if name in required_set:
                failures.append(f"{name}: {e}")

    if quality_report_path is not None:
        quality_report_path.parent.mkdir(parents=True, exist_ok=True)
        quality_report_path.write_text(json.dumps(quality_records, indent=2), encoding="utf-8")

    if failures:
        joined = "; ".join(failures)
        raise RuntimeError(f"Required YCB objects failed to generate synthetic data: {joined}")


if __name__ == "__main__":
    from grasping_ai.config.yaml_loader import (
        config_float_list,
        config_get,
        config_path,
        config_str_list,
        load_project_yaml_config,
        parse_config_dir_from_argv,
    )

    config_dir = parse_config_dir_from_argv()
    cfg = load_project_yaml_config(config_dir, "base", "data", "object", "evaluation", "gripper", "env")
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
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=None,
        help="Optional JSON path for per-object synthetic label quality metrics.",
    )
    parser.add_argument(
        "--sim-validate",
        action="store_true",
        default=bool(config_get(cfg, "synthetic", "sim_validate", default=False)),
        help="Keep only MuJoCo lift-success grasps (slow; requires MJCF assets).",
    )

    args = parser.parse_args()
    if args.output_index is None:
        parser.error(
            "--output-index is required (set in configs/data/default.yaml paths.output_index or pass explicitly)"
        )

    target_dir = args.output_dir if args.output_dir is not None else args.dataset_root

    if args.mode == "synthetic":
        if args.ycb_root is None:
            parser.error("--ycb-root is required when --mode is synthetic")
        if target_dir is None:
            parser.error("--output-dir or --dataset-root is required when --mode is synthetic")

        gripper_close = config_get(cfg, "robot", "gripper", "close_command")
        close_command = np.asarray(gripper_close, dtype=np.float64) if isinstance(gripper_close, list) else None
        synthetic_cfg = cast(dict[str, object], config_get(cfg, "synthetic", default={}) or {})
        metrics_cfg = cast(dict[str, object], config_get(cfg, "metrics", default={}) or {})
        limits_cfg = cast(dict[str, object], config_get(cfg, "limits", default={}) or {})
        sim_position = config_float_list(cfg, "synthetic", "sim_object_position")
        sim_object_position = (
            np.asarray(sim_position, dtype=np.float64) if sim_position else np.array([0.5, 0.0, 0.1], dtype=np.float64)
        )
        sim_table = config_get(cfg, "synthetic", "sim_table_xml")
        table_xml = Path(str(sim_table)) if isinstance(sim_table, str) else None
        if table_xml is not None and not table_xml.is_absolute():
            table_xml = Path(__file__).resolve().parents[1] / table_xml

        generate_synthetic_dataset(
            ycb_root=args.ycb_root,
            output_dir=target_dir,
            num_samples=args.num_samples,
            num_grasps=args.num_grasps,
            gripper_width=args.gripper_width,
            seed=args.seed,
            required_objects=args.required_objects,
            oversample_factor=int(synthetic_cfg.get("oversample_factor", 2)),
            oversample_extra=int(synthetic_cfg.get("oversample_extra", 256)),
            neighborhood_size=int(synthetic_cfg.get("neighborhood_size", 30)),
            voxel_size=float(synthetic_cfg.get("voxel_size", 1e-5)),
            strict_antipodal_dot=float(synthetic_cfg.get("strict_antipodal_dot", 0.5)),
            strict_alignment_dot=float(synthetic_cfg.get("strict_alignment_dot", 0.5)),
            relaxed_antipodal_dot=float(synthetic_cfg.get("relaxed_antipodal_dot", 0.3)),
            allow_relaxed=bool(synthetic_cfg.get("allow_relaxed", True)),
            search_multiplier=int(synthetic_cfg.get("search_multiplier", 50)),
            candidate_multiplier=int(synthetic_cfg.get("candidate_multiplier", 3)),
            min_grasp_translation=float(synthetic_cfg.get("min_grasp_translation", 0.01)),
            min_grasp_rotation=float(synthetic_cfg.get("min_grasp_rotation", 0.2)),
            min_quality_score=float(synthetic_cfg.get("min_quality_score", 0.0)),
            friction_coefficient=float(
                synthetic_cfg.get(
                    "friction_coefficient",
                    metrics_cfg.get("friction_coefficient", 0.5),
                )
            ),
            collision_clearance=float(
                synthetic_cfg.get(
                    "collision_clearance",
                    metrics_cfg.get("collision_clearance", 0.005),
                )
            ),
            sim_validate=args.sim_validate,
            mjcf_root=config_path(cfg, "paths", "ycb_mjcf"),
            robot_xml=config_path(cfg, "robot", "description"),
            num_simulation_steps=int(config_get(cfg, "num_steps", default=500)),
            gripper_close_command=close_command,
            lift_height_threshold=float(metrics_cfg.get("lift_height_threshold", 0.05)),
            max_linear_velocity=float(limits_cfg.get("max_linear_velocity", 0.05)),
            max_angular_velocity=float(limits_cfg.get("max_angular_velocity", 0.1)),
            quality_report_path=args.quality_report,
            sim_object_position=sim_object_position,
            sim_validate_require_lift=bool(synthetic_cfg.get("sim_validate_require_lift", False)),
            sim_validate_require_ik=bool(synthetic_cfg.get("sim_validate_require_ik", True)),
            sim_validate_min_contacts=float(synthetic_cfg.get("sim_validate_min_contacts", 1.0)),
            sim_validate_fallback_analytical=bool(synthetic_cfg.get("sim_validate_fallback_analytical", True)),
            table_xml=table_xml,
        )
        prepare_data_index(target_dir, args.output_index)
    else:
        if target_dir is None:
            parser.error("--dataset-root or --output-dir is required when --mode is index")
        prepare_data_index(target_dir, args.output_index)
