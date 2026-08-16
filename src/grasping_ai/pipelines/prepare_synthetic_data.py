"""Synthetic grasp dataset generation from YCB meshes."""

from __future__ import annotations

from grasping_ai.data.pointcloud_dataset import (
    discover_dataset_files,
    generate_analytical_grasps,
    resolve_ycb_object_id,
    save_grasp_sample,
)
from grasping_ai.data.transforms import save_grasp_dataset_index
from grasping_ai.evaluation.scoring import score_grasp_poses_by_contacts
from grasping_ai.perception.geometry import make_transform
from grasping_ai.perception.pointcloud import (
    estimate_point_cloud_normals,
    farthest_point_sampling,
    sample_point_cloud,
    voxel_downsample,
)
from grasping_ai.pipelines.simulate_grasp import simulate_grasp
from grasping_ai.robotics.gripper import default_gripper_point_cloud
from grasping_ai.robotics.transforms import convert_grasps_to_world_frame
from grasping_ai.sensors.pointcloud_sensor import sample_point_cloud_from_mesh
from grasping_ai.simulation.ycb import list_ycb_objects

import json
from pathlib import Path
from typing import cast

import numpy as np
from loguru import logger


def prepare_data_index(dataset_root: Path, output_index_path: Path) -> None:
    """Discover dataset files and write a dataset index file."""
    records = discover_dataset_files(dataset_root)
    entries = [{"path": str(record)} for record in records]
    save_grasp_dataset_index(output_index_path.parent, entries, output_index_path.name)


def resolve_mesh_path(ycb_root: Path, name: str) -> Path | None:
    """Resolve a YCB object mesh path, falling back to nested OBJ/PLY files."""
    mesh_path = resolve_ycb_object_id(ycb_root, name)
    if mesh_path.is_file():
        return mesh_path
    candidates = list(mesh_path.rglob("*.obj")) + list(mesh_path.rglob("*.ply"))
    if candidates:
        return candidates[0]
    return None


def sample_object_points(
    mesh_path: Path,
    num_samples: int,
    oversample_factor: int,
    oversample_extra: int,
    voxel_size: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample, refine, and downsample object points from a mesh file."""
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
    return points


def generate_labeled_analytical_grasps(
    points: np.ndarray,
    normals: np.ndarray,
    candidate_count: int,
    gripper_width: float,
    rng: np.random.Generator,
    *,
    strict_antipodal_dot: float,
    strict_alignment_dot: float,
    relaxed_antipodal_dot: float,
    allow_relaxed: bool,
    search_multiplier: int,
) -> tuple[np.ndarray, str]:
    """Generate analytical grasps and report whether strict or relaxed succeeded."""
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
    if grasps.shape[0] > 0:
        return grasps, "strict"

    if not allow_relaxed:
        return grasps, "none"

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
    if grasps.shape[0] > 0:
        return grasps, "relaxed"
    return grasps, "none"


def sim_validation_passes(
    outcome: dict[str, object],
    *,
    sim_validate_require_ik: bool,
    sim_validate_require_lift: bool,
    sim_validate_min_contacts: float,
    lift_height_threshold: float,
) -> bool:
    """Return whether a MuJoCo grasp outcome satisfies validation criteria."""
    fk_error = float(outcome.get("fk_position_error", float("inf")))
    contact_count = float(outcome.get("contact_count", 0.0))
    ik_ok = np.isfinite(fk_error) if sim_validate_require_ik else True
    contact_ok = contact_count >= sim_validate_min_contacts
    lift_ok = True
    if sim_validate_require_lift:
        initial_height = float(outcome.get("initial_height", 0.0))
        final_height = float(outcome.get("final_height", 0.0))
        lift_ok = (final_height - initial_height) >= lift_height_threshold
    return ik_ok and contact_ok and lift_ok


def apply_sim_validation(
    analytical_scored: list[tuple[np.ndarray, float]],
    *,
    name: str,
    grasp_source: str,
    object_to_world: np.ndarray,
    mjcf_root: Path,
    robot_xml: Path,
    table_xml: Path | None,
    num_simulation_steps: int,
    gripper_close_command: np.ndarray,
    lift_height_threshold: float,
    max_linear_velocity: float,
    max_angular_velocity: float,
    gripper_width: float,
    sim_validate_require_ik: bool,
    sim_validate_require_lift: bool,
    sim_validate_min_contacts: float,
    sim_validate_fallback_analytical: bool,
) -> tuple[list[tuple[np.ndarray, float]], str, int]:
    """Filter analytically scored grasps using MuJoCo simulation outcomes."""
    sim_filtered: list[tuple[np.ndarray, float]] = []
    table_xml_path = table_xml if table_xml is not None and table_xml.is_file() else None
    for pose, score in analytical_scored:
        world_pose = convert_grasps_to_world_frame(pose.reshape(1, 4, 4), object_to_world)[0]
        outcome = simulate_grasp(
            world_pose,
            name,
            mjcf_root,
            robot_xml,
            table_xml_path=table_xml_path,
            num_simulation_steps=num_simulation_steps,
            gripper_close_command=gripper_close_command,
            lift_height_threshold=lift_height_threshold,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
            grasp_width=gripper_width,
            quiet=True,
        )
        if sim_validation_passes(
            outcome,
            sim_validate_require_ik=sim_validate_require_ik,
            sim_validate_require_lift=sim_validate_require_lift,
            sim_validate_min_contacts=sim_validate_min_contacts,
            lift_height_threshold=lift_height_threshold,
        ):
            sim_filtered.append((pose, score))

    sim_pass_count = len(sim_filtered)
    if sim_filtered:
        return sim_filtered, f"{grasp_source}+sim", sim_pass_count
    if sim_validate_fallback_analytical and analytical_scored:
        logger.info(
            "{}: no sim-validated grasps; keeping {} analytical labels.",
            name,
            len(analytical_scored),
        )
        return analytical_scored, f"{grasp_source}+sim_fallback", sim_pass_count
    return [], f"{grasp_source}+sim_none", sim_pass_count


def select_diverse_grasps(
    scored_grasps: list[tuple[np.ndarray, float]],
    num_grasps: int,
    min_grasp_translation: float,
    min_grasp_rotation: float,
) -> tuple[list[np.ndarray], list[float]]:
    """Keep high-scoring grasps that are sufficiently separated in pose space."""
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
    return kept_poses, kept_scores


def quality_record(
    object_id: str,
    source: str,
    *,
    candidates: int = 0,
    contact_scored: int = 0,
    scored: int = 0,
    kept: int = 0,
    sim_pass: int = 0,
    mean_score: float = 0.0,
) -> dict[str, object]:
    """Build a per-object synthetic dataset quality record."""
    return {
        "object_id": object_id,
        "source": source,
        "candidates": candidates,
        "contact_scored": contact_scored,
        "scored": scored,
        "kept": kept,
        "sim_pass": sim_pass,
        "mean_score": mean_score,
    }


def process_object_synthetic(
    name: str,
    *,
    ycb_root: Path,
    output_dir: Path,
    num_samples: int,
    num_grasps: int,
    gripper_width: float,
    rng: np.random.Generator,
    gripper_point_cloud: np.ndarray,
    object_to_world: np.ndarray,
    oversample_factor: int,
    oversample_extra: int,
    neighborhood_size: int,
    voxel_size: float,
    strict_antipodal_dot: float,
    strict_alignment_dot: float,
    relaxed_antipodal_dot: float,
    allow_relaxed: bool,
    search_multiplier: int,
    candidate_multiplier: int,
    min_grasp_translation: float,
    min_grasp_rotation: float,
    min_quality_score: float,
    friction_coefficient: float,
    collision_clearance: float,
    sim_validate: bool,
    mjcf_root: Path | None,
    robot_xml: Path | None,
    num_simulation_steps: int,
    gripper_close_command: np.ndarray | None,
    lift_height_threshold: float,
    max_linear_velocity: float,
    max_angular_velocity: float,
    sim_validate_require_lift: bool,
    sim_validate_require_ik: bool,
    sim_validate_min_contacts: float,
    sim_validate_fallback_analytical: bool,
    table_xml: Path | None,
) -> dict[str, object] | None:
    """Generate and persist one synthetic grasp dataset record for an object."""
    mesh_path = resolve_mesh_path(ycb_root, name)
    if mesh_path is None:
        logger.info("Skipping {}: no mesh files found.", name)
        return None

    points = sample_object_points(
        mesh_path,
        num_samples,
        oversample_factor,
        oversample_extra,
        voxel_size,
        rng,
    )
    normals = estimate_point_cloud_normals(points, neighborhood_size=neighborhood_size)
    candidate_count = num_grasps * candidate_multiplier
    grasps, grasp_source = generate_labeled_analytical_grasps(
        points,
        normals,
        candidate_count,
        gripper_width,
        rng,
        strict_antipodal_dot=strict_antipodal_dot,
        strict_alignment_dot=strict_alignment_dot,
        relaxed_antipodal_dot=relaxed_antipodal_dot,
        allow_relaxed=allow_relaxed,
        search_multiplier=search_multiplier,
    )
    if grasps.shape[0] == 0:
        logger.info("Skipping {}: no valid grasps found.", name)
        return quality_record(name, "none")

    analytical_scored = score_grasp_poses_by_contacts(
        grasps,
        points,
        gripper_point_cloud,
        friction_coefficient,
        collision_clearance,
        min_quality_score,
    )
    contact_scored_count = len(analytical_scored)
    scored_grasps = list(analytical_scored)
    sim_pass_count = 0
    label_source = grasp_source

    if sim_validate and mjcf_root is not None and robot_xml is not None and gripper_close_command is not None:
        scored_grasps, label_source, sim_pass_count = apply_sim_validation(
            analytical_scored,
            name=name,
            grasp_source=grasp_source,
            object_to_world=object_to_world,
            mjcf_root=mjcf_root,
            robot_xml=robot_xml,
            table_xml=table_xml,
            num_simulation_steps=num_simulation_steps,
            gripper_close_command=cast("np.ndarray", gripper_close_command),
            lift_height_threshold=lift_height_threshold,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
            gripper_width=gripper_width,
            sim_validate_require_ik=sim_validate_require_ik,
            sim_validate_require_lift=sim_validate_require_lift,
            sim_validate_min_contacts=sim_validate_min_contacts,
            sim_validate_fallback_analytical=sim_validate_fallback_analytical,
        )

    kept_poses, kept_scores = select_diverse_grasps(
        scored_grasps,
        num_grasps,
        min_grasp_translation,
        min_grasp_rotation,
    )
    if not kept_poses:
        logger.info("Skipping {}: no grasps passed quality filters.", name)
        return quality_record(
            name,
            grasp_source,
            candidates=int(grasps.shape[0]),
            contact_scored=contact_scored_count,
            scored=len(scored_grasps),
            sim_pass=len(scored_grasps) if sim_validate else 0,
        )

    mean_score = float(np.mean(kept_scores))
    logger.info(
        "{}: source={}, kept={}/{}, mean_score={:.4f}, sim_pass={}",
        name,
        label_source,
        len(kept_poses),
        num_grasps,
        mean_score,
        sim_pass_count,
    )
    sample = {
        "point_cloud": points.astype(np.float32),
        "grasp_poses": np.stack(kept_poses, axis=0).astype(np.float32),
        "scores": np.asarray(kept_scores, dtype=np.float32),
        "object_id": name,
    }
    output_file = output_dir / f"{name}.npz"
    save_grasp_sample(output_file, sample)
    return quality_record(
        name,
        label_source,
        candidates=int(grasps.shape[0]),
        contact_scored=contact_scored_count,
        scored=len(scored_grasps),
        kept=len(kept_poses),
        sim_pass=sim_pass_count,
        mean_score=mean_score,
    )


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
    """Generate synthetic grasp dataset from YCB meshes."""
    if not ycb_root.is_dir():
        msg = f"YCB root directory '{ycb_root}' does not exist."
        raise FileNotFoundError(msg)
    if sim_validate and (mjcf_root is None or robot_xml is None):
        msg = "sim_validate requires mjcf_root and robot_xml"
        raise ValueError(msg)
    if sim_validate and gripper_close_command is None:
        msg = "sim_validate requires gripper_close_command"
        raise ValueError(msg)
    if candidate_multiplier <= 0:
        msg = "candidate_multiplier must be positive"
        raise ValueError(msg)
    if search_multiplier <= 0:
        msg = "search_multiplier must be positive"
        raise ValueError(msg)

    required_set = set(required_objects or [])
    failures: list[str] = []
    quality_records: list[dict[str, object]] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    gripper_point_cloud = default_gripper_point_cloud()

    object_position = (
        sim_object_position if sim_object_position is not None else np.array([0.5, 0.0, 0.3], dtype=np.float64)
    )
    if object_position.shape != (3,):
        msg = "sim_object_position must have shape (3,)"
        raise ValueError(msg)
    object_to_world = make_transform(np.eye(3, dtype=np.float64), object_position)

    process_kwargs = {
        "ycb_root": ycb_root,
        "output_dir": output_dir,
        "num_samples": num_samples,
        "num_grasps": num_grasps,
        "gripper_width": gripper_width,
        "rng": rng,
        "gripper_point_cloud": gripper_point_cloud,
        "object_to_world": object_to_world,
        "oversample_factor": oversample_factor,
        "oversample_extra": oversample_extra,
        "neighborhood_size": neighborhood_size,
        "voxel_size": voxel_size,
        "strict_antipodal_dot": strict_antipodal_dot,
        "strict_alignment_dot": strict_alignment_dot,
        "relaxed_antipodal_dot": relaxed_antipodal_dot,
        "allow_relaxed": allow_relaxed,
        "search_multiplier": search_multiplier,
        "candidate_multiplier": candidate_multiplier,
        "min_grasp_translation": min_grasp_translation,
        "min_grasp_rotation": min_grasp_rotation,
        "min_quality_score": min_quality_score,
        "friction_coefficient": friction_coefficient,
        "collision_clearance": collision_clearance,
        "sim_validate": sim_validate,
        "mjcf_root": mjcf_root,
        "robot_xml": robot_xml,
        "num_simulation_steps": num_simulation_steps,
        "gripper_close_command": gripper_close_command,
        "lift_height_threshold": lift_height_threshold,
        "max_linear_velocity": max_linear_velocity,
        "max_angular_velocity": max_angular_velocity,
        "sim_validate_require_lift": sim_validate_require_lift,
        "sim_validate_require_ik": sim_validate_require_ik,
        "sim_validate_min_contacts": sim_validate_min_contacts,
        "sim_validate_fallback_analytical": sim_validate_fallback_analytical,
        "table_xml": table_xml,
    }

    for name in list_ycb_objects(ycb_root):
        try:
            quality_record_value = process_object_synthetic(name, **process_kwargs)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.error("Failed to generate synthetic data for '{}': {}", name, exc)
            if name in required_set:
                failures.append(f"{name}: {exc}")
            continue

        if quality_record_value is None:
            if name in required_set:
                failures.append(f"{name}: no mesh files found")
            continue

        quality_records.append(quality_record_value)
        kept = int(quality_record_value["kept"])
        if kept == 0 and name in required_set:
            source = str(quality_record_value["source"])
            if source == "none":
                failures.append(f"{name}: no valid grasps")
            else:
                failures.append(f"{name}: no grasps passed quality filters")

    if quality_report_path is not None:
        quality_report_path.parent.mkdir(parents=True, exist_ok=True)
        quality_report_path.write_text(json.dumps(quality_records, indent=2), encoding="utf-8")

    if failures:
        joined = "; ".join(failures)
        msg = f"Required YCB objects failed to generate synthetic data: {joined}"
        raise RuntimeError(msg)
