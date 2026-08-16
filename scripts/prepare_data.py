"""Prepare processed grasp datasets from raw YCB assets."""

from __future__ import annotations

from grasping_ai.config import SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig

from grasping_ai.data.pointcloud_dataset import (
    discover_dataset_files,
    generate_analytical_grasps,
    resolve_ycb_object_id,
    save_grasp_sample,
)

from grasping_ai.data.transforms import save_grasp_dataset_index

from grasping_ai.evaluation.collision import generate_analytical_contacts

from grasping_ai.evaluation.force_closure import compute_grasp_quality

from grasping_ai.perception.geometry import make_transform

from grasping_ai.perception.pointcloud import (
    estimate_point_cloud_normals,
    farthest_point_sampling,
    sample_point_cloud,
    voxel_downsample,
)

from grasping_ai.pipelines.simulate_grasp import simulate_grasp

from grasping_ai.robotics.transforms import convert_grasps_to_world_frame

from grasping_ai.sensors.pointcloud_sensor import sample_point_cloud_from_mesh

from grasping_ai.simulation.ycb import list_ycb_objects

import json
from pathlib import Path
from typing import cast

import hydra
import numpy as np
import open3d as _open3d  # noqa: F401
from loguru import logger
from omegaconf import DictConfig

def prepare_data_index(dataset_root: Path, output_index_path: Path) -> None:
    """Discover dataset files and write a dataset index file.

    Args:
        dataset_root: Root directory containing raw dataset records.
        output_index_path: Destination path for the generated index file.
    """
    records = discover_dataset_files(dataset_root)
    entries = [{"path": str(record)} for record in records]
    save_grasp_dataset_index(output_index_path.parent, entries, output_index_path.name)

def _default_gripper_point_cloud() -> np.ndarray:
    x = np.linspace(-0.03, 0.03, 4)
    y = np.linspace(-0.02, 0.02, 3)
    z = np.linspace(-0.04, 0.04, 5)
    gripper_point_cloud = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
    return gripper_point_cloud.astype(np.float32)

def _resolve_mesh_path(ycb_root: Path, name: str) -> Path | None:
    mesh_path = resolve_ycb_object_id(ycb_root, name)
    if mesh_path.is_file():
        return mesh_path
    candidates = list(mesh_path.rglob("*.obj")) + list(mesh_path.rglob("*.ply"))
    if candidates:
        return candidates[0]
    return None

def _sample_object_points(
    mesh_path: Path,
    num_samples: int,
    oversample_factor: int,
    oversample_extra: int,
    voxel_size: float,
    rng: np.random.Generator,
) -> np.ndarray:
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

def _generate_candidate_grasps(
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

def _score_analytical_grasps(
    grasps: np.ndarray,
    points: np.ndarray,
    gripper_point_cloud: np.ndarray,
    friction_coefficient: float,
    collision_clearance: float,
    min_quality_score: float,
) -> list[tuple[np.ndarray, float]]:
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
    return scored_grasps

def _sim_validation_passes(
    outcome: dict[str, object],
    *,
    sim_validate_require_ik: bool,
    sim_validate_require_lift: bool,
    sim_validate_min_contacts: float,
    lift_height_threshold: float,
) -> bool:
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

def _apply_sim_validation(
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
        if _sim_validation_passes(
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

def _select_diverse_grasps(
    scored_grasps: list[tuple[np.ndarray, float]],
    num_grasps: int,
    min_grasp_translation: float,
    min_grasp_rotation: float,
) -> tuple[list[np.ndarray], list[float]]:
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

def _quality_record(
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

def _process_object_synthetic(
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
    mesh_path = _resolve_mesh_path(ycb_root, name)
    if mesh_path is None:
        logger.info("Skipping {}: no mesh files found.", name)
        return None

    points = _sample_object_points(
        mesh_path,
        num_samples,
        oversample_factor,
        oversample_extra,
        voxel_size,
        rng,
    )
    normals = estimate_point_cloud_normals(points, neighborhood_size=neighborhood_size)
    candidate_count = num_grasps * candidate_multiplier
    grasps, grasp_source = _generate_candidate_grasps(
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
        return _quality_record(name, "none")

    analytical_scored = _score_analytical_grasps(
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
        scored_grasps, label_source, sim_pass_count = _apply_sim_validation(
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

    kept_poses, kept_scores = _select_diverse_grasps(
        scored_grasps,
        num_grasps,
        min_grasp_translation,
        min_grasp_rotation,
    )
    if not kept_poses:
        logger.info("Skipping {}: no grasps passed quality filters.", name)
        return _quality_record(
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
    return _quality_record(
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
    """Generate synthetic grasp dataset from YCB meshes.

    Args:
        ycb_root: Path to YCB raw data directory.
        output_dir: Output directory where ``.npz`` files will be saved.
        num_samples: Number of points to sample per mesh.
        num_grasps: Number of analytical grasps to generate.
        gripper_width: Maximum gripper width.
        seed: Random seed for reproducibility.
        required_objects: Optional list of object identifiers whose generation
            must succeed. If any listed object fails to produce a record (no
            strict or relaxed grasps, missing mesh, exception), the function
            raises. Objects not in this list are still skipped with a logged
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
    gripper_point_cloud = _default_gripper_point_cloud()

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
            quality_record = _process_object_synthetic(name, **process_kwargs)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.error("Failed to generate synthetic data for '{}': {}", name, exc)
            if name in required_set:
                failures.append(f"{name}: {exc}")
            continue

        if quality_record is None:
            if name in required_set:
                failures.append(f"{name}: no mesh files found")
            continue

        quality_records.append(quality_record)
        kept = int(quality_record["kept"])
        if kept == 0 and name in required_set:
            source = str(quality_record["source"])
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

@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/prepare_data")
def main(cfg: DictConfig) -> None:
    yaml_config = FlattenedYAMLConfig(cfg)
    mode = str(yaml_config.value("mode", "prepare", "mode", value_type=object, script_or=True))
    dataset_root = yaml_config.value("paths", "dataset_root", value_type=Path)
    output_dir = yaml_config.value("output_dir", "prepare", "output_dir", value_type=Path, script_or=True)
    output_index = yaml_config.value("paths", "output_index", value_type=Path, required=True)
    target_dir = output_dir if output_dir is not None else dataset_root

    if mode == "synthetic":
        if dataset_root is None and output_dir is None:
            msg = "prepare.output_dir or paths.dataset_root is required when prepare.mode is synthetic"
            raise ValueError(msg)
        ycb_root = yaml_config.value("paths", "ycb_root", value_type=Path, required=True)
        if target_dir is None:
            msg = "prepare.output_dir or paths.dataset_root is required when prepare.mode is synthetic"
            raise ValueError(msg)

        gripper_close = yaml_config.get_path("robot", "gripper", "close_command")
        close_command = np.asarray(gripper_close, dtype=np.float64) if isinstance(gripper_close, list) else None
        synthetic_cfg = yaml_config.get("synthetic")
        limits_cfg = yaml_config.get("limits")
        sim_position = yaml_config.value("synthetic", "sim_object_position", value_type=list[float], required=True)
        sim_object_position = np.asarray(sim_position, dtype=np.float64)
        sim_table = yaml_config.get_path("synthetic", "sim_table_xml")
        table_xml = Path(str(sim_table)) if isinstance(sim_table, str) else None
        if table_xml is not None and not table_xml.is_absolute():
            table_xml = Path(__file__).resolve().parents[1] / table_xml

        generate_synthetic_dataset(
            ycb_root=ycb_root,
            output_dir=target_dir,
            num_samples=yaml_config.value("synthetic", "num_samples", value_type=int),
            num_grasps=yaml_config.value("synthetic", "num_grasps", value_type=int),
            gripper_width=yaml_config.value("synthetic", "gripper_width", value_type=float),
            seed=yaml_config.value("synthetic", "seed", value_type=int),
            required_objects=yaml_config.value("objects", "ids", value_type=list[str]),
            oversample_factor=yaml_config.value("synthetic", "oversample_factor", value_type=int),
            oversample_extra=yaml_config.value("synthetic", "oversample_extra", value_type=int),
            neighborhood_size=yaml_config.value("synthetic", "neighborhood_size", value_type=int),
            voxel_size=yaml_config.value("synthetic", "voxel_size", value_type=float),
            strict_antipodal_dot=yaml_config.value("synthetic", "strict_antipodal_dot", value_type=float),
            strict_alignment_dot=yaml_config.value("synthetic", "strict_alignment_dot", value_type=float),
            relaxed_antipodal_dot=yaml_config.value("synthetic", "relaxed_antipodal_dot", value_type=float),
            allow_relaxed=yaml_config.value("synthetic", "allow_relaxed", value_type=bool),
            search_multiplier=yaml_config.value("synthetic", "search_multiplier", value_type=int),
            candidate_multiplier=yaml_config.value("synthetic", "candidate_multiplier", value_type=int),
            min_grasp_translation=yaml_config.value("synthetic", "min_grasp_translation", value_type=float),
            min_grasp_rotation=yaml_config.value("synthetic", "min_grasp_rotation", value_type=float),
            min_quality_score=yaml_config.value("synthetic", "min_quality_score", value_type=float),
            friction_coefficient=yaml_config.value("synthetic", "friction_coefficient", value_type=float),
            collision_clearance=yaml_config.value("synthetic", "collision_clearance", value_type=float),
            sim_validate=yaml_config.value("synthetic", "sim_validate", value_type=bool),
            mjcf_root=yaml_config.value("paths", "ycb_mjcf", value_type=Path),
            robot_xml=yaml_config.value("robot", "description", value_type=Path),
            num_simulation_steps=yaml_config.value("synthetic", "num_simulation_steps", value_type=int),
            gripper_close_command=close_command,
            lift_height_threshold=yaml_config.value("metrics", "lift_height_threshold", value_type=float),
            max_linear_velocity=yaml_config.value("limits", "max_linear_velocity", value_type=float),
            max_angular_velocity=yaml_config.value("limits", "max_angular_velocity", value_type=float),
            quality_report_path=yaml_config.value("quality_report", "prepare", "quality_report", value_type=Path, script_or=True),
            sim_object_position=sim_object_position,
            sim_validate_require_lift=yaml_config.value("synthetic", "sim_validate_require_lift", value_type=bool),
            sim_validate_require_ik=yaml_config.value("synthetic", "sim_validate_require_ik", value_type=bool),
            sim_validate_min_contacts=yaml_config.value("synthetic", "sim_validate_min_contacts", value_type=float),
            sim_validate_fallback_analytical=yaml_config.value(
                "synthetic", "sim_validate_fallback_analytical", value_type=bool
            ),
            table_xml=table_xml,
        )
        prepare_data_index(target_dir, output_index)
    else:
        if target_dir is None:
            msg = "paths.dataset_root or prepare.output_dir is required when prepare.mode is index"
            raise ValueError(msg)
        prepare_data_index(target_dir, output_index)

if __name__ == "__main__":
    main()
