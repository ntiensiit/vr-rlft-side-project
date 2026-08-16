"""Synthetic grasp dataset generation from YCB meshes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from loguru import logger

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
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

SYNTHETIC_DEFAULTS = {
    key: FLATTENED_YAML_CONFIG.get(f"synthetic.{key}")
    for key in (
        "num_samples",
        "num_grasps",
        "gripper_width",
        "seed",
        "oversample_factor",
        "oversample_extra",
        "neighborhood_size",
        "voxel_size",
        "strict_antipodal_dot",
        "strict_alignment_dot",
        "relaxed_antipodal_dot",
        "allow_relaxed",
        "search_multiplier",
        "candidate_multiplier",
        "min_grasp_translation",
        "min_grasp_rotation",
        "min_quality_score",
        "friction_coefficient",
        "collision_clearance",
        "sim_validate",
        "num_simulation_steps",
        "lift_height_threshold",
        "max_linear_velocity",
        "max_angular_velocity",
        "sim_validate_require_lift",
        "sim_validate_require_ik",
        "sim_validate_min_contacts",
        "sim_validate_fallback_analytical",
    )
}
SIM_OBJECT_POSITION = tuple(FLATTENED_YAML_CONFIG.get_path("synthetic", "sim_object_position"))

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class _PointSamplingConfig:
    """Point-cloud sampling and downsampling knobs."""

    num_samples: int
    oversample_factor: int
    oversample_extra: int
    voxel_size: float
    neighborhood_size: int


@dataclass(frozen=True)
class _GraspCandidateConfig:
    """Analytical grasp candidate generation and scoring knobs."""

    num_grasps: int
    gripper_width: float
    strict_antipodal_dot: float
    strict_alignment_dot: float
    relaxed_antipodal_dot: float
    allow_relaxed: bool
    search_multiplier: int
    candidate_multiplier: int
    min_grasp_translation: float
    min_grasp_rotation: float
    min_quality_score: float
    friction_coefficient: float
    collision_clearance: float


@dataclass(frozen=True)
class _SimValidationConfig:
    """MuJoCo simulation-validation knobs for synthetic labeling."""

    sim_validate: bool
    mjcf_root: Path | None
    robot_xml: Path | None
    table_xml: Path | None
    num_simulation_steps: int
    gripper_close_command: np.ndarray | None
    lift_height_threshold: float
    max_linear_velocity: float
    max_angular_velocity: float
    sim_validate_require_lift: bool
    sim_validate_require_ik: bool
    sim_validate_min_contacts: float
    sim_validate_fallback_analytical: bool


@dataclass(frozen=True)
class _QualityStats:
    """Per-object labeling counters reported in the quality record."""

    candidates: int = 0
    contact_scored: int = 0
    scored: int = 0
    kept: int = 0
    sim_pass: int = 0
    mean_score: float = 0.0


@dataclass(frozen=True)
class _ObjectProcessingContext:
    """Shared, object-independent inputs for synthetic record generation."""

    ycb_root: Path
    output_dir: Path
    rng: np.random.Generator
    gripper_point_cloud: np.ndarray
    object_to_world: np.ndarray
    sampling: _PointSamplingConfig
    grasp: _GraspCandidateConfig
    sim: _SimValidationConfig


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
    sampling: _PointSamplingConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample, refine, and downsample object points from a mesh file."""
    oversample_count = max(
        sampling.num_samples,
        sampling.num_samples * sampling.oversample_factor,
        sampling.num_samples + sampling.oversample_extra,
    )
    raw_points = sample_point_cloud_from_mesh(mesh_path, oversample_count, rng)
    if raw_points.shape[0] >= sampling.num_samples:
        fps_indices = farthest_point_sampling(raw_points, sampling.num_samples, rng)
        points = raw_points[fps_indices]
    else:
        points = sample_point_cloud(raw_points, sampling.num_samples, rng)
    points = voxel_downsample(points, voxel_size=sampling.voxel_size)
    if points.shape[0] != sampling.num_samples:
        points = sample_point_cloud(points, sampling.num_samples, rng)
    return points


def generate_labeled_analytical_grasps(
    points: np.ndarray,
    normals: np.ndarray,
    candidate_count: int,
    rng: np.random.Generator,
    grasp: _GraspCandidateConfig,
) -> tuple[np.ndarray, str]:
    """Generate analytical grasps and report whether strict or relaxed succeeded."""
    grasp_kwargs = {
        "strict_antipodal_dot": grasp.strict_antipodal_dot,
        "strict_alignment_dot": grasp.strict_alignment_dot,
        "search_multiplier": grasp.search_multiplier,
    }
    grasps = generate_analytical_grasps(
        points,
        normals,
        candidate_count,
        grasp.gripper_width,
        rng,
        **grasp_kwargs,
    )
    if grasps.shape[0] > 0:
        return grasps, "strict"

    if not grasp.allow_relaxed:
        return grasps, "none"

    grasps = generate_analytical_grasps(
        points,
        normals,
        candidate_count,
        grasp.gripper_width,
        rng,
        allow_relaxed=True,
        relaxed_antipodal_dot=grasp.relaxed_antipodal_dot,
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
    ctx: _ObjectProcessingContext,
) -> tuple[list[tuple[np.ndarray, float]], str, int]:
    """Filter analytically scored grasps using MuJoCo simulation outcomes."""
    sim = ctx.sim
    sim_filtered: list[tuple[np.ndarray, float]] = []
    table_xml_path = sim.table_xml if sim.table_xml is not None and sim.table_xml.is_file() else None
    for pose, score in analytical_scored:
        world_pose = convert_grasps_to_world_frame(pose.reshape(1, 4, 4), ctx.object_to_world)[0]
        outcome = simulate_grasp(
            world_pose,
            name,
            cast("Path", sim.mjcf_root),
            cast("Path", sim.robot_xml),
            table_xml_path=table_xml_path,
            num_simulation_steps=sim.num_simulation_steps,
            gripper_close_command=cast("np.ndarray", sim.gripper_close_command),
            lift_height_threshold=sim.lift_height_threshold,
            max_linear_velocity=sim.max_linear_velocity,
            max_angular_velocity=sim.max_angular_velocity,
            grasp_width=ctx.grasp.gripper_width,
            quiet=True,
        )
        if sim_validation_passes(
            outcome,
            sim_validate_require_ik=sim.sim_validate_require_ik,
            sim_validate_require_lift=sim.sim_validate_require_lift,
            sim_validate_min_contacts=sim.sim_validate_min_contacts,
            lift_height_threshold=sim.lift_height_threshold,
        ):
            sim_filtered.append((pose, score))

    sim_pass_count = len(sim_filtered)
    if sim_filtered:
        return sim_filtered, f"{grasp_source}+sim", sim_pass_count
    if sim.sim_validate_fallback_analytical and analytical_scored:
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
    stats: _QualityStats | None = None,
) -> dict[str, object]:
    """Build a per-object synthetic dataset quality record."""
    resolved = stats if stats is not None else _QualityStats()
    return {
        "object_id": object_id,
        "source": source,
        "candidates": resolved.candidates,
        "contact_scored": resolved.contact_scored,
        "scored": resolved.scored,
        "kept": resolved.kept,
        "sim_pass": resolved.sim_pass,
        "mean_score": resolved.mean_score,
    }


def process_object_synthetic(
    name: str,
    ctx: _ObjectProcessingContext,
) -> dict[str, object] | None:
    """Generate and persist one synthetic grasp dataset record for an object."""
    mesh_path = resolve_mesh_path(ctx.ycb_root, name)
    if mesh_path is None:
        logger.info("Skipping {}: no mesh files found.", name)
        return None

    points = sample_object_points(mesh_path, ctx.sampling, ctx.rng)
    normals = estimate_point_cloud_normals(points, neighborhood_size=ctx.sampling.neighborhood_size)
    candidate_count = ctx.grasp.num_grasps * ctx.grasp.candidate_multiplier
    grasps, grasp_source = generate_labeled_analytical_grasps(
        points,
        normals,
        candidate_count,
        ctx.rng,
        ctx.grasp,
    )
    if grasps.shape[0] == 0:
        logger.info("Skipping {}: no valid grasps found.", name)
        return quality_record(name, "none")

    analytical_scored = score_grasp_poses_by_contacts(
        grasps,
        points,
        ctx.gripper_point_cloud,
        ctx.grasp.friction_coefficient,
        ctx.grasp.collision_clearance,
        ctx.grasp.min_quality_score,
    )
    contact_scored_count = len(analytical_scored)
    scored_grasps = list(analytical_scored)
    sim_pass_count = 0
    label_source = grasp_source

    sim = ctx.sim
    sim_assets_ready = sim.mjcf_root is not None and sim.robot_xml is not None and sim.gripper_close_command is not None
    if sim.sim_validate and sim_assets_ready:
        scored_grasps, label_source, sim_pass_count = apply_sim_validation(
            analytical_scored,
            name=name,
            grasp_source=grasp_source,
            ctx=ctx,
        )

    kept_poses, kept_scores = select_diverse_grasps(
        scored_grasps,
        ctx.grasp.num_grasps,
        ctx.grasp.min_grasp_translation,
        ctx.grasp.min_grasp_rotation,
    )
    if not kept_poses:
        logger.info("Skipping {}: no grasps passed quality filters.", name)
        return quality_record(
            name,
            grasp_source,
            _QualityStats(
                candidates=int(grasps.shape[0]),
                contact_scored=contact_scored_count,
                scored=len(scored_grasps),
                sim_pass=len(scored_grasps) if sim.sim_validate else 0,
            ),
        )

    mean_score = float(np.mean(kept_scores))
    logger.info(
        "{}: source={}, kept={}/{}, mean_score={:.4f}, sim_pass={}",
        name,
        label_source,
        len(kept_poses),
        ctx.grasp.num_grasps,
        mean_score,
        sim_pass_count,
    )
    sample = {
        "point_cloud": points.astype(np.float32),
        "grasp_poses": np.stack(kept_poses, axis=0).astype(np.float32),
        "scores": np.asarray(kept_scores, dtype=np.float32),
        "object_id": name,
    }
    output_file = ctx.output_dir / f"{name}.npz"
    save_grasp_sample(output_file, sample)
    return quality_record(
        name,
        label_source,
        _QualityStats(
            candidates=int(grasps.shape[0]),
            contact_scored=contact_scored_count,
            scored=len(scored_grasps),
            kept=len(kept_poses),
            sim_pass=sim_pass_count,
            mean_score=mean_score,
        ),
    )


def _validate_synthetic_dataset_args(
    ycb_root: Path,
    grasp: _GraspCandidateConfig,
    sim: _SimValidationConfig,
) -> None:
    """Validate inputs before generating any synthetic records."""
    if not ycb_root.is_dir():
        msg = f"YCB root directory '{ycb_root}' does not exist."
        raise FileNotFoundError(msg)
    if sim.sim_validate and (sim.mjcf_root is None or sim.robot_xml is None):
        msg = "sim_validate requires mjcf_root and robot_xml"
        raise ValueError(msg)
    if sim.sim_validate and sim.gripper_close_command is None:
        msg = "sim_validate requires gripper_close_command"
        raise ValueError(msg)
    if grasp.candidate_multiplier <= 0:
        msg = "candidate_multiplier must be positive"
        raise ValueError(msg)
    if grasp.search_multiplier <= 0:
        msg = "search_multiplier must be positive"
        raise ValueError(msg)


def _process_one_object(
    name: str,
    ctx: _ObjectProcessingContext,
    required_set: set[str],
    quality_records: list[dict[str, object]],
    failures: list[str],
) -> None:
    """Process one YCB object, recording failures for required objects."""
    try:
        record = process_object_synthetic(name, ctx)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.error("Failed to generate synthetic data for '{}': {}", name, exc)
        if name in required_set:
            failures.append(f"{name}: {exc}")
        return

    if record is None:
        if name in required_set:
            failures.append(f"{name}: no mesh files found")
        return

    quality_records.append(record)
    kept = int(record["kept"])
    if kept == 0 and name in required_set:
        source = str(record["source"])
        if source == "none":
            failures.append(f"{name}: no valid grasps")
        else:
            failures.append(f"{name}: no grasps passed quality filters")


def generate_synthetic_dataset(  # noqa: PLR0913, PLR0915  # public pipeline API; defaults resolve from config
    ycb_root: Path,
    output_dir: Path,
    num_samples: int | None = None,
    num_grasps: int | None = None,
    gripper_width: float | None = None,
    *,
    seed: int | None = None,
    required_objects: list[str] | None = None,
    oversample_factor: int | None = None,
    oversample_extra: int | None = None,
    neighborhood_size: int | None = None,
    voxel_size: float | None = None,
    strict_antipodal_dot: float | None = None,
    strict_alignment_dot: float | None = None,
    relaxed_antipodal_dot: float | None = None,
    allow_relaxed: bool | None = None,
    search_multiplier: int | None = None,
    candidate_multiplier: int | None = None,
    min_grasp_translation: float | None = None,
    min_grasp_rotation: float | None = None,
    min_quality_score: float | None = None,
    friction_coefficient: float | None = None,
    collision_clearance: float | None = None,
    sim_validate: bool | None = None,
    mjcf_root: Path | None = None,
    robot_xml: Path | None = None,
    num_simulation_steps: int | None = None,
    gripper_close_command: np.ndarray | None = None,
    lift_height_threshold: float | None = None,
    max_linear_velocity: float | None = None,
    max_angular_velocity: float | None = None,
    quality_report_path: Path | None = None,
    sim_object_position: np.ndarray | None = None,
    sim_validate_require_lift: bool | None = None,
    sim_validate_require_ik: bool | None = None,
    sim_validate_min_contacts: float | None = None,
    sim_validate_fallback_analytical: bool | None = None,
    table_xml: Path | None = None,
) -> None:
    """Generate synthetic grasp dataset from YCB meshes."""
    num_samples = SYNTHETIC_DEFAULTS["num_samples"] if num_samples is None else num_samples
    num_grasps = SYNTHETIC_DEFAULTS["num_grasps"] if num_grasps is None else num_grasps
    gripper_width = SYNTHETIC_DEFAULTS["gripper_width"] if gripper_width is None else gripper_width
    seed = SYNTHETIC_DEFAULTS["seed"] if seed is None else seed
    oversample_factor = SYNTHETIC_DEFAULTS["oversample_factor"] if oversample_factor is None else oversample_factor
    oversample_extra = SYNTHETIC_DEFAULTS["oversample_extra"] if oversample_extra is None else oversample_extra
    neighborhood_size = SYNTHETIC_DEFAULTS["neighborhood_size"] if neighborhood_size is None else neighborhood_size
    voxel_size = SYNTHETIC_DEFAULTS["voxel_size"] if voxel_size is None else voxel_size
    strict_antipodal_dot = (
        SYNTHETIC_DEFAULTS["strict_antipodal_dot"] if strict_antipodal_dot is None else strict_antipodal_dot
    )
    strict_alignment_dot = (
        SYNTHETIC_DEFAULTS["strict_alignment_dot"] if strict_alignment_dot is None else strict_alignment_dot
    )
    relaxed_antipodal_dot = (
        SYNTHETIC_DEFAULTS["relaxed_antipodal_dot"] if relaxed_antipodal_dot is None else relaxed_antipodal_dot
    )
    allow_relaxed = SYNTHETIC_DEFAULTS["allow_relaxed"] if allow_relaxed is None else allow_relaxed
    search_multiplier = SYNTHETIC_DEFAULTS["search_multiplier"] if search_multiplier is None else search_multiplier
    candidate_multiplier = (
        SYNTHETIC_DEFAULTS["candidate_multiplier"] if candidate_multiplier is None else candidate_multiplier
    )
    min_grasp_translation = (
        SYNTHETIC_DEFAULTS["min_grasp_translation"] if min_grasp_translation is None else min_grasp_translation
    )
    min_grasp_rotation = SYNTHETIC_DEFAULTS["min_grasp_rotation"] if min_grasp_rotation is None else min_grasp_rotation
    min_quality_score = SYNTHETIC_DEFAULTS["min_quality_score"] if min_quality_score is None else min_quality_score
    friction_coefficient = (
        SYNTHETIC_DEFAULTS["friction_coefficient"] if friction_coefficient is None else friction_coefficient
    )
    collision_clearance = (
        SYNTHETIC_DEFAULTS["collision_clearance"] if collision_clearance is None else collision_clearance
    )
    sim_validate = SYNTHETIC_DEFAULTS["sim_validate"] if sim_validate is None else sim_validate
    num_simulation_steps = (
        SYNTHETIC_DEFAULTS["num_simulation_steps"] if num_simulation_steps is None else num_simulation_steps
    )
    lift_height_threshold = (
        SYNTHETIC_DEFAULTS["lift_height_threshold"] if lift_height_threshold is None else lift_height_threshold
    )
    max_linear_velocity = (
        SYNTHETIC_DEFAULTS["max_linear_velocity"] if max_linear_velocity is None else max_linear_velocity
    )
    max_angular_velocity = (
        SYNTHETIC_DEFAULTS["max_angular_velocity"] if max_angular_velocity is None else max_angular_velocity
    )
    sim_validate_require_lift = (
        SYNTHETIC_DEFAULTS["sim_validate_require_lift"]
        if sim_validate_require_lift is None
        else sim_validate_require_lift
    )
    sim_validate_require_ik = (
        SYNTHETIC_DEFAULTS["sim_validate_require_ik"] if sim_validate_require_ik is None else sim_validate_require_ik
    )
    sim_validate_min_contacts = (
        SYNTHETIC_DEFAULTS["sim_validate_min_contacts"]
        if sim_validate_min_contacts is None
        else sim_validate_min_contacts
    )
    sim_validate_fallback_analytical = (
        SYNTHETIC_DEFAULTS["sim_validate_fallback_analytical"]
        if sim_validate_fallback_analytical is None
        else sim_validate_fallback_analytical
    )
    if sim_object_position is None:
        sim_object_position = np.asarray(
            SIM_OBJECT_POSITION,
            dtype=np.float64,
        )
    sampling_cfg = _PointSamplingConfig(
        num_samples=num_samples,
        oversample_factor=oversample_factor,
        oversample_extra=oversample_extra,
        voxel_size=voxel_size,
        neighborhood_size=neighborhood_size,
    )
    grasp_cfg = _GraspCandidateConfig(
        num_grasps=num_grasps,
        gripper_width=gripper_width,
        strict_antipodal_dot=strict_antipodal_dot,
        strict_alignment_dot=strict_alignment_dot,
        relaxed_antipodal_dot=relaxed_antipodal_dot,
        allow_relaxed=allow_relaxed,
        search_multiplier=search_multiplier,
        candidate_multiplier=candidate_multiplier,
        min_grasp_translation=min_grasp_translation,
        min_grasp_rotation=min_grasp_rotation,
        min_quality_score=min_quality_score,
        friction_coefficient=friction_coefficient,
        collision_clearance=collision_clearance,
    )
    sim_cfg = _SimValidationConfig(
        sim_validate=sim_validate,
        mjcf_root=mjcf_root,
        robot_xml=robot_xml,
        table_xml=table_xml,
        num_simulation_steps=num_simulation_steps,
        gripper_close_command=gripper_close_command,
        lift_height_threshold=lift_height_threshold,
        max_linear_velocity=max_linear_velocity,
        max_angular_velocity=max_angular_velocity,
        sim_validate_require_lift=sim_validate_require_lift,
        sim_validate_require_ik=sim_validate_require_ik,
        sim_validate_min_contacts=sim_validate_min_contacts,
        sim_validate_fallback_analytical=sim_validate_fallback_analytical,
    )
    _validate_synthetic_dataset_args(ycb_root, grasp_cfg, sim_cfg)

    required_set = set(required_objects or [])
    failures: list[str] = []
    quality_records: list[dict[str, object]] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    gripper_point_cloud = default_gripper_point_cloud()

    object_position = (
        sim_object_position if sim_object_position is not None else np.asarray(SIM_OBJECT_POSITION, dtype=np.float64)
    )
    if object_position.shape != (3,):
        msg = "sim_object_position must have shape (3,)"
        raise ValueError(msg)
    ctx = _ObjectProcessingContext(
        ycb_root=ycb_root,
        output_dir=output_dir,
        rng=rng,
        gripper_point_cloud=gripper_point_cloud,
        object_to_world=make_transform(np.eye(3, dtype=np.float64), object_position),
        sampling=sampling_cfg,
        grasp=grasp_cfg,
        sim=sim_cfg,
    )

    for name in list_ycb_objects(ycb_root):
        _process_one_object(name, ctx, required_set, quality_records, failures)

    if quality_report_path is not None:
        quality_report_path.parent.mkdir(parents=True, exist_ok=True)
        quality_report_path.write_text(json.dumps(quality_records, indent=2), encoding="utf-8")

    if failures:
        joined = "; ".join(failures)
        msg = f"Required YCB objects failed to generate synthetic data: {joined}"
        raise RuntimeError(msg)
