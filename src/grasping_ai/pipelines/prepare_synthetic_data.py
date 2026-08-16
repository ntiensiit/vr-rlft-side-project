"""Synthetic grasp dataset generation from YCB meshes."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from loguru import logger

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
from grasping_ai.data.pointcloud_dataset import (
    GraspSample,
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

NUM_SAMPLES = int(FLATTENED_YAML_CONFIG.get("synthetic.num_samples"))
NUM_GRASPS = int(FLATTENED_YAML_CONFIG.get("synthetic.num_grasps"))
GRIPPER_WIDTH = float(FLATTENED_YAML_CONFIG.get("synthetic.gripper_width"))
SEED = int(FLATTENED_YAML_CONFIG.get("synthetic.seed"))
OVERSAMPLE_FACTOR = int(FLATTENED_YAML_CONFIG.get("synthetic.oversample_factor"))
OVERSAMPLE_EXTRA = int(FLATTENED_YAML_CONFIG.get("synthetic.oversample_extra"))
NEIGHBORHOOD_SIZE = int(FLATTENED_YAML_CONFIG.get("synthetic.neighborhood_size"))
VOXEL_SIZE = float(FLATTENED_YAML_CONFIG.get("synthetic.voxel_size"))
STRICT_ANTIPODAL_DOT = float(FLATTENED_YAML_CONFIG.get("synthetic.strict_antipodal_dot"))
STRICT_ALIGNMENT_DOT = float(FLATTENED_YAML_CONFIG.get("synthetic.strict_alignment_dot"))
RELAXED_ANTIPODAL_DOT = float(FLATTENED_YAML_CONFIG.get("synthetic.relaxed_antipodal_dot"))
ALLOW_RELAXED = bool(FLATTENED_YAML_CONFIG.get("synthetic.allow_relaxed"))
SEARCH_MULTIPLIER = int(FLATTENED_YAML_CONFIG.get("synthetic.search_multiplier"))
CANDIDATE_MULTIPLIER = int(FLATTENED_YAML_CONFIG.get("synthetic.candidate_multiplier"))
MIN_GRASP_TRANSLATION = float(FLATTENED_YAML_CONFIG.get("synthetic.min_grasp_translation"))
MIN_GRASP_ROTATION = float(FLATTENED_YAML_CONFIG.get("synthetic.min_grasp_rotation"))
MIN_QUALITY_SCORE = float(FLATTENED_YAML_CONFIG.get("synthetic.min_quality_score"))
FRICTION_COEFFICIENT = float(FLATTENED_YAML_CONFIG.get("synthetic.friction_coefficient"))
COLLISION_CLEARANCE = float(FLATTENED_YAML_CONFIG.get("synthetic.collision_clearance"))
SIM_VALIDATE = bool(FLATTENED_YAML_CONFIG.get("synthetic.sim_validate"))
NUM_SIMULATION_STEPS = int(FLATTENED_YAML_CONFIG.get("synthetic.num_simulation_steps"))
LIFT_HEIGHT_THRESHOLD = float(FLATTENED_YAML_CONFIG.get("synthetic.lift_height_threshold"))
MAX_LINEAR_VELOCITY = float(FLATTENED_YAML_CONFIG.get("synthetic.max_linear_velocity"))
MAX_ANGULAR_VELOCITY = float(FLATTENED_YAML_CONFIG.get("synthetic.max_angular_velocity"))
SIM_VALIDATE_REQUIRE_LIFT = bool(FLATTENED_YAML_CONFIG.get("synthetic.sim_validate_require_lift"))
SIM_VALIDATE_REQUIRE_IK = bool(FLATTENED_YAML_CONFIG.get("synthetic.sim_validate_require_ik"))
SIM_VALIDATE_MIN_CONTACTS = float(FLATTENED_YAML_CONFIG.get("synthetic.sim_validate_min_contacts"))
SIM_VALIDATE_FALLBACK_ANALYTICAL = bool(
    FLATTENED_YAML_CONFIG.get("synthetic.sim_validate_fallback_analytical"),
)
SIM_OBJECT_POSITION = tuple(FLATTENED_YAML_CONFIG.get_path("synthetic", "sim_object_position"))

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


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
    if mesh_path.is_dir():
        candidates = list(mesh_path.rglob("*.obj")) + list(mesh_path.rglob("*.ply"))
        if candidates:
            return candidates[0]
    return None


def sample_object_points(  # noqa: PLR0913, PLR0917  # sampling parameters are intentionally explicit
    mesh_path: Path,
    rng: np.random.Generator,
    num_samples: int,
    oversample_factor: int,
    oversample_extra: int,
    voxel_size: float,
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


def generate_labeled_analytical_grasps(  # noqa: PLR0913, PLR0917  # generation parameters are explicit
    points: np.ndarray,
    normals: np.ndarray,
    candidate_count: int,
    rng: np.random.Generator,
    gripper_width: float,
    strict_antipodal_dot: float,
    strict_alignment_dot: float,
    search_multiplier: int,
    allow_relaxed: bool,  # noqa: FBT001
    relaxed_antipodal_dot: float,
) -> tuple[np.ndarray, str]:
    """Generate analytical grasps and report whether strict or relaxed succeeded."""
    grasps = generate_analytical_grasps(
        points,
        normals,
        candidate_count,
        gripper_width,
        rng,
        strict_antipodal_dot=strict_antipodal_dot,
        strict_alignment_dot=strict_alignment_dot,
        search_multiplier=search_multiplier,
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
        strict_antipodal_dot=strict_antipodal_dot,
        strict_alignment_dot=strict_alignment_dot,
        search_multiplier=search_multiplier,
    )
    if grasps.shape[0] > 0:
        return grasps, "relaxed"
    return grasps, "none"


def sim_validation_passes(
    outcome: Mapping[str, object],
    *,
    sim_validate_require_ik: bool,
    sim_validate_require_lift: bool,
    sim_validate_min_contacts: float,
    lift_height_threshold: float,
) -> bool:
    """Return whether a MuJoCo grasp outcome satisfies validation criteria."""
    fk_error = float(cast("float", outcome.get("fk_position_error", float("inf"))))
    contact_count = float(cast("float", outcome.get("contact_count", 0.0)))
    ik_ok = np.isfinite(fk_error) if sim_validate_require_ik else True
    contact_ok = contact_count >= sim_validate_min_contacts
    lift_ok = True
    if sim_validate_require_lift:
        initial_height = float(cast("float", outcome.get("initial_height", 0.0)))
        final_height = float(cast("float", outcome.get("final_height", 0.0)))
        lift_ok = (final_height - initial_height) >= lift_height_threshold
    return ik_ok and contact_ok and lift_ok


def apply_sim_validation(  # noqa: PLR0913  # grouped configuration dictionaries keep this API compact
    analytical_scored: list[tuple[np.ndarray, float]],
    *,
    name: str,
    grasp_source: str,
    shared: dict[str, Any],
    grasp: dict[str, Any],
    simulation: dict[str, Any],
) -> tuple[list[tuple[np.ndarray, float]], str, int]:
    """Filter analytically scored grasps using MuJoCo simulation outcomes."""
    sim_filtered: list[tuple[np.ndarray, float]] = []
    table_xml_path = (
        simulation["table_xml"]
        if simulation["table_xml"] is not None and simulation["table_xml"].is_file()
        else None
    )
    for pose, score in analytical_scored:
        world_pose = convert_grasps_to_world_frame(pose.reshape(1, 4, 4), shared["object_to_world"])[0]
        outcome = simulate_grasp(
            world_pose,
            name,
            cast("Path", simulation["mjcf_root"]),
            cast("Path", simulation["robot_xml"]),
            table_xml_path=table_xml_path,
            num_simulation_steps=simulation["num_simulation_steps"],
            gripper_close_command=cast("np.ndarray", simulation["gripper_close_command"]),
            lift_height_threshold=simulation["lift_height_threshold"],
            max_linear_velocity=simulation["max_linear_velocity"],
            max_angular_velocity=simulation["max_angular_velocity"],
            grasp_width=grasp["gripper_width"],
            quiet=True,
        )
        if sim_validation_passes(
            outcome,
            sim_validate_require_ik=simulation["sim_validate_require_ik"],
            sim_validate_require_lift=simulation["sim_validate_require_lift"],
            sim_validate_min_contacts=simulation["sim_validate_min_contacts"],
            lift_height_threshold=simulation["lift_height_threshold"],
        ):
            sim_filtered.append((pose, score))

    sim_pass_count = len(sim_filtered)
    if sim_filtered:
        return sim_filtered, f"{grasp_source}+sim", sim_pass_count
    if simulation["sim_validate_fallback_analytical"] and analytical_scored:
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
    stats: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Build a per-object synthetic dataset quality record."""
    resolved = stats or {
        "candidates": 0,
        "contact_scored": 0,
        "scored": 0,
        "kept": 0,
        "sim_pass": 0,
        "mean_score": 0.0,
    }
    return {
        "object_id": object_id,
        "source": source,
        "candidates": resolved["candidates"],
        "contact_scored": resolved["contact_scored"],
        "scored": resolved["scored"],
        "kept": resolved["kept"],
        "sim_pass": resolved["sim_pass"],
        "mean_score": resolved["mean_score"],
    }


def process_object_synthetic(
    name: str,
    shared: dict[str, Any],
    sampling: dict[str, Any],
    grasp: dict[str, Any],
    simulation: dict[str, Any],
) -> dict[str, object] | None:
    """Generate and persist one synthetic grasp dataset record for an object."""
    mesh_path = resolve_mesh_path(shared["ycb_root"], name)
    if mesh_path is None:
        logger.info("Skipping {}: no mesh files found.", name)
        return None

    points = sample_object_points(
        mesh_path,
        shared["rng"],
        sampling["num_samples"],
        sampling["oversample_factor"],
        sampling["oversample_extra"],
        sampling["voxel_size"],
    )
    normals = estimate_point_cloud_normals(points, neighborhood_size=sampling["neighborhood_size"])
    candidate_count = grasp["num_grasps"] * grasp["candidate_multiplier"]
    grasps, grasp_source = generate_labeled_analytical_grasps(
        points,
        normals,
        candidate_count,
        shared["rng"],
        grasp["gripper_width"],
        grasp["strict_antipodal_dot"],
        grasp["strict_alignment_dot"],
        grasp["search_multiplier"],
        grasp["allow_relaxed"],
        grasp["relaxed_antipodal_dot"],
    )
    if grasps.shape[0] == 0:
        logger.info("Skipping {}: no valid grasps found.", name)
        return quality_record(name, "none")

    analytical_scored = score_grasp_poses_by_contacts(
        grasps,
        points,
        shared["gripper_point_cloud"],
        grasp["friction_coefficient"],
        grasp["collision_clearance"],
        grasp["min_quality_score"],
    )
    contact_scored_count = len(analytical_scored)
    scored_grasps = list(analytical_scored)
    sim_pass_count = 0
    label_source = grasp_source

    sim_assets_ready = (
        simulation["mjcf_root"] is not None
        and simulation["robot_xml"] is not None
        and simulation["gripper_close_command"] is not None
    )
    if simulation["sim_validate"] and sim_assets_ready:
        scored_grasps, label_source, sim_pass_count = apply_sim_validation(
            analytical_scored,
            name=name,
            grasp_source=grasp_source,
            shared=shared,
            grasp=grasp,
            simulation=simulation,
        )

    kept_poses, kept_scores = select_diverse_grasps(
        scored_grasps,
        grasp["num_grasps"],
        grasp["min_grasp_translation"],
        grasp["min_grasp_rotation"],
    )
    if not kept_poses:
        logger.info("Skipping {}: no grasps passed quality filters.", name)
        return quality_record(
            name,
            grasp_source,
            {
                "candidates": int(grasps.shape[0]),
                "contact_scored": contact_scored_count,
                "scored": len(scored_grasps),
                "sim_pass": len(scored_grasps) if simulation["sim_validate"] else 0,
            },
        )

    mean_score = float(np.mean(kept_scores))
    logger.info(
        "{}: source={}, kept={}/{}, mean_score={:.4f}, sim_pass={}",
        name,
        label_source,
        len(kept_poses),
        grasp["num_grasps"],
        mean_score,
        sim_pass_count,
    )
    sample: GraspSample = {
        "point_cloud": points.astype(np.float32),
        "grasp_poses": np.stack(kept_poses, axis=0).astype(np.float32),
        "scores": np.asarray(kept_scores, dtype=np.float32),
        "object_id": name,
    }
    output_file = shared["output_dir"] / f"{name}.npz"
    save_grasp_sample(output_file, sample)
    return quality_record(
        name,
        label_source,
        {
            "candidates": int(grasps.shape[0]),
            "contact_scored": contact_scored_count,
            "scored": len(scored_grasps),
            "kept": len(kept_poses),
            "sim_pass": sim_pass_count,
            "mean_score": mean_score,
        },
    )


def _validate_synthetic_dataset_args(  # noqa: PLR0913, PLR0917  # validation parameters are explicit
    ycb_root: Path,
    candidate_multiplier: int,
    search_multiplier: int,
    sim_validate: bool,  # noqa: FBT001
    mjcf_root: Path | None,
    robot_xml: Path | None,
    gripper_close_command: np.ndarray | None,
) -> None:
    """Validate inputs before generating any synthetic records."""
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


def _process_one_object(  # noqa: PLR0913, PLR0917  # grouped configuration dictionaries keep this API compact
    name: str,
    shared: dict[str, Any],
    sampling: dict[str, Any],
    grasp: dict[str, Any],
    simulation: dict[str, Any],
    required_set: set[str],
    quality_records: list[dict[str, object]],
    failures: list[str],
) -> None:
    """Process one YCB object, recording failures for required objects."""
    try:
        record = process_object_synthetic(name, shared, sampling, grasp, simulation)
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
    kept = int(cast("int", record["kept"]))
    if kept == 0 and name in required_set:
        source = str(record["source"])
        if source == "none":
            failures.append(f"{name}: no valid grasps")
        else:
            failures.append(f"{name}: no grasps passed quality filters")


def generate_synthetic_dataset(  # noqa: PLR0913  # preserve legacy keyword compatibility
    ycb_root: Path,
    output_dir: Path,
    *,
    required_objects: list[str] | None = None,
    quality_report_path: Path | None = None,
    sim_object_position: np.ndarray | None = None,
    table_xml: Path | None = None,
    **options: Any,  # noqa: ANN401
) -> None:
    """Generate synthetic grasp dataset from YCB meshes."""
    seed = options.pop("seed", SEED)
    sampling = {
        "num_samples": options.pop("num_samples", NUM_SAMPLES),
        "oversample_factor": options.pop("oversample_factor", OVERSAMPLE_FACTOR),
        "oversample_extra": options.pop("oversample_extra", OVERSAMPLE_EXTRA),
        "neighborhood_size": options.pop("neighborhood_size", NEIGHBORHOOD_SIZE),
        "voxel_size": options.pop("voxel_size", VOXEL_SIZE),
    }
    grasp = {
        "num_grasps": options.pop("num_grasps", NUM_GRASPS),
        "gripper_width": options.pop("gripper_width", GRIPPER_WIDTH),
        "strict_antipodal_dot": options.pop("strict_antipodal_dot", STRICT_ANTIPODAL_DOT),
        "strict_alignment_dot": options.pop("strict_alignment_dot", STRICT_ALIGNMENT_DOT),
        "relaxed_antipodal_dot": options.pop("relaxed_antipodal_dot", RELAXED_ANTIPODAL_DOT),
        "allow_relaxed": options.pop("allow_relaxed", ALLOW_RELAXED),
        "search_multiplier": options.pop("search_multiplier", SEARCH_MULTIPLIER),
        "candidate_multiplier": options.pop("candidate_multiplier", CANDIDATE_MULTIPLIER),
        "min_grasp_translation": options.pop("min_grasp_translation", MIN_GRASP_TRANSLATION),
        "min_grasp_rotation": options.pop("min_grasp_rotation", MIN_GRASP_ROTATION),
        "min_quality_score": options.pop("min_quality_score", MIN_QUALITY_SCORE),
        "friction_coefficient": options.pop("friction_coefficient", FRICTION_COEFFICIENT),
        "collision_clearance": options.pop("collision_clearance", COLLISION_CLEARANCE),
    }
    simulation = {
        "sim_validate": options.pop("sim_validate", SIM_VALIDATE),
        "mjcf_root": options.pop("mjcf_root", None),
        "robot_xml": options.pop("robot_xml", None),
        "num_simulation_steps": options.pop("num_simulation_steps", NUM_SIMULATION_STEPS),
        "gripper_close_command": options.pop("gripper_close_command", None),
        "lift_height_threshold": options.pop("lift_height_threshold", LIFT_HEIGHT_THRESHOLD),
        "max_linear_velocity": options.pop("max_linear_velocity", MAX_LINEAR_VELOCITY),
        "max_angular_velocity": options.pop("max_angular_velocity", MAX_ANGULAR_VELOCITY),
        "sim_validate_require_lift": options.pop("sim_validate_require_lift", SIM_VALIDATE_REQUIRE_LIFT),
        "sim_validate_require_ik": options.pop("sim_validate_require_ik", SIM_VALIDATE_REQUIRE_IK),
        "sim_validate_min_contacts": options.pop("sim_validate_min_contacts", SIM_VALIDATE_MIN_CONTACTS),
        "sim_validate_fallback_analytical": options.pop(
            "sim_validate_fallback_analytical", SIM_VALIDATE_FALLBACK_ANALYTICAL,
        ),
        "table_xml": table_xml,
    }
    if options:
        unexpected = ", ".join(sorted(options))
        raise TypeError(f"Unexpected synthetic dataset options: {unexpected}")
    if sim_object_position is None:
        sim_object_position = np.asarray(
            SIM_OBJECT_POSITION,
            dtype=np.float64,
        )
    _validate_synthetic_dataset_args(
        ycb_root,
        grasp["candidate_multiplier"],
        grasp["search_multiplier"],
        simulation["sim_validate"],
        simulation["mjcf_root"],
        simulation["robot_xml"],
        simulation["gripper_close_command"],
    )

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
    shared = {
        "ycb_root": ycb_root,
        "output_dir": output_dir,
        "rng": rng,
        "gripper_point_cloud": gripper_point_cloud,
        "object_to_world": make_transform(np.eye(3, dtype=np.float64), object_position),
    }
    for name in list_ycb_objects(ycb_root):
        _process_one_object(name, shared, sampling, grasp, simulation, required_set, quality_records, failures)

    if quality_report_path is not None:
        quality_report_path.parent.mkdir(parents=True, exist_ok=True)
        quality_report_path.write_text(json.dumps(quality_records, indent=2), encoding="utf-8")

    if failures:
        joined = "; ".join(failures)
        msg = f"Required YCB objects failed to generate synthetic data: {joined}"
        raise RuntimeError(msg)
