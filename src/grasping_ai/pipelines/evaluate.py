"""Evaluate generated grasps against quality metrics."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
from grasping_ai.evaluation.collision import (
    build_collision_checker,
    check_collision,
    filter_collision_free_grasps,
    generate_analytical_contacts,
)
from grasping_ai.evaluation.force_closure import (
    build_force_closure_judge,
    compute_grasp_quality,
    compute_grasp_wrench_matrix,
    evaluate_force_closure,
    load_contact_set,
)
from grasping_ai.evaluation.metrics import aggregate_grasp_success_rate
from grasping_ai.utils.path_validation import require_path

GRASP_POSES_NDIM = int(FLATTENED_YAML_CONFIG.get("grasp.poses_ndim", 3))
POINT_CLOUD_NDIM = int(FLATTENED_YAML_CONFIG.get("geometry.point_cloud_ndim", 2))
SE3_MATRIX_SHAPE = tuple(int(v) for v in FLATTENED_YAML_CONFIG.get("grasp.se3_matrix_shape", [4, 4]))
FRICTION_COEFFICIENT = float(FLATTENED_YAML_CONFIG.get("metrics.friction_coefficient", 0.5))
LIFT_HEIGHT_THRESHOLD = float(FLATTENED_YAML_CONFIG.get("metrics.lift_height_threshold", 0.05))
CLEARANCE = float(FLATTENED_YAML_CONFIG.get("metrics.collision_clearance", 0.005))
WRENCH_REGULARIZATION = float(FLATTENED_YAML_CONFIG.get("metrics.wrench_regularization", 1.0))
FILTER_COLLISIONS = bool(FLATTENED_YAML_CONFIG.get("evaluation.filter_collisions", False))

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _validate_evaluation_args(
    grasp_poses: np.ndarray,
    friction_coefficient: float,
    lift_height_threshold: float,
    clearance: float,
    wrench_regularization: float,
) -> np.ndarray:
    """Validate evaluation inputs and normalize ``grasp_poses`` to ``(K, 4, 4)``."""
    if grasp_poses.ndim == POINT_CLOUD_NDIM:
        if grasp_poses.shape == (4, 4):
            grasp_poses = grasp_poses.reshape(1, 4, 4)
        else:
            msg = "grasp_poses must have shape (K, 4, 4) or (4, 4)"
            raise ValueError(msg)

    if grasp_poses.ndim != GRASP_POSES_NDIM or grasp_poses.shape[1:] != SE3_MATRIX_SHAPE:
        msg = "grasp_poses must have shape (K, 4, 4)"
        raise ValueError(msg)
    if friction_coefficient < 0:
        msg = "friction_coefficient must be non-negative"
        raise ValueError(msg)
    if lift_height_threshold < 0:
        msg = "lift_height_threshold must be non-negative"
        raise ValueError(msg)
    if clearance < 0:
        msg = "clearance must be non-negative"
        raise ValueError(msg)
    if wrench_regularization < 0:
        msg = "wrench_regularization must be non-negative"
        raise ValueError(msg)
    return grasp_poses


def evaluate_generated_grasps(  # noqa: PLR0913  # public evaluation API; tests/scripts pass metric options as keywords
    grasp_poses: np.ndarray,
    object_point_cloud: np.ndarray,
    gripper_point_cloud: np.ndarray,
    contact_set_provider: Callable[[np.ndarray], list[dict[str, np.ndarray]]] | None = None,
    *,
    friction_coefficient: float = FRICTION_COEFFICIENT,
    lift_height_threshold: float = LIFT_HEIGHT_THRESHOLD,
    clearance: float = CLEARANCE,
    wrench_regularization: float = WRENCH_REGULARIZATION,
    contact_path: Path | None = None,
    filter_collisions: bool = FILTER_COLLISIONS,
) -> list[dict[str, float | bool]]:
    """Evaluate a set of generated grasps using a common evaluation pipeline.

    Args:
        grasp_poses: Grasp poses with shape ``(K, 4, 4)``.
        object_point_cloud: Object point cloud used for collision checking.
        gripper_point_cloud: Gripper point cloud used for collision checking.
        contact_set_provider: Callable returning contacts for a given grasp pose.
            If None, contacts are generated analytically.
        friction_coefficient: Friction coefficient used by force-closure analysis.
        lift_height_threshold: Height threshold used by lift-success evaluation.
        clearance: Collision clearance distance.
        wrench_regularization: Wrench regularization parameter for force-closure.
        contact_path: Optional serialized contact set used for every grasp when
            ``contact_set_provider`` is not supplied.
        filter_collisions: When ``True``, evaluate only collision-free grasps.

    Returns:
        A list of per-grasp evaluation dictionaries.

    Raises:
        ValueError: If ``grasp_poses`` shape or metric parameters are invalid.
    """
    grasp_poses = _validate_evaluation_args(
        grasp_poses,
        friction_coefficient,
        lift_height_threshold,
        clearance,
        wrench_regularization,
    )

    collision_checker = build_collision_checker(object_point_cloud, gripper_point_cloud, clearance=clearance)
    logger.info("Evaluating {} grasp poses", grasp_poses.shape[0])
    if filter_collisions:
        grasp_poses = filter_collision_free_grasps(collision_checker, grasp_poses)
        logger.info("Filtered to {} collision-free grasp poses", grasp_poses.shape[0])
        if grasp_poses.shape[0] == 0:
            return []

    if contact_set_provider is None and contact_path is not None:
        file_contacts = load_contact_set(contact_path)

        def contact_set_provider(pose: np.ndarray) -> list[dict[str, np.ndarray]]:
            del pose
            return file_contacts

    fc_judge = build_force_closure_judge(friction_coefficient, wrench_regularization=wrench_regularization)

    results: list[dict[str, float | bool]] = []
    for i in range(grasp_poses.shape[0]):
        pose = grasp_poses[i]
        collision_free = check_collision(collision_checker, pose)
        if contact_set_provider is not None:
            contacts = contact_set_provider(pose)
        else:
            contacts = generate_analytical_contacts(object_point_cloud, gripper_point_cloud, pose, clearance)

        force_closure = evaluate_force_closure(fc_judge, contacts)
        wrench_matrix = compute_grasp_wrench_matrix(contacts, friction_coefficient)
        wrench_rank = float(np.linalg.matrix_rank(wrench_matrix)) if wrench_matrix.size else 0.0
        grasp_quality = compute_grasp_quality(contacts, friction_coefficient)
        grasp_success = bool(collision_free and force_closure)

        results.append(
            {
                "collision_free": collision_free,
                "force_closure": force_closure,
                "grasp_success": grasp_success,
                "grasp_quality": float(grasp_quality),
                "wrench_rank": wrench_rank,
            },
        )
    return results


def aggregate_evaluation_results(
    per_object_results: dict[str, list[dict[str, float | bool]]],
) -> dict[str, float]:
    """Aggregate per-object evaluation results into summary metrics.

    Args:
        per_object_results: Mapping from object identifier to per-grasp results.

    Returns:
        A dictionary of aggregated metric names and values.

    Raises:
        TypeError: If ``per_object_results`` is not a dictionary.
    """
    if not isinstance(per_object_results, dict):
        msg = "per_object_results must be a dictionary"
        raise TypeError(msg)

    total_grasps = 0
    collision_free_count = 0
    force_closure_count = 0
    grasp_success_count = 0
    qualities = []

    for results in per_object_results.values():
        for res in results:
            total_grasps += 1
            if res.get("collision_free"):
                collision_free_count += 1
            if res.get("force_closure"):
                force_closure_count += 1
            if res.get("grasp_success"):
                grasp_success_count += 1

            q_val = res.get("grasp_quality")
            if q_val is not None:
                qualities.append(float(q_val))

    if total_grasps == 0:
        return {
            "success_rate": 0.0,
            "collision_free_rate": 0.0,
            "force_closure_rate": 0.0,
            "object_success_rate": 0.0,
            "mean_grasp_quality": 0.0,
            "min_grasp_quality": 0.0,
            "max_grasp_quality": 0.0,
        }

    if qualities:
        mean_q = float(np.mean(qualities))
        min_q = float(np.min(qualities))
        max_q = float(np.max(qualities))
    else:
        mean_q = 0.0
        min_q = 0.0
        max_q = 0.0

    per_object_success = {
        object_id: any(bool(res.get("grasp_success")) for res in results)
        for object_id, results in per_object_results.items()
    }
    object_success_rate = aggregate_grasp_success_rate(per_object_success)

    return {
        "success_rate": float(grasp_success_count / total_grasps),
        "collision_free_rate": float(collision_free_count / total_grasps),
        "force_closure_rate": float(force_closure_count / total_grasps),
        "object_success_rate": object_success_rate,
        "mean_grasp_quality": mean_q,
        "min_grasp_quality": min_q,
        "max_grasp_quality": max_q,
    }


def write_jsonl_records(
    output_path: Path,
    records: list[dict[str, object]],
    mode: str = "w",
) -> None:
    """Write JSON Lines records to disk.

    Args:
        output_path: Destination ``.jsonl`` file path.
        records: Ordered mapping objects, one per output line.
        mode: Open mode ('w' to write/overwrite, 'a' to append).

    Raises:
        TypeError: If ``output_path`` is not a ``pathlib.Path`` instance.
        ValueError: If writing the file fails.
    """
    require_path(output_path, "output_path")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def json_default(value: object) -> object:
        """Convert NumPy report values to JSON-native containers/scalars."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    try:
        with output_path.open(mode, encoding="utf-8") as fp:
            for record in records:
                fp.write(json.dumps(record, allow_nan=True, default=json_default))
                fp.write("\n")
    except Exception as e:
        msg = f"Failed to write JSONL records: {e}"
        raise ValueError(msg) from e


def read_jsonl_records(input_path: Path) -> list[dict[str, object]]:
    """Read JSON Lines records from disk.

    Args:
        input_path: Source ``.jsonl`` file path.

    Returns:
        Parsed mapping objects in file order.

    Raises:
        TypeError: If ``input_path`` is not a ``pathlib.Path`` instance, or a
            line does not decode to a mapping.
        ValueError: If the file cannot be read or parsed.
    """
    require_path(input_path, "input_path")

    try:
        with input_path.open(encoding="utf-8") as fp:
            lines = [line.strip() for line in fp]
    except OSError as e:
        msg = f"Failed to read JSONL records: {e}"
        raise ValueError(msg) from e

    records: list[dict[str, object]] = []
    for line_number, stripped_line in enumerate(lines, start=1):
        if not stripped_line:
            continue
        try:
            loaded = json.loads(stripped_line)
        except json.JSONDecodeError as e:
            msg = f"Failed to read JSONL records: {e}"
            raise ValueError(msg) from e
        if not isinstance(loaded, dict):
            msg = f"JSONL line {line_number} in {input_path} must be a mapping"
            raise TypeError(msg)
        records.append(loaded)
    return records


def write_evaluation_report(
    report_path: Path,
    results: dict[str, float],
    experiment_log_dir: Path | None = None,
    per_object_results: dict[str, dict[str, float]] | None = None,
) -> None:
    """Persist an evaluation report as JSON Lines.

    When ``per_object_results`` is supplied, one ``object`` record is written
    per object followed by a final ``summary`` record. Otherwise only the
    summary record is written.

    Args:
        report_path: Destination path for the report file.
        results: Aggregated evaluation metrics to serialize.
        experiment_log_dir: Optional path to write TensorBoard experiment events.
        per_object_results: Optional per-object aggregated metrics keyed by
            object identifier.

    Raises:
        TypeError: If ``report_path`` is not a ``pathlib.Path`` instance.
        ValueError: If writing the report fails.
    """
    require_path(report_path, "report_path")

    records: list[dict[str, object]] = []
    if per_object_results is not None:
        for object_id, metrics in per_object_results.items():
            records.append(
                {
                    "record_type": "object",
                    "object_id": object_id,
                    **metrics,
                },
            )
    records.append({"record_type": "summary", **results})
    write_jsonl_records(report_path, records, mode="a")

    if experiment_log_dir is not None:
        writer = SummaryWriter(log_dir=str(experiment_log_dir))
        try:
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(k, float(v), global_step=0)
        finally:
            writer.close()
