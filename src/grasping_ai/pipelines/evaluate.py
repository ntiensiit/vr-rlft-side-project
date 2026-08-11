from collections.abc import Callable
from pathlib import Path

import numpy as np

GraspEvaluation = dict[str, float | bool]


def evaluate_generated_grasps(
    grasp_poses: np.ndarray,
    object_point_cloud: np.ndarray,
    gripper_point_cloud: np.ndarray,
    contact_set_provider: Callable[[np.ndarray], list[dict[str, np.ndarray]]],
    friction_coefficient: float,
    lift_height_threshold: float,
) -> list[GraspEvaluation]:
    """Evaluate a set of generated grasps using a common evaluation pipeline.

    Args:
        grasp_poses: Grasp poses with shape ``(K, 4, 4)``.
        object_point_cloud: Object point cloud used for collision checking.
        gripper_point_cloud: Gripper point cloud used for collision checking.
        contact_set_provider: Callable returning contacts for a given grasp pose.
        friction_coefficient: Friction coefficient used by force-closure analysis.
        lift_height_threshold: Height threshold used by lift-success evaluation.

    Returns:
        A list of per-grasp evaluation dictionaries.
    """
    raise NotImplementedError


def aggregate_evaluation_results(
    per_object_results: dict[str, list[GraspEvaluation]],
) -> dict[str, float]:
    """Aggregate per-object evaluation results into summary metrics.

    Args:
        per_object_results: Mapping from object identifier to per-grasp results.

    Returns:
        A dictionary of aggregated metric names and values.
    """
    raise NotImplementedError


def write_evaluation_report(report_path: Path, results: dict[str, float]) -> None:
    """Persist a human-readable evaluation report to disk.

    Args:
        report_path: Destination path for the report file.
        results: Aggregated evaluation metrics to serialize.
    """
    raise NotImplementedError
