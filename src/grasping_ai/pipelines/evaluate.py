import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from grasping_ai.evaluation.collision import build_collision_checker, check_collision
from grasping_ai.evaluation.force_closure import build_force_closure_judge, evaluate_force_closure

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
    if grasp_poses.ndim == 2:
        if grasp_poses.shape == (4, 4):
            grasp_poses = grasp_poses.reshape(1, 4, 4)
        else:
            raise ValueError("grasp_poses must have shape (K, 4, 4) or (4, 4)")

    if grasp_poses.ndim != 3 or grasp_poses.shape[1:] != (4, 4):
        raise ValueError("grasp_poses must have shape (K, 4, 4)")
    if friction_coefficient < 0:
        raise ValueError("friction_coefficient must be non-negative")
    if lift_height_threshold < 0:
        raise ValueError("lift_height_threshold must be non-negative")

    collision_checker = build_collision_checker(
        object_point_cloud, gripper_point_cloud, clearance=0.005
    )
    fc_judge = build_force_closure_judge(friction_coefficient, wrench_regularization=1.0)

    evaluations = []
    for i in range(grasp_poses.shape[0]):
        pose = grasp_poses[i]
        collision_free = check_collision(collision_checker, pose)

        contacts = contact_set_provider(pose)
        force_closure = evaluate_force_closure(fc_judge, contacts)

        # Grasp is successful if it is collision-free and force-closed
        lift_success = bool(collision_free and force_closure)

        eval_dict: GraspEvaluation = {
            "collision_free": collision_free,
            "force_closure": force_closure,
            "lift_success": lift_success,
        }
        evaluations.append(eval_dict)

    return evaluations


def aggregate_evaluation_results(
    per_object_results: dict[str, list[GraspEvaluation]],
) -> dict[str, float]:
    """Aggregate per-object evaluation results into summary metrics.

    Args:
        per_object_results: Mapping from object identifier to per-grasp results.

    Returns:
        A dictionary of aggregated metric names and values.
    """
    if not isinstance(per_object_results, dict):
        raise TypeError("per_object_results must be a dictionary")

    total_grasps = 0
    collision_free_count = 0
    force_closure_count = 0
    lift_success_count = 0

    for results in per_object_results.values():
        for res in results:
            total_grasps += 1
            if res.get("collision_free"):
                collision_free_count += 1
            if res.get("force_closure"):
                force_closure_count += 1
            if res.get("lift_success"):
                lift_success_count += 1

    if total_grasps == 0:
        return {
            "success_rate": 0.0,
            "collision_free_rate": 0.0,
            "force_closure_rate": 0.0,
        }

    return {
        "success_rate": float(lift_success_count / total_grasps),
        "collision_free_rate": float(collision_free_count / total_grasps),
        "force_closure_rate": float(force_closure_count / total_grasps),
    }


def write_evaluation_report(
    report_path: Path,
    results: dict[str, float],
    experiment_log_dir: Path | None = None,
) -> None:
    """Persist a human-readable evaluation report to disk.

    Args:
        report_path: Destination path for the report file.
        results: Aggregated evaluation metrics to serialize.
        experiment_log_dir: Optional path to write TensorBoard experiment events.
    """
    if not isinstance(report_path, Path):
        raise TypeError("report_path must be a pathlib.Path instance")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with report_path.open("w") as fp:
            json.dump(results, fp, indent=4)
    except Exception as e:
        raise ValueError(f"Failed to write evaluation report: {e}") from e

    if experiment_log_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(experiment_log_dir))
        try:
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(k, float(v), global_step=0)
        finally:
            writer.close()
