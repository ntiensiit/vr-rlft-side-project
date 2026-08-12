import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from grasping_ai.evaluation.collision import (
    CollisionChecker,
    build_collision_checker,
    check_collision,
    filter_collision_free_grasps,
    generate_analytical_contacts,
)
from grasping_ai.evaluation.force_closure import (
    ForceClosureJudge,
    build_force_closure_judge,
    compute_grasp_quality,
    compute_grasp_wrench_matrix,
    evaluate_force_closure,
    load_contact_set,
)
from grasping_ai.evaluation.metrics import aggregate_grasp_success_rate

GraspEvaluation = dict[str, float | bool]


def _contact_provider_from_path(
    contact_path: Path,
) -> Callable[[np.ndarray], list[dict[str, np.ndarray]]]:
    """Build a contact provider that returns a fixed loaded contact set."""
    file_contacts = load_contact_set(contact_path)

    def provider(pose: np.ndarray) -> list[dict[str, np.ndarray]]:
        del pose
        return file_contacts

    return provider


def _evaluate_single_grasp(
    pose: np.ndarray,
    collision_checker: CollisionChecker,
    contact_set_provider: Callable[[np.ndarray], list[dict[str, np.ndarray]]] | None,
    object_point_cloud: np.ndarray,
    gripper_point_cloud: np.ndarray,
    clearance: float,
    fc_judge: ForceClosureJudge,
    friction_coefficient: float,
) -> GraspEvaluation:
    """Evaluate one grasp pose and return its metric dictionary."""
    collision_free = check_collision(collision_checker, pose)
    if contact_set_provider is not None:
        contacts = contact_set_provider(pose)
    else:
        contacts = generate_analytical_contacts(
            object_point_cloud, gripper_point_cloud, pose, clearance
        )

    force_closure = evaluate_force_closure(fc_judge, contacts)
    wrench_matrix = compute_grasp_wrench_matrix(contacts, friction_coefficient)
    wrench_rank = (
        float(np.linalg.matrix_rank(wrench_matrix))
        if wrench_matrix.size
        else 0.0
    )
    grasp_quality = compute_grasp_quality(contacts, friction_coefficient)
    grasp_success = bool(collision_free and force_closure)

    return {
        "collision_free": collision_free,
        "force_closure": force_closure,
        "grasp_success": grasp_success,
        "grasp_quality": float(grasp_quality),
        "wrench_rank": wrench_rank,
    }


def evaluate_generated_grasps(
    grasp_poses: np.ndarray,
    object_point_cloud: np.ndarray,
    gripper_point_cloud: np.ndarray,
    contact_set_provider: Callable[[np.ndarray], list[dict[str, np.ndarray]]] | None = None,
    friction_coefficient: float = 0.5,
    lift_height_threshold: float = 0.05,
    clearance: float = 0.005,
    wrench_regularization: float = 1.0,
    contact_path: Path | None = None,
    filter_collisions: bool = False,
) -> list[GraspEvaluation]:
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
    if clearance < 0:
        raise ValueError("clearance must be non-negative")
    if wrench_regularization < 0:
        raise ValueError("wrench_regularization must be non-negative")

    collision_checker = build_collision_checker(
        object_point_cloud, gripper_point_cloud, clearance=clearance
    )
    if filter_collisions:
        grasp_poses = filter_collision_free_grasps(collision_checker, grasp_poses)
        if grasp_poses.shape[0] == 0:
            return []

    if contact_set_provider is None and contact_path is not None:
        contact_set_provider = _contact_provider_from_path(contact_path)

    fc_judge = build_force_closure_judge(
        friction_coefficient, wrench_regularization=wrench_regularization
    )

    return [
        _evaluate_single_grasp(
            grasp_poses[i],
            collision_checker,
            contact_set_provider,
            object_point_cloud,
            gripper_point_cloud,
            clearance,
            fc_judge,
            friction_coefficient,
        )
        for i in range(grasp_poses.shape[0])
    ]


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
