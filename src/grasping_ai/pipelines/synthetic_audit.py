"""Audit synthetic grasp labels for consistency and coverage."""

from __future__ import annotations

from grasping_ai.data.pointcloud_dataset import iterate_grasp_dataset
from grasping_ai.evaluation.collision import build_collision_checker, filter_collision_free_grasps
from grasping_ai.evaluation.scoring import recompute_contact_scores
from grasping_ai.robotics.gripper import default_gripper_point_cloud

import json
from pathlib import Path

import numpy as np


def audit_synthetic_labels(
    dataset_root: Path | str,
    friction_coefficient: float,
    collision_clearance: float,
    output_path: Path | str | None = None,
) -> list[dict[str, object]]:
    """Compute per-object synthetic label quality metrics for a processed dataset."""
    dataset_root = Path(dataset_root)
    resolved_output_path = Path(output_path) if output_path is not None else None

    if not dataset_root.is_dir():
        msg = f"Dataset root directory '{dataset_root}' does not exist."
        raise FileNotFoundError(msg)

    gripper_point_cloud = default_gripper_point_cloud()

    records: list[dict[str, object]] = []
    sample_count = 0
    for sample in iterate_grasp_dataset(dataset_root):
        sample_count += 1
        object_id = str(sample.get("object_id", "unknown"))
        point_cloud = sample["point_cloud"]
        grasp_poses = sample["grasp_poses"]
        scores = sample.get("scores")
        if not isinstance(point_cloud, np.ndarray) or not isinstance(grasp_poses, np.ndarray):
            msg = f"Record for {object_id} has invalid point cloud or grasp poses"
            raise TypeError(msg)

        num_grasps = int(grasp_poses.shape[0])
        recomputed_scores, contact_scored = recompute_contact_scores(
            grasp_poses,
            point_cloud,
            gripper_point_cloud,
            friction_coefficient,
            collision_clearance,
        )

        collision_checker = build_collision_checker(point_cloud, gripper_point_cloud, clearance=collision_clearance)
        collision_free = filter_collision_free_grasps(collision_checker, grasp_poses)

        stored_mean = None
        if isinstance(scores, np.ndarray) and scores.shape[0] == num_grasps:
            stored_mean = float(np.mean(scores))

        records.append(
            {
                "object_id": object_id,
                "num_grasps": num_grasps,
                "contact_scored_rate": float(contact_scored / max(num_grasps, 1)),
                "collision_free_rate": float(collision_free.shape[0] / max(num_grasps, 1)),
                "mean_recomputed_score": float(np.mean(recomputed_scores)) if recomputed_scores else 0.0,
                "min_recomputed_score": float(np.min(recomputed_scores)) if recomputed_scores else 0.0,
                "max_recomputed_score": float(np.max(recomputed_scores)) if recomputed_scores else 0.0,
                "stored_mean_score": stored_mean,
            },
        )

    if sample_count == 0:
        msg = f"No records found under '{dataset_root}'"
        raise ValueError(msg)

    if resolved_output_path is not None:
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    return records
