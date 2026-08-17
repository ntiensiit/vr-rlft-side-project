"""Private implementation for the synthetic-label audit script."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grasping_ai.data.pointcloud_dataset import iterate_grasp_dataset, load_grasp_sample
from grasping_ai.evaluation.collision import build_collision_checker, filter_collision_free_grasps
from grasping_ai.evaluation.scoring import recompute_contact_scores
from grasping_ai.robotics.gripper import default_gripper_point_cloud


def audit_synthetic_labels(
    dataset_root: Path | str,
    friction_coefficient: float,
    collision_clearance: float,
    output_path: Path | str | None = None,
) -> list[dict[str, object]]:
    """Compute per-object synthetic label quality metrics for a processed dataset."""
    dataset_root = Path(dataset_root)
    resolved_output_path = Path(output_path) if output_path is not None else None

    if not dataset_root.exists():
        msg = f"Dataset root or record file '{dataset_root}' does not exist."
        raise FileNotFoundError(msg)

    gripper_point_cloud = default_gripper_point_cloud()

    records: list[dict[str, object]] = []
    sample_count = 0
    if dataset_root.is_file():
        if dataset_root.suffix.lower() != ".npz":
            msg = f"Single-record audit requires an .npz file, got '{dataset_root}'."
            raise ValueError(msg)
        samples = iter((load_grasp_sample(dataset_root),))
    elif dataset_root.is_dir():
        samples = iterate_grasp_dataset(dataset_root)
    else:
        msg = f"Dataset root or record file '{dataset_root}' is not accessible."
        raise FileNotFoundError(msg)

    for sample in samples:
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
        existing_records: list[dict[str, object]] = []
        if resolved_output_path.exists():
            try:
                existing = json.loads(resolved_output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Existing audit report is not valid JSON: '{resolved_output_path}'") from exc
            if not isinstance(existing, list) or not all(isinstance(entry, dict) for entry in existing):
                msg = f"Existing audit report must contain a JSON list of records: '{resolved_output_path}'"
                raise ValueError(msg)
            existing_records = existing
        resolved_output_path.write_text(
            json.dumps(existing_records + records, indent=2),
            encoding="utf-8",
        )

    return records
