"""Audit synthetic grasp labels for consistency and coverage."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from grasping_ai.config.config import SCRIPTS_CONFIG_PATH, config_value
from grasping_ai.data.pointcloud_dataset import iterate_grasp_dataset
from grasping_ai.evaluation.collision import (
    build_collision_checker,
    filter_collision_free_grasps,
    generate_analytical_contacts,
)
from grasping_ai.evaluation.force_closure import compute_grasp_quality


def _default_gripper_point_cloud() -> np.ndarray:
    x = np.linspace(-0.03, 0.03, 4)
    y = np.linspace(-0.02, 0.02, 3)
    z = np.linspace(-0.04, 0.04, 5)
    gripper_point_cloud = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
    return gripper_point_cloud.astype(np.float32)


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

    gripper_point_cloud = _default_gripper_point_cloud()

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
        recomputed_scores: list[float] = []
        contact_scored = 0
        for pose in grasp_poses:
            contacts = generate_analytical_contacts(
                point_cloud,
                gripper_point_cloud,
                pose,
                contact_clearance=collision_clearance,
            )
            if len(contacts) < 2:
                continue
            contact_scored += 1
            recomputed_scores.append(compute_grasp_quality(contacts, friction_coefficient))

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


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/audit_synthetic_labels")
def main(cfg: DictConfig) -> None:
    report = audit_synthetic_labels(
        dataset_root=config_value(cfg, "paths", "dataset_root", value_type=Path, required=True),
        friction_coefficient=config_value(
            cfg,
            "synthetic",
            "friction_coefficient",
            value_type=float,
            default=config_value(cfg, "metrics", "friction_coefficient", value_type=float),
        ),
        collision_clearance=config_value(
            cfg,
            "synthetic",
            "collision_clearance",
            value_type=float,
            default=config_value(cfg, "metrics", "collision_clearance", value_type=float),
        ),
        output_path=config_value(cfg, "output", value_type=Path, script_or=True),
    )
    for entry in report:
        logger.info("{}", json.dumps(entry))


if __name__ == "__main__":
    main()
