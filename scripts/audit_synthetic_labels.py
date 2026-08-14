import argparse
import json
from pathlib import Path

import numpy as np

from grasping_ai.config.yaml_loader import (
    config_get,
    config_path,
    load_project_yaml_config,
    parse_config_dir_from_argv,
)
from grasping_ai.data.pointcloud_dataset import iterate_grasp_dataset
from grasping_ai.evaluation.collision import (
    build_collision_checker,
    filter_collision_free_grasps,
    generate_analytical_contacts,
)
from grasping_ai.evaluation.force_closure import compute_grasp_quality


def audit_synthetic_labels(
    dataset_root: Path,
    friction_coefficient: float,
    collision_clearance: float,
    output_path: Path | None = None,
) -> list[dict[str, object]]:
    """Compute per-object synthetic label quality metrics for a processed dataset.

    Args:
        dataset_root: Root directory containing synthetic ``.npy`` records.
        friction_coefficient: Friction coefficient for force-closure scoring.
        collision_clearance: Clearance for collision and contact checks.
        output_path: Optional JSON path to write the audit report.

    Returns:
        A list of per-object metric records.

    Raises:
        FileNotFoundError: If ``dataset_root`` does not exist.
        ValueError: If no records are found under ``dataset_root``.
    """
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root directory '{dataset_root}' does not exist.")

    x = np.linspace(-0.03, 0.03, 4)
    y = np.linspace(-0.02, 0.02, 3)
    z = np.linspace(-0.04, 0.04, 5)
    gripper_point_cloud = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
    gripper_point_cloud = gripper_point_cloud.astype(np.float32)

    records: list[dict[str, object]] = []
    sample_count = 0
    for sample in iterate_grasp_dataset(dataset_root):
        sample_count += 1
        object_id = str(sample.get("object_id", "unknown"))
        point_cloud = sample["point_cloud"]
        grasp_poses = sample["grasp_poses"]
        scores = sample.get("scores")
        if not isinstance(point_cloud, np.ndarray) or not isinstance(grasp_poses, np.ndarray):
            raise TypeError(f"Record for {object_id} has invalid point cloud or grasp poses")

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
            }
        )

    if sample_count == 0:
        raise ValueError(f"No records found under '{dataset_root}'")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    return records


if __name__ == "__main__":
    config_dir = parse_config_dir_from_argv()
    cfg = load_project_yaml_config(config_dir)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-dir", type=Path, default=config_dir)
    parser = argparse.ArgumentParser(
        description="Audit synthetic grasp label quality metrics",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=config_path(cfg, "paths", "dataset_root"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for the audit report.",
    )
    parser.add_argument(
        "--friction-coefficient",
        type=float,
        default=float(
            config_get(
                cfg,
                "synthetic",
                "friction_coefficient",
                default=config_get(cfg, "metrics", "friction_coefficient", default=0.5),
            )
        ),
    )
    parser.add_argument(
        "--collision-clearance",
        type=float,
        default=float(
            config_get(
                cfg,
                "synthetic",
                "collision_clearance",
                default=config_get(cfg, "metrics", "collision_clearance", default=0.005),
            )
        ),
    )
    args = parser.parse_args()
    if args.dataset_root is None:
        parser.error(
            "--dataset-root is required (set in configs/data/default.yaml paths.dataset_root or pass explicitly)"
        )

    report = audit_synthetic_labels(
        dataset_root=args.dataset_root,
        friction_coefficient=args.friction_coefficient,
        collision_clearance=args.collision_clearance,
        output_path=args.output,
    )
    for entry in report:
        print(json.dumps(entry))
