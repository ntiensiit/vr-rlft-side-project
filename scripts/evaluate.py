from pathlib import Path

import numpy as np

from grasping_ai.pipelines.evaluate import (
    aggregate_evaluation_results,
    evaluate_generated_grasps,
    write_evaluation_report,
)
from grasping_ai.pipelines.generate_grasps import load_generated_grasps


def evaluate_main(
    grasps_path: Path,
    object_id: str,
    object_point_cloud_path: Path,
    gripper_point_cloud_path: Path,
    report_path: Path,
    friction_coefficient: float,
    lift_height_threshold: float,
    contact_clearance: float = 0.005,
    wrench_regularization: float = 1.0,
    experiment_log_dir: Path | None = None,
    contact_path: Path | None = None,
    filter_collisions: bool = False,
) -> None:
    """Evaluate a set of generated grasps and write a report to disk."""
    grasps = load_generated_grasps(grasps_path, object_key=object_id)

    object_point_cloud = np.load(object_point_cloud_path)
    gripper_point_cloud = np.load(gripper_point_cloud_path)

    per_grasp = evaluate_generated_grasps(
        grasp_poses=grasps,
        object_point_cloud=object_point_cloud,
        gripper_point_cloud=gripper_point_cloud,
        contact_set_provider=None,
        friction_coefficient=friction_coefficient,
        lift_height_threshold=lift_height_threshold,
        clearance=contact_clearance,
        wrench_regularization=wrench_regularization,
        contact_path=contact_path,
        filter_collisions=filter_collisions,
    )
    aggregated = aggregate_evaluation_results({object_id: per_grasp})
    write_evaluation_report(report_path, aggregated, experiment_log_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate generated grasp poses")
    parser.add_argument("--grasps", type=Path, required=True)
    parser.add_argument("--object-id", type=str, required=True)
    parser.add_argument("--object-point-cloud", type=Path, required=True)
    parser.add_argument("--gripper-point-cloud", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--friction-coefficient", type=float, required=True)
    parser.add_argument("--lift-height-threshold", type=float, required=True)
    parser.add_argument("--contact-clearance", type=float, default=0.005)
    parser.add_argument("--wrench-regularization", type=float, default=1.0)
    parser.add_argument("--experiment-log-dir", type=Path, default=None)
    parser.add_argument("--contact-path", type=Path, default=None)
    parser.add_argument(
        "--filter-collisions",
        action="store_true",
        help="Evaluate only collision-free grasps",
    )
    args = parser.parse_args()
    evaluate_main(
        args.grasps,
        args.object_id,
        args.object_point_cloud,
        args.gripper_point_cloud,
        args.report,
        args.friction_coefficient,
        args.lift_height_threshold,
        args.contact_clearance,
        args.wrench_regularization,
        args.experiment_log_dir,
        args.contact_path,
        args.filter_collisions,
    )
