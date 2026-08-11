from pathlib import Path

import numpy as np

from grasping_ai.pipelines.evaluate import (
    aggregate_evaluation_results,
    evaluate_generated_grasps,
    write_evaluation_report,
)


def evaluate_main(
    grasps_path: Path,
    object_id: str,
    object_point_cloud_path: Path,
    gripper_point_cloud_path: Path,
    report_path: Path,
    friction_coefficient: float,
    lift_height_threshold: float,
) -> None:
    """Evaluate a set of generated grasps and write a report to disk.

    Args:
        grasps_path: Path to the generated grasp poses file.
        object_id: Identifier of the object whose grasps are being evaluated.
        object_point_cloud_path: Path to the object point cloud file.
        gripper_point_cloud_path: Path to the gripper point cloud file.
        report_path: Destination path for the evaluation report.
        friction_coefficient: Friction coefficient used in force-closure checks.
        lift_height_threshold: Height threshold used for lift-success checks.
    """
    grasps = np.load(grasps_path)
    object_point_cloud = np.load(object_point_cloud_path)
    gripper_point_cloud = np.load(gripper_point_cloud_path)

    def _contact_provider(_grasp_pose: np.ndarray) -> list[dict[str, np.ndarray]]:
        return []

    per_grasp = evaluate_generated_grasps(
        grasp_poses=grasps,
        object_point_cloud=object_point_cloud,
        gripper_point_cloud=gripper_point_cloud,
        contact_set_provider=_contact_provider,
        friction_coefficient=friction_coefficient,
        lift_height_threshold=lift_height_threshold,
    )
    aggregated = aggregate_evaluation_results({object_id: per_grasp})
    write_evaluation_report(report_path, aggregated)


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
    args = parser.parse_args()
    evaluate_main(
        args.grasps,
        args.object_id,
        args.object_point_cloud,
        args.gripper_point_cloud,
        args.report,
        args.friction_coefficient,
        args.lift_height_threshold,
    )
