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
    contact_clearance: float = 0.005,
    wrench_regularization: float = 1.0,
    experiment_log_dir: Path | None = None,
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
        contact_clearance: Clearance threshold for analytical contact detection.
        wrench_regularization: Regularization strength for force-closure judge.
        experiment_log_dir: Optional path to write TensorBoard experiment events.
    """
    loaded_data = np.load(grasps_path, allow_pickle=True)
    if isinstance(loaded_data, np.ndarray) and loaded_data.dtype == object:
        loaded = loaded_data.item()
    else:
        loaded = loaded_data

    if isinstance(loaded, dict):
        if object_id in loaded:
            grasps = loaded[object_id]
        elif len(loaded) == 1:
            grasps = next(iter(loaded.values()))
        else:
            raise ValueError(
                f"Object ID '{object_id}' not found in grasp dictionary keys: {list(loaded.keys())}"
            )
    else:
        grasps = loaded

    if isinstance(grasps, np.ndarray):
        if grasps.ndim == 4 and grasps.shape[0] == 1:
            grasps = grasps[0]
        elif grasps.ndim == 4 and grasps.shape[0] > 1:
            raise ValueError(
                "Batched grasp array with batch size > 1 is not supported by evaluate script"
            )

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
    )

