"""CLI entry point for grasp evaluation metrics."""

from __future__ import annotations

from grasping_ai.config import SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig

from grasping_ai.pipelines.evaluate import (
    aggregate_evaluation_results,
    evaluate_generated_grasps,
    write_evaluation_report,
)

from grasping_ai.pipelines.generate_grasps import load_generated_grasps

from grasping_ai.utils.logging_utils import (
    init_mlflow,
    setup_logging,
)

from pathlib import Path

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/evaluate")
def main(cfg: DictConfig) -> None:
    yaml_config = FlattenedYAMLConfig(cfg)
    grasps_path = yaml_config.value(
        "grasps", "model", "exports", "grasp_candidates", value_type=Path, script_or=True, required=True
    )
    gripper_point_cloud_path = yaml_config.value(
        "gripper_point_cloud",
        "observations",
        "gripper_point_cloud",
        value_type=Path,
        script_or=True,
        required=True,
    )
    report_path = yaml_config.value(
        "report", "evaluation", "analytical_report", value_type=Path, script_or=True, required=True
    )
    multi_object = yaml_config.value("multi_object", value_type=bool, default=False, script_or=True)
    filter_collisions = yaml_config.value(
        "filter_collisions", "evaluation", "filter_collisions", value_type=bool, default=False, script_or=True
    )
    experiment_log_dir = yaml_config.value("script", "experiment_log_dir", value_type=Path)
    contact_path = yaml_config.value("script", "contact_path", value_type=Path)

    friction_coefficient = yaml_config.value("metrics", "friction_coefficient", value_type=float)
    lift_height_threshold = yaml_config.value("metrics", "lift_height_threshold", value_type=float)
    contact_clearance = yaml_config.value("metrics", "collision_clearance", value_type=float)
    wrench_regularization = yaml_config.value("metrics", "wrench_regularization", value_type=float)

    gripper_point_cloud = np.load(gripper_point_cloud_path)
    per_object: dict[str, list[dict[str, float | bool]]] = {}

    if multi_object:
        observations_dir = yaml_config.value(
            "observations_dir", "paths", "observations", value_type=Path, script_or=True, required=True
        )
        object_ids = yaml_config.value("object_ids", "objects", "ids", value_type=list[str], script_or=True)
        grasp_dict = np.load(grasps_path, allow_pickle=True).item()
        if not isinstance(grasp_dict, dict):
            raise ValueError("multi_object evaluation requires a pickled dict grasp artifact")
        for index, object_key in enumerate(sorted(grasp_dict.keys())):
            if index >= len(object_ids):
                msg = f"grasp artifact key {object_key!r} has no matching YCB object id"
                raise ValueError(msg)
            ycb_object_id = object_ids[index]
            observation_path = observations_dir / f"{ycb_object_id}.npy"
            if not observation_path.is_file():
                msg = f"observation point cloud not found: {observation_path}"
                raise FileNotFoundError(msg)
            grasps = load_generated_grasps(grasps_path, object_key=object_key)
            object_point_cloud = np.load(observation_path)
            per_object[object_key] = evaluate_generated_grasps(
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
    else:
        object_id = str(
            yaml_config.value("object_id", "evaluation", "single_object_key", value_type=object, script_or=True)
        )
        object_point_cloud_path = yaml_config.value("object_point_cloud", value_type=Path, script_or=True)
        if object_point_cloud_path is None:
            raise ValueError("object_point_cloud is required unless multi_object is true")
        grasps = load_generated_grasps(grasps_path, object_key=object_id)
        object_point_cloud = np.load(object_point_cloud_path)
        per_object[object_id] = evaluate_generated_grasps(
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

    setup_logging(module_name="evaluate")
    cfg_dict = yaml_config.source
    use_mlflow = init_mlflow(cfg_dict)

    per_object_aggregated = {
        object_id: aggregate_evaluation_results({object_id: results}) for object_id, results in per_object.items()
    }
    aggregated = aggregate_evaluation_results(per_object)

    if use_mlflow:
        with mlflow.start_run(run_name="evaluation"):
            write_evaluation_report(
                report_path,
                aggregated,
                experiment_log_dir,
                per_object_results=per_object_aggregated,
            )
            for key, val in aggregated.items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(key, val)
    else:
        write_evaluation_report(
            report_path,
            aggregated,
            experiment_log_dir,
            per_object_results=per_object_aggregated,
        )

if __name__ == "__main__":
    main()
