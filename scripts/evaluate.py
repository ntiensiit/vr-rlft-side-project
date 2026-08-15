from __future__ import annotations

from pathlib import Path

import numpy as np

from grasping_ai.config.yaml_loader import (
    config_get,
    config_path,
    config_str_list,
    load_project_yaml_config,
    parse_config_dir_from_argv,
    parse_config_overrides_from_argv,
)
from grasping_ai.pipelines.evaluate import (
    aggregate_evaluation_results,
    evaluate_generated_grasps,
    write_evaluation_report,
)
from grasping_ai.pipelines.generate_grasps import load_generated_grasps

if __name__ == "__main__":
    import argparse

    config_dir = parse_config_dir_from_argv()
    overrides = parse_config_overrides_from_argv()
    cfg = load_project_yaml_config(config_dir, overrides=overrides)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-dir", type=Path, default=config_dir)
    parser = argparse.ArgumentParser(
        description="Evaluate generated grasp poses",
        parents=[pre_parser],
    )
    parser.add_argument("--grasps", type=Path, required=True)
    parser.add_argument("--object-id", type=str, default=None)
    parser.add_argument("--object-point-cloud", type=Path, default=None)
    parser.add_argument("--gripper-point-cloud", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=config_path(cfg, "evaluation", "analytical_report")
        or config_path(cfg, "diffusion", "exports", "evaluation_report"),
    )
    parser.add_argument(
        "--observations-dir",
        type=Path,
        default=config_path(cfg, "paths", "observations"),
    )
    parser.add_argument(
        "--object-ids",
        type=str,
        nargs="+",
        default=config_str_list(cfg, "objects", "ids"),
    )
    parser.add_argument(
        "--multi-object",
        action="store_true",
        help="Evaluate every object entry in a multi-object grasp artifact",
    )
    parser.add_argument(
        "--friction-coefficient",
        type=float,
        default=float(config_get(cfg, "metrics", "friction_coefficient")),
    )
    parser.add_argument(
        "--lift-height-threshold",
        type=float,
        default=float(config_get(cfg, "metrics", "lift_height_threshold")),
    )
    parser.add_argument(
        "--contact-clearance",
        type=float,
        default=float(config_get(cfg, "metrics", "collision_clearance")),
    )
    parser.add_argument(
        "--wrench-regularization",
        type=float,
        default=float(config_get(cfg, "metrics", "wrench_regularization")),
    )
    parser.add_argument("--experiment-log-dir", type=Path, default=None)
    parser.add_argument("--contact-path", type=Path, default=None)
    parser.add_argument(
        "--filter-collisions",
        action="store_true",
        help="Evaluate only collision-free grasps",
    )
    args = parser.parse_args()
    if args.report is None:
        parser.error(
            "--report is required (set in configs/evaluation/diffusion.yaml "
            "evaluation.analytical_report or pass explicitly)"
        )

    gripper_point_cloud = np.load(args.gripper_point_cloud)
    per_object: dict[str, list[dict[str, float | bool]]] = {}

    if args.multi_object:
        if args.observations_dir is None:
            parser.error("--observations-dir is required when --multi-object is set")
        if args.object_ids is None:
            parser.error("--object-ids is required when --multi-object is set")
        grasp_dict = np.load(args.grasps, allow_pickle=True).item()
        if not isinstance(grasp_dict, dict):
            parser.error("--multi-object requires a pickled dict grasp artifact")
        for index, object_key in enumerate(sorted(grasp_dict.keys())):
            if index >= len(args.object_ids):
                parser.error(f"grasp artifact key {object_key!r} has no matching YCB object id")
            ycb_object_id = args.object_ids[index]
            observation_path = args.observations_dir / f"{ycb_object_id}.npy"
            if not observation_path.is_file():
                parser.error(f"observation point cloud not found: {observation_path}")
            grasps = load_generated_grasps(args.grasps, object_key=object_key)
            object_point_cloud = np.load(observation_path)
            per_object[object_key] = evaluate_generated_grasps(
                grasp_poses=grasps,
                object_point_cloud=object_point_cloud,
                gripper_point_cloud=gripper_point_cloud,
                contact_set_provider=None,
                friction_coefficient=args.friction_coefficient,
                lift_height_threshold=args.lift_height_threshold,
                clearance=args.contact_clearance,
                wrench_regularization=args.wrench_regularization,
                contact_path=args.contact_path,
                filter_collisions=args.filter_collisions,
            )
    else:
        if args.object_id is None:
            parser.error("--object-id is required unless --multi-object is set")
        if args.object_point_cloud is None:
            parser.error("--object-point-cloud is required unless --multi-object is set")
        grasps = load_generated_grasps(args.grasps, object_key=args.object_id)
        object_point_cloud = np.load(args.object_point_cloud)
        per_object[args.object_id] = evaluate_generated_grasps(
            grasp_poses=grasps,
            object_point_cloud=object_point_cloud,
            gripper_point_cloud=gripper_point_cloud,
            contact_set_provider=None,
            friction_coefficient=args.friction_coefficient,
            lift_height_threshold=args.lift_height_threshold,
            clearance=args.contact_clearance,
            wrench_regularization=args.wrench_regularization,
            contact_path=args.contact_path,
            filter_collisions=args.filter_collisions,
        )

    from grasping_ai.utils.logging_utils import init_mlflow, setup_logging
    setup_logging(module_name="evaluate")
    use_mlflow = init_mlflow(cfg)

    if use_mlflow:
        import mlflow
        with mlflow.start_run(run_name="evaluation"):
            per_object_aggregated = {
                object_id: aggregate_evaluation_results({object_id: results})
                for object_id, results in per_object.items()
            }
            aggregated = aggregate_evaluation_results(per_object)
            write_evaluation_report(
                args.report,
                aggregated,
                args.experiment_log_dir,
                per_object_results=per_object_aggregated,
            )
            for key, val in aggregated.items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(key, val)
    else:
        per_object_aggregated = {
            object_id: aggregate_evaluation_results({object_id: results}) for object_id, results in per_object.items()
        }
        aggregated = aggregate_evaluation_results(per_object)
        write_evaluation_report(
            args.report,
            aggregated,
            args.experiment_log_dir,
            per_object_results=per_object_aggregated,
        )
