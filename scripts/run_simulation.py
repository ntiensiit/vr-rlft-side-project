from pathlib import Path

import numpy as np

from grasping_ai.config.yaml_loader import (
    config_float_list,
    config_get,
    config_path,
    load_project_yaml_config,
    optional_cli_path,
    parse_config_dir_from_argv,
)
from grasping_ai.pipelines.evaluate import write_jsonl_records
from grasping_ai.pipelines.simulate_grasp import run_simulation_sweep

if __name__ == "__main__":
    import argparse

    config_dir = parse_config_dir_from_argv()
    cfg = load_project_yaml_config(config_dir, "base", "data", "model", "gripper", "env", "object")
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-dir", type=Path, default=config_dir)
    parser = argparse.ArgumentParser(
        description="Run grasps in MuJoCo simulation",
        parents=[pre_parser],
    )
    parser.add_argument("--grasps", type=Path, required=True)
    parser.add_argument("--object-id", type=str, required=True)
    parser.add_argument(
        "--ycb-root",
        type=Path,
        default=config_path(cfg, "paths", "ycb_mjcf"),
    )
    parser.add_argument(
        "--robot-xml",
        type=Path,
        default=config_path(cfg, "robot", "description"),
    )
    parser.add_argument("--table-xml", type=optional_cli_path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=config_path(cfg, "diffusion", "exports", "simulation_report"),
    )
    parser.add_argument(
        "--num-simulation-steps",
        type=int,
        default=int(config_get(cfg, "num_steps")),
    )
    close_default = config_float_list(cfg, "robot", "gripper", "close_command") or [0.0]
    parser.add_argument(
        "--gripper-close-command",
        type=float,
        nargs="+",
        default=close_default,
    )
    parser.add_argument(
        "--grasp-pose-format",
        type=str,
        choices=["world", "object"],
        default="world",
        help="Coordinate frame of the input grasps ('world' or 'object')",
    )
    args = parser.parse_args()
    if args.ycb_root is None:
        parser.error("--ycb-root is required (set in configs/base.yaml paths.ycb_mjcf or pass explicitly)")
    if args.robot_xml is None:
        parser.error(
            "--robot-xml is required (set in configs/gripper/default.yaml robot.description or pass explicitly)"
        )
    if args.output is None:
        parser.error(
            "--output is required (set in configs/model/default.yaml "
            "diffusion.exports.simulation_report or pass explicitly)"
        )

    grasp_poses = np.load(args.grasps)
    if args.grasp_pose_format == "object":
        from grasping_ai.perception.geometry import identity_transform
        from grasping_ai.robotics.transforms import convert_grasps_to_world_frame

        grasp_poses = convert_grasps_to_world_frame(grasp_poses, identity_transform())
    elif args.grasp_pose_format != "world":
        raise ValueError(
            f"Unsupported grasp pose format '{args.grasp_pose_format}'; supported values are 'world' and 'object'"
        )

    outcomes = run_simulation_sweep(
        grasp_poses=grasp_poses,
        object_id=args.object_id,
        ycb_root=args.ycb_root,
        robot_xml_path=args.robot_xml,
        table_xml_path=args.table_xml,
        num_simulation_steps=args.num_simulation_steps,
        gripper_close_command=np.asarray(args.gripper_close_command, dtype=np.float64),
    )

    serialized: list[dict[str, object]] = []
    for grasp_index, outcome in enumerate(outcomes):
        converted: dict[str, object] = {
            "record_type": "grasp_outcome",
            "object_id": args.object_id,
            "grasp_index": grasp_index,
        }
        for key, value in outcome.items():
            converted[key] = value.tolist() if hasattr(value, "tolist") else value
        serialized.append(converted)

    write_jsonl_records(args.output, serialized)
