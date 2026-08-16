"""Launch MuJoCo grasp simulation from the command line."""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from grasping_ai.config.config import SCRIPTS_CONFIG_PATH, config_value
from grasping_ai.perception.geometry import identity_transform
from grasping_ai.pipelines.evaluate import write_jsonl_records
from grasping_ai.pipelines.simulate_grasp import run_simulation_sweep
from grasping_ai.robotics.transforms import convert_grasps_to_world_frame
from grasping_ai.utils.logging_utils import setup_logging


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/run_simulation")
def main(cfg: DictConfig) -> None:
    setup_logging(module_name="simulation")

    grasp_pose_format = str(config_value(cfg, "grasp_pose_format", value_type=object, default="world", script_or=True))
    grasp_poses = np.load(
        config_value(cfg, "grasps", "model", "exports", "grasp_poses", value_type=Path, script_or=True, required=True)
    )
    if grasp_pose_format == "object":
        grasp_poses = convert_grasps_to_world_frame(grasp_poses, identity_transform())
    elif grasp_pose_format != "world":
        msg = f"Unsupported grasp pose format '{grasp_pose_format}'; supported values are 'world' and 'object'"
        raise ValueError(msg)

    close_default = config_value(cfg, "robot", "gripper", "close_command", value_type=list[float]) or [0.0]
    object_id_value = config_value(cfg, "object_id", value_type=object, default=None, script_or=True)
    if object_id_value is None:
        object_ids = config_value(cfg, "objects", "ids", value_type=list[str])
        object_id_value = object_ids[0]
    object_id = str(object_id_value)
    outcomes = run_simulation_sweep(
        grasp_poses=grasp_poses,
        object_id=object_id,
        ycb_root=config_value(cfg, "ycb_root", "paths", "ycb_mjcf", value_type=Path, script_or=True, required=True),
        robot_xml_path=config_value(
            cfg, "robot_xml", "robot", "description", value_type=Path, script_or=True, required=True
        ),
        table_xml_path=config_value(cfg, "table_xml", "env", "table_xml", value_type=Path, script_or=True),
        num_simulation_steps=config_value(cfg, "num_steps", value_type=int),
        gripper_close_command=np.asarray(close_default, dtype=np.float64),
    )

    output_path = config_value(
        cfg, "output", "model", "exports", "simulation_report", value_type=Path, script_or=True, required=True
    )
    serialized: list[dict[str, object]] = []
    for grasp_index, outcome in enumerate(outcomes):
        converted: dict[str, object] = {
            "record_type": "grasp_outcome",
            "object_id": object_id,
            "grasp_index": grasp_index,
        }
        for key, value in outcome.items():
            converted[key] = value.tolist() if hasattr(value, "tolist") else value
        serialized.append(converted)

    write_jsonl_records(output_path, serialized)


if __name__ == "__main__":
    main()
