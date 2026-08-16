"""Launch MuJoCo grasp simulation from the command line."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np

from grasping_ai.config import FLATTENED_YAML_CONFIG, SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.perception.geometry import identity_transform
from grasping_ai.pipelines.evaluate import write_jsonl_records
from grasping_ai.pipelines.simulate_grasp import run_simulation_sweep
from grasping_ai.robotics.transforms import convert_grasps_to_world_frame
from grasping_ai.utils.logging_utils import setup_logging

OBJECT_IDS = tuple(FLATTENED_YAML_CONFIG.get("script.object_ids", []))
ROBOT_XML_PATH = Path(str(FLATTENED_YAML_CONFIG.get("script.robot_xml", "deploy/robot.xml")))
YCB_ROOT = Path(str(FLATTENED_YAML_CONFIG.get("script.ycb_root", "data/processed/ycb_mjcf")))
OBJECT_ID = FLATTENED_YAML_CONFIG.get("script.object_id")

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/run_simulation")
def main(cfg: DictConfig) -> None:
    """Run the MuJoCo grasp simulation sweep and write outcome records."""
    setup_logging(module_name="simulation")
    yaml_config = FlattenedYAMLConfig(cfg)

    grasp_pose_format = str(yaml_config.value("grasp_pose_format", value_type=object, default="world", script_or=True))
    grasp_poses = np.load(
        yaml_config.value("grasps", "model", "exports", "grasp_poses", value_type=Path, script_or=True, required=True),
    )
    if grasp_pose_format == "object":
        grasp_poses = convert_grasps_to_world_frame(grasp_poses, identity_transform())
    elif grasp_pose_format != "world":
        msg = f"Unsupported grasp pose format '{grasp_pose_format}'; supported values are 'world' and 'object'"
        raise ValueError(msg)

    close_default = yaml_config.value(
        "close_command", "robot", "gripper", "close_command", value_type=list[float], script_or=True,
    ) or [0.0]
    object_id_value = yaml_config.value(
        "object_id", "script", "object_id", value_type=object, default=OBJECT_ID, script_or=True,
    )
    if object_id_value is None:
        object_ids = yaml_config.value(
            "object_ids", "objects", "ids", value_type=list[str], script_or=True, default=list(OBJECT_IDS),
        )
        object_id_value = object_ids[0]
    object_id = str(object_id_value)
    outcomes = run_simulation_sweep(
        grasp_poses=grasp_poses,
        object_id=object_id,
        ycb_root=yaml_config.value(
            "ycb_root", "paths", "ycb_mjcf", value_type=Path, script_or=True, default=YCB_ROOT,
        ),
        robot_xml_path=yaml_config.value(
            "robot_xml", "robot", "description", value_type=Path, script_or=True, default=ROBOT_XML_PATH,
        ),
        table_xml_path=yaml_config.value("table_xml", "env", "table_xml", value_type=Path, script_or=True),
        num_simulation_steps=yaml_config.value("num_steps", "num_steps", value_type=int, script_or=True),
        gripper_close_command=np.asarray(close_default, dtype=np.float64),
    )

    output_path = yaml_config.value(
        "output", "model", "exports", "simulation_report", value_type=Path, script_or=True, required=True,
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
