"""Visualize robot in MuJoCo."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra

from grasping_ai.config import FLATTENED_YAML_CONFIG, SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.pipelines.visualize_robot import load_visualization_scene, run_robot_viewer

ROBOT_XML_PATH = Path(str(FLATTENED_YAML_CONFIG.get("script.robot_xml", "deploy/robot.xml")))
YCB_ROOT = Path(str(FLATTENED_YAML_CONFIG.get("script.ycb_root", "data/raw/ycb")))
OBJECT_ID = FLATTENED_YAML_CONFIG.get("script.object_id")

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/visualize_robot")
def main(cfg: DictConfig) -> None:
    """Open the MuJoCo viewer for the configured robot scene."""
    yaml_config = FlattenedYAMLConfig(cfg)
    table_xml = yaml_config.value("table_xml", "env", "table_xml", value_type=Path, script_or=True)
    mj_model, mj_data = load_visualization_scene(
        yaml_config.value(
            "robot_xml", "robot", "description", value_type=Path, script_or=True, default=ROBOT_XML_PATH,
        ),
        object_id=yaml_config.value(
            "object_id", "script", "object_id", value_type=object, default=OBJECT_ID, script_or=True,
        ),
        ycb_root=yaml_config.value("ycb_root", "paths", "ycb_root", value_type=Path, script_or=True, default=YCB_ROOT),
        table_xml_path=Path(table_xml) if table_xml is not None else None,
    )
    run_robot_viewer(mj_model, mj_data)

if __name__ == "__main__":
    main()
