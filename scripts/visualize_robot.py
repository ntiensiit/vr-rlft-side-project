"""Visualize robot teleoperation in MuJoCo."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from grasping_ai.config.config import SCRIPTS_CONFIG_PATH, config_value
from grasping_ai.pipelines.visualize_robot import (
    load_visualization_scene,
    run_robot_viewer,
)


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/visualize_robot")
def main(cfg: DictConfig) -> None:
    table_xml = config_value(cfg, "table_xml", "env", "table_xml", value_type=Path, script_or=True)
    mj_model, mj_data = load_visualization_scene(
        config_value(cfg, "robot", "description", value_type=Path, required=True),
        object_id=config_value(cfg, "object_id", value_type=object, default=None, script_or=True),
        ycb_root=config_value(cfg, "paths", "ycb_root", value_type=Path),
        table_xml_path=Path(table_xml) if table_xml is not None else None,
    )
    run_robot_viewer(mj_model, mj_data)


if __name__ == "__main__":
    main()
