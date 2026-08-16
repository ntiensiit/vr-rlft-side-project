"""Extract grasp poses from prepared object datasets."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra

from grasping_ai.config import SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.pipelines.generate_grasps import (
    load_generated_grasps,
    write_generated_grasps_array,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/extract_object_grasps")
def main(cfg: DictConfig) -> None:
    """Extract one object's grasp poses from the generated grasp artifact."""
    yaml_config = FlattenedYAMLConfig(cfg)
    grasps = load_generated_grasps(
        yaml_config.value(
            "input",
            "model",
            "exports",
            "grasp_candidates",
            value_type=Path,
            script_or=True,
            required=True,
        ),
        object_key=str(
            yaml_config.value("key", "evaluation", "single_object_key", value_type=object, script_or=True),
        ),
    )
    write_generated_grasps_array(
        yaml_config.value(
            "output",
            "model",
            "exports",
            "grasp_poses",
            value_type=Path,
            script_or=True,
            required=True,
        ),
        grasps,
    )

if __name__ == "__main__":
    main()
