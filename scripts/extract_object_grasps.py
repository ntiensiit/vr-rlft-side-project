"""Extract grasp poses from prepared object datasets."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from grasping_ai.config.config import SCRIPTS_CONFIG_PATH, config_value
from grasping_ai.pipelines.generate_grasps import (
    load_generated_grasps,
    write_generated_grasps_array,
)


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/extract_object_grasps")
def main(cfg: DictConfig) -> None:
    grasps = load_generated_grasps(
        config_value(
            cfg,
            "input",
            "model",
            "exports",
            "grasp_candidates",
            value_type=Path,
            script_or=True,
            required=True,
        ),
        object_key=str(
            config_value(cfg, "key", "evaluation", "single_object_key", value_type=object, script_or=True)
        ),
    )
    write_generated_grasps_array(
        config_value(
            cfg,
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
