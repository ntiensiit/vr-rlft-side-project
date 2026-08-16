"""Build observation tensors for training pipelines."""

from __future__ import annotations

from grasping_ai.config import SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.pipelines.prepare_observations import make_observations

from pathlib import Path

import hydra
import open3d as _open3d  # noqa: F401
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/prepare_observations")
def main(cfg: DictConfig) -> None:
    yaml_config = FlattenedYAMLConfig(cfg)
    output_cfg = yaml_config.get_path("observations", "output")
    gripper_grid = yaml_config.get_path("observations", "gripper_grid")
    if not isinstance(output_cfg, dict) or not isinstance(gripper_grid, dict):
        msg = "observations.output and observations.gripper_grid must be mappings"
        raise TypeError(msg)
    make_observations(
        yaml_config.value("paths", "ycb_root", value_type=Path, required=True),
        yaml_config.value("paths", "observations", value_type=Path, required=True),
        yaml_config.value("observations", "num_samples", value_type=int),
        yaml_config.value("observations", "seed", value_type=int),
        str(output_cfg["merged_objects"]),
        str(output_cfg["merged_objects_normalized"]),
        str(output_cfg["gripper"]),
        gripper_grid,
    )


if __name__ == "__main__":
    main()
