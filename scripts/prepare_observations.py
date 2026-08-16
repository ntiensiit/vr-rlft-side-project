"""Build observation tensors for training pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import open3d as _open3d  # noqa: F401
import hydra

from grasping_ai.config import SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig

try:
    from scripts._prepare_observations import make_observations
except ModuleNotFoundError:
    from _prepare_observations import make_observations

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/prepare_observations")
def main(cfg: DictConfig) -> None:
    """Build observation tensors from the hydra configuration."""
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
