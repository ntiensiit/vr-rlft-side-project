"""Build observation tensors for training pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import open3d as _open3d  # noqa: F401
import hydra

from grasping_ai.config import FLATTENED_YAML_CONFIG, SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.pipelines.prepare_observations import make_observations

YCB_ROOT = Path(str(FLATTENED_YAML_CONFIG.get("script.ycb_root", "data/raw/ycb")))

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
        yaml_config.value("ycb_root", "paths", "ycb_root", value_type=Path, script_or=True, default=YCB_ROOT),
        yaml_config.value("observations_dir", "paths", "observations", value_type=Path, script_or=True, required=True),
        yaml_config.value("num_samples", "observations", "num_samples", value_type=int, script_or=True),
        yaml_config.value("seed", "observations", "seed", value_type=int, script_or=True),
        str(output_cfg["merged_objects"]),
        str(output_cfg["merged_objects_normalized"]),
        str(output_cfg["gripper"]),
        gripper_grid,
        object_ids=yaml_config.value(
            "object_ids",
            "objects",
            "ids",
            value_type=list[str],
            script_or=True,
            default=None,
        ),
    )


if __name__ == "__main__":
    main()
