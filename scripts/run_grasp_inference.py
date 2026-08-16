"""Run grasp inference on saved point clouds."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra

from grasping_ai.config import SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.inference.grasp_inference_runtime import run_single_object_grasp_inference

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/run_grasp_inference")
def main(cfg: DictConfig) -> None:
    """Run single-object grasp inference from the hydra configuration."""
    yaml_config = FlattenedYAMLConfig(cfg)
    observation_path = yaml_config.value("observation", value_type=Path, script_or=True)
    run_single_object_grasp_inference(
        checkpoint_path=yaml_config.value(
            "checkpoint", "model", "checkpoint", value_type=Path, script_or=True, required=True,
        ),
        output_path=yaml_config.value(
            "output", "model", "exports", "inference_candidates", value_type=Path, script_or=True, required=True,
        ),
        method=str(
            yaml_config.value(
                "method", value_type=object, default=str(yaml_config.get("default_method")), script_or=True,
            ),
        ),
        feature_dim=yaml_config.value("architecture", "feature_dim", value_type=int),
        num_steps=yaml_config.value("model", "inference_steps", value_type=int, default=5),
        num_grasps=yaml_config.value("architecture", "num_grasps", value_type=int),
        device=str(yaml_config.get("device")),
        seed=yaml_config.value("seed", value_type=int),
        observation_path=observation_path,
        ycb_root=yaml_config.value("ycb_root", "paths", "ycb_root", value_type=Path, script_or=True)
        if observation_path is None
        else None,
        object_id=yaml_config.value("object_id", value_type=object, default=None, script_or=True),
    )

if __name__ == "__main__":
    main()
