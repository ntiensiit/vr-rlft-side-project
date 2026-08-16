"""CLI entry point for analytical grasp generation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra

from grasping_ai.config import SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.inference.grasp_inference_runtime import run_batch_grasp_inference

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/generate_grasps")
def main(cfg: DictConfig) -> None:
    """Run batch grasp generation from the hydra configuration."""
    yaml_config = FlattenedYAMLConfig(cfg)
    run_batch_grasp_inference(
        checkpoint_path=yaml_config.value(
            "checkpoint", "model", "checkpoint", value_type=Path, script_or=True, required=True,
        ),
        output_path=yaml_config.value(
            "output", "model", "exports", "grasp_candidates", value_type=Path, script_or=True, required=True,
        ),
        observation_paths=yaml_config.value(
            "observations", "observations", "files", value_type=list[Path], script_or=True,
        ),
        feature_dim=yaml_config.value("architecture", "feature_dim", value_type=int),
        num_grasps=yaml_config.value("architecture", "num_grasps", value_type=int),
        device=str(yaml_config.get("device")),
        seed=yaml_config.value("seed", value_type=int),
    )


if __name__ == "__main__":
    main()
