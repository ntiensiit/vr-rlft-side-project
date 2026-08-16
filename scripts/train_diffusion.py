"""Train diffusion grasp models from the command line."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra

from grasping_ai.config import SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.pipelines.supervised_training_script import run_supervised_training_script
from grasping_ai.pipelines.train_diffusion import run_diffusion_training_pipeline

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/train_diffusion")
def main(cfg: DictConfig) -> None:
    """Train the diffusion grasp generator from the hydra configuration."""
    yaml_config = FlattenedYAMLConfig(cfg)
    run_supervised_training_script(
        yaml_config,
        module_name="train_diffusion",
        experiment_log_dir=yaml_config.value("diffusion", "tensorboard", value_type=Path),
        mlflow_run_name="diffusion_training",
        pipeline_fn=run_diffusion_training_pipeline,
    )


if __name__ == "__main__":
    main()
