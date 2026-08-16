"""Print architecture and parameter counts for trained models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from loguru import logger

from grasping_ai.config import SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.training.checkpoint_io import read_model_checkpoint_metadata

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/print_model_info")
def main(cfg: DictConfig) -> None:
    """Print architecture and parameter metadata for the configured checkpoint."""
    yaml_config = FlattenedYAMLConfig(cfg)
    metadata = read_model_checkpoint_metadata(
        yaml_config.value("checkpoint", "model", "checkpoint", value_type=Path, script_or=True, required=True),
        str(yaml_config.get("device")),
    )
    for key in (
        "checkpoint_path",
        "kind",
        "pipeline",
        "architecture",
        "feature_dim",
        "hidden_dim",
        "num_layers",
        "observation_dim",
        "action_dim",
        "epoch",
        "seed",
    ):
        if key in metadata and metadata[key] is not None:
            logger.info("{}: {}", key, metadata[key])

if __name__ == "__main__":
    main()
