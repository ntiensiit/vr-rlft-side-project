"""Print architecture and parameter counts for trained models."""

from __future__ import annotations

from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from grasping_ai.config.config import SCRIPTS_CONFIG_PATH, config_get, config_value
from grasping_ai.training.checkpoint_io import read_model_checkpoint_metadata


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/print_model_info")
def main(cfg: DictConfig) -> None:
    metadata = read_model_checkpoint_metadata(
        config_value(cfg, "checkpoint", "model", "checkpoint", value_type=Path, script_or=True, required=True),
        str(config_get(cfg, "device")),
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
