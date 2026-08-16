"""CLI entry point for analytical grasp generation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra

from grasping_ai.config import FLATTENED_YAML_CONFIG, SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.inference.grasp_inference_runtime import run_batch_grasp_inference

CHECKPOINT_PATH = Path(
    str(FLATTENED_YAML_CONFIG.get("script.checkpoint", "artifacts/checkpoints/model.pt")),
)
FEATURE_DIM = int(FLATTENED_YAML_CONFIG.get("script.feature_dim", 64))
NUM_GRASPS = int(FLATTENED_YAML_CONFIG.get("script.num_grasps", 8))
DEVICE = str(FLATTENED_YAML_CONFIG.get("script.device", "cpu"))
SEED = int(FLATTENED_YAML_CONFIG.get("script.seed", 42))

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/generate_grasps")
def main(cfg: DictConfig) -> None:
    """Run batch grasp generation from the hydra configuration."""
    yaml_config = FlattenedYAMLConfig(cfg)
    run_batch_grasp_inference(
        checkpoint_path=yaml_config.value(
            "checkpoint",
            "model",
            "checkpoint",
            value_type=Path,
            script_or=True,
            default=CHECKPOINT_PATH,
        ),
        output_path=yaml_config.value(
            "output", "model", "exports", "grasp_candidates", value_type=Path, script_or=True, required=True,
        ),
        observation_paths=yaml_config.value(
            "observations", "observations", "files", value_type=list[Path], script_or=True,
        ),
        feature_dim=yaml_config.value(
            "feature_dim", "architecture", "feature_dim", value_type=int, script_or=True, default=FEATURE_DIM,
        ),
        num_grasps=yaml_config.value(
            "num_grasps", "architecture", "num_grasps", value_type=int, script_or=True, default=NUM_GRASPS,
        ),
        device=str(yaml_config.value("device", "device", value_type=object, script_or=True, default=DEVICE)),
        seed=yaml_config.value("seed", "seed", value_type=int, script_or=True, default=SEED),
    )


if __name__ == "__main__":
    main()
