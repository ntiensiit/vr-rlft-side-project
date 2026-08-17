"""Run grasp inference on saved point clouds."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra

from grasping_ai.config import FLATTENED_YAML_CONFIG, SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.inference.grasp_inference_runtime import run_single_object_grasp_inference

CHECKPOINT_PATH = Path(
    str(FLATTENED_YAML_CONFIG.get("script.checkpoint", "artifacts/checkpoints/model.pt")),
)
FEATURE_DIM = int(FLATTENED_YAML_CONFIG.get("script.feature_dim", 64))
NUM_GRASPS = int(FLATTENED_YAML_CONFIG.get("script.num_grasps", 8))
FLOW_INFERENCE_STEPS = int(FLATTENED_YAML_CONFIG.get("script.num_steps", 10))
DEVICE = str(FLATTENED_YAML_CONFIG.get("script.device", "cpu"))
SEED = int(FLATTENED_YAML_CONFIG.get("script.seed", 42))
OBJECT_ID = FLATTENED_YAML_CONFIG.get("script.object_id")
METHOD = str(FLATTENED_YAML_CONFIG.get("script.method", "diffusion"))

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/run_grasp_inference")
def main(cfg: DictConfig) -> None:
    """Run single-object grasp inference from the hydra configuration."""
    yaml_config = FlattenedYAMLConfig(cfg)
    observation_path = yaml_config.value("observation", value_type=Path, script_or=True)
    run_single_object_grasp_inference(
        checkpoint_path=yaml_config.value(
            "checkpoint",
            "model",
            "checkpoint",
            value_type=Path,
            script_or=True,
            default=CHECKPOINT_PATH,
        ),
        output_path=yaml_config.value(
            "output", "model", "exports", "inference_candidates", value_type=Path, script_or=True, required=True,
        ),
        method=str(
            yaml_config.value(
                "method", value_type=object, default=METHOD, script_or=True,
            ),
        ),
        feature_dim=yaml_config.value(
            "feature_dim", "architecture", "feature_dim", value_type=int, script_or=True, default=FEATURE_DIM,
        ),
        num_steps=yaml_config.value(
            "num_steps",
            "model",
            "inference_steps",
            value_type=int,
            script_or=True,
            default=FLOW_INFERENCE_STEPS,
        ),
        num_grasps=yaml_config.value(
            "num_grasps", "architecture", "num_grasps", value_type=int, script_or=True, default=NUM_GRASPS,
        ),
        device=str(yaml_config.value("device", "device", value_type=object, script_or=True, default=DEVICE)),
        seed=yaml_config.value("seed", "seed", value_type=int, script_or=True, default=SEED),
        observation_path=observation_path,
        ycb_root=yaml_config.value("ycb_root", "paths", "ycb_root", value_type=Path, script_or=True)
        if observation_path is None
        else None,
        object_id=(
            yaml_config.value(
                "object_id", "script", "object_id", value_type=object, default=OBJECT_ID, script_or=True,
            )
            if observation_path is None
            else None
        ),
    )

if __name__ == "__main__":
    main()
