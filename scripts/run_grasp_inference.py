"""Run grasp inference on saved point clouds."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from grasping_ai.config.config import SCRIPTS_CONFIG_PATH, config_get, config_value
from grasping_ai.inference.grasp_inference_runtime import run_single_object_grasp_inference


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/run_grasp_inference")
def main(cfg: DictConfig) -> None:
    observation_path = config_value(cfg, "observation", value_type=Path, script_or=True)
    run_single_object_grasp_inference(
        checkpoint_path=config_value(
            cfg, "checkpoint", "model", "checkpoint", value_type=Path, script_or=True, required=True
        ),
        output_path=config_value(
            cfg, "output", "model", "exports", "inference_candidates", value_type=Path, script_or=True, required=True
        ),
        method=str(
            config_value(cfg, "method", value_type=object, default=str(config_get(cfg, "default_method")), script_or=True)
        ),
        feature_dim=config_value(cfg, "architecture", "feature_dim", value_type=int),
        num_steps=config_value(cfg, "model", "inference_steps", value_type=int, default=5),
        num_grasps=config_value(cfg, "architecture", "num_grasps", value_type=int),
        device=str(config_get(cfg, "device")),
        seed=config_value(cfg, "seed", value_type=int),
        observation_path=observation_path,
        ycb_root=config_value(cfg, "ycb_root", "paths", "ycb_root", value_type=Path, script_or=True)
        if observation_path is None
        else None,
        object_id=config_value(cfg, "object_id", value_type=object, default=None, script_or=True),
    )


if __name__ == "__main__":
    main()
