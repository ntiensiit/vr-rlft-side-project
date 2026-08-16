"""Train flow-matching grasp models from the command line."""

from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig

from grasping_ai.config.config import (
    SCRIPTS_CONFIG_PATH,
    config_get,
    config_value,
    hydra_cfg_to_dict,
)
from grasping_ai.pipelines.train_flow import run_flow_training_pipeline
from grasping_ai.utils.logging_utils import init_mlflow, setup_logging


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/train_flow")
def main(cfg: DictConfig) -> None:
    dataset_root = config_value(cfg, "paths", "dataset_root", value_type=Path, required=True)
    checkpoint_path = config_value(cfg, "model", "checkpoint", value_type=Path, required=True)
    experiment_log_dir = config_value(cfg, "flow", "tensorboard", value_type=Path)

    setup_logging(module_name="train_flow")
    cfg_dict = hydra_cfg_to_dict(cfg)
    use_mlflow = init_mlflow(cfg_dict)

    pipeline_kwargs = {
        "dataset_root": dataset_root,
        "checkpoint_path": checkpoint_path,
        "feature_dim": config_value(cfg, "architecture", "feature_dim", value_type=int),
        "hidden_dim": config_value(cfg, "architecture", "hidden_dim", value_type=int),
        "num_layers": config_value(cfg, "architecture", "num_layers", value_type=int),
        "learning_rate": config_value(cfg, "supervised", "learning_rate", value_type=float),
        "num_epochs": config_value(cfg, "supervised", "num_epochs", value_type=int),
        "batch_size": config_value(cfg, "supervised", "batch_size", value_type=int),
        "device": str(config_get(cfg, "device")),
        "seed": config_value(cfg, "seed", value_type=int),
        "experiment_log_dir": experiment_log_dir,
        "pretrained_encoder_path": config_value(cfg, "training", "pretrained_encoder", value_type=Path),
        "resume_checkpoint_path": config_value(cfg, "training", "resume", value_type=Path),
        "augment": config_value(cfg, "training", "augment", value_type=bool, default=False),
        "min_grasp_score": config_value(cfg, "supervised", "min_grasp_score", value_type=float, default=0.0),
        "score_repeat_factor": config_value(cfg, "supervised", "score_repeat_factor", value_type=int, default=0),
        "score_repeat_power": config_value(cfg, "supervised", "score_repeat_power", value_type=float, default=1.0),
    }

    if use_mlflow:
        with mlflow.start_run(run_name="flow_training"):
            run_flow_training_pipeline(**pipeline_kwargs)
    else:
        run_flow_training_pipeline(**pipeline_kwargs)


if __name__ == "__main__":
    main()
