"""Train flow-matching grasp models from the command line."""

from __future__ import annotations

from grasping_ai.config import SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig

from grasping_ai.pipelines.train_flow import run_flow_training_pipeline

from grasping_ai.utils.logging_utils import (
    init_mlflow,
    setup_logging,
)

from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/train_flow")
def main(cfg: DictConfig) -> None:
    yaml_config = FlattenedYAMLConfig(cfg)
    dataset_root = yaml_config.value("paths", "dataset_root", value_type=Path, required=True)
    checkpoint_path = yaml_config.value("model", "checkpoint", value_type=Path, required=True)
    experiment_log_dir = yaml_config.value("flow", "tensorboard", value_type=Path)

    setup_logging(module_name="train_flow")
    cfg_dict = yaml_config.source
    use_mlflow = init_mlflow(cfg_dict)

    pipeline_kwargs = {
        "dataset_root": dataset_root,
        "checkpoint_path": checkpoint_path,
        "feature_dim": yaml_config.value("architecture", "feature_dim", value_type=int),
        "hidden_dim": yaml_config.value("architecture", "hidden_dim", value_type=int),
        "num_layers": yaml_config.value("architecture", "num_layers", value_type=int),
        "learning_rate": yaml_config.value("supervised", "learning_rate", value_type=float),
        "num_epochs": yaml_config.value("supervised", "num_epochs", value_type=int),
        "batch_size": yaml_config.value("supervised", "batch_size", value_type=int),
        "device": str(yaml_config.get("device")),
        "seed": yaml_config.value("seed", value_type=int),
        "experiment_log_dir": experiment_log_dir,
        "pretrained_encoder_path": yaml_config.value("training", "pretrained_encoder", value_type=Path),
        "resume_checkpoint_path": yaml_config.value("training", "resume", value_type=Path),
        "augment": yaml_config.value("training", "augment", value_type=bool, default=False),
        "min_grasp_score": yaml_config.value("supervised", "min_grasp_score", value_type=float, default=0.0),
        "score_repeat_factor": yaml_config.value("supervised", "score_repeat_factor", value_type=int, default=0),
        "score_repeat_power": yaml_config.value("supervised", "score_repeat_power", value_type=float, default=1.0),
    }

    if use_mlflow:
        with mlflow.start_run(run_name="flow_training"):
            run_flow_training_pipeline(**pipeline_kwargs)
    else:
        run_flow_training_pipeline(**pipeline_kwargs)

if __name__ == "__main__":
    main()
