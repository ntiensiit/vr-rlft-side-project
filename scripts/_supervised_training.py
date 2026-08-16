"""Private helpers shared by supervised-training CLI scripts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlflow

from grasping_ai.utils.logging_utils import init_mlflow, setup_logging

if TYPE_CHECKING:
    from collections.abc import Callable

    from grasping_ai.config.flattened_yaml_config import FlattenedYAMLConfig


def build_supervised_training_kwargs(
    yaml_config: FlattenedYAMLConfig,
    experiment_log_dir: Path | None,
) -> dict[str, object]:
    """Build keyword arguments shared by diffusion and flow training scripts."""
    return {
        "dataset_root": yaml_config.value("paths", "dataset_root", value_type=Path, required=True),
        "checkpoint_path": yaml_config.value("model", "checkpoint", value_type=Path, required=True),
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


def run_supervised_training_script(
    yaml_config: FlattenedYAMLConfig,
    *,
    module_name: str,
    experiment_log_dir: Path | None,
    mlflow_run_name: str,
    pipeline_fn: Callable[..., object],
) -> None:
    """Configure logging/MLflow and run a supervised training pipeline."""
    setup_logging(module_name=module_name)
    use_mlflow = init_mlflow(yaml_config.source)
    pipeline_kwargs = build_supervised_training_kwargs(yaml_config, experiment_log_dir)

    if use_mlflow:
        with mlflow.start_run(run_name=mlflow_run_name):
            pipeline_fn(**pipeline_kwargs)
    else:
        pipeline_fn(**pipeline_kwargs)
