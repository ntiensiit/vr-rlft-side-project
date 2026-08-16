"""Private helpers shared by supervised-training CLI scripts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlflow

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
from grasping_ai.utils.logging_utils import init_mlflow, setup_logging

if TYPE_CHECKING:
    from collections.abc import Callable

    from grasping_ai.config.flattened_yaml_config import FlattenedYAMLConfig


DATASET_ROOT = Path(FLATTENED_YAML_CONFIG.get("paths.dataset_root", "data/processed"))
CHECKPOINT_PATH = Path(
    FLATTENED_YAML_CONFIG.get("model.checkpoint", "artifacts/checkpoints/model.pt"),
)
FEATURE_DIM = int(FLATTENED_YAML_CONFIG.get("architecture.feature_dim", 32))
HIDDEN_DIM = int(FLATTENED_YAML_CONFIG.get("architecture.hidden_dim", 32))
NUM_LAYERS = int(FLATTENED_YAML_CONFIG.get("architecture.num_layers", 2))
LEARNING_RATE = float(FLATTENED_YAML_CONFIG.get("supervised.learning_rate", 0.001))
NUM_EPOCHS = int(FLATTENED_YAML_CONFIG.get("supervised.num_epochs", 3))
BATCH_SIZE = int(FLATTENED_YAML_CONFIG.get("supervised.batch_size", 2))
DEVICE = str(FLATTENED_YAML_CONFIG.get("device", "cpu"))
SEED = int(FLATTENED_YAML_CONFIG.get("seed", 42))
_PRETRAINED_ENCODER_VALUE = FLATTENED_YAML_CONFIG.get("training.pretrained_encoder")
PRETRAINED_ENCODER_PATH = Path(_PRETRAINED_ENCODER_VALUE) if _PRETRAINED_ENCODER_VALUE else None
_RESUME_CHECKPOINT_VALUE = FLATTENED_YAML_CONFIG.get("training.resume")
RESUME_CHECKPOINT_PATH = Path(_RESUME_CHECKPOINT_VALUE) if _RESUME_CHECKPOINT_VALUE else None
AUGMENT = bool(FLATTENED_YAML_CONFIG.get("training.augment", False))
MIN_GRASP_SCORE = float(FLATTENED_YAML_CONFIG.get("supervised.min_grasp_score", 0.0))
SCORE_REPEAT_FACTOR = int(FLATTENED_YAML_CONFIG.get("supervised.score_repeat_factor", 0))
SCORE_REPEAT_POWER = float(FLATTENED_YAML_CONFIG.get("supervised.score_repeat_power", 1.0))


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
    pipeline_kwargs = {
        "dataset_root": yaml_config.value(
            "dataset_root", "paths", "dataset_root", value_type=Path, script_or=True, default=DATASET_ROOT,
        ),
        "checkpoint_path": yaml_config.value(
            "checkpoint", "model", "checkpoint", value_type=Path, script_or=True, default=CHECKPOINT_PATH,
        ),
        "feature_dim": yaml_config.value(
            "feature_dim", "architecture", "feature_dim", value_type=int, script_or=True, default=FEATURE_DIM,
        ),
        "hidden_dim": yaml_config.value(
            "hidden_dim", "architecture", "hidden_dim", value_type=int, script_or=True, default=HIDDEN_DIM,
        ),
        "num_layers": yaml_config.value(
            "num_layers", "architecture", "num_layers", value_type=int, script_or=True, default=NUM_LAYERS,
        ),
        "learning_rate": yaml_config.value(
            "learning_rate", "supervised", "learning_rate", value_type=float, script_or=True, default=LEARNING_RATE,
        ),
        "num_epochs": yaml_config.value(
            "num_epochs", "supervised", "num_epochs", value_type=int, script_or=True, default=NUM_EPOCHS,
        ),
        "batch_size": yaml_config.value(
            "batch_size", "supervised", "batch_size", value_type=int, script_or=True, default=BATCH_SIZE,
        ),
        "device": str(yaml_config.value("device", "device", value_type=object, script_or=True, default=DEVICE)),
        "seed": yaml_config.value("seed", "seed", value_type=int, script_or=True, default=SEED),
        "experiment_log_dir": experiment_log_dir,
        "pretrained_encoder_path": yaml_config.value(
            "pretrained_encoder",
            "training",
            "pretrained_encoder",
            value_type=Path,
            script_or=True,
            default=PRETRAINED_ENCODER_PATH,
        ),
        "resume_checkpoint_path": yaml_config.value(
            "resume", "training", "resume", value_type=Path, script_or=True, default=RESUME_CHECKPOINT_PATH,
        ),
        "augment": yaml_config.value(
            "augment", "training", "augment", value_type=bool, script_or=True, default=AUGMENT,
        ),
        "min_grasp_score": yaml_config.value(
            "min_grasp_score",
            "supervised",
            "min_grasp_score",
            value_type=float,
            script_or=True,
            default=MIN_GRASP_SCORE,
        ),
        "score_repeat_factor": yaml_config.value(
            "score_repeat_factor",
            "supervised",
            "score_repeat_factor",
            value_type=int,
            script_or=True,
            default=SCORE_REPEAT_FACTOR,
        ),
        "score_repeat_power": yaml_config.value(
            "score_repeat_power",
            "supervised",
            "score_repeat_power",
            value_type=float,
            script_or=True,
            default=SCORE_REPEAT_POWER,
        ),
    }

    if use_mlflow:
        with mlflow.start_run(run_name=mlflow_run_name):
            pipeline_fn(**pipeline_kwargs)
    else:
        pipeline_fn(**pipeline_kwargs)
