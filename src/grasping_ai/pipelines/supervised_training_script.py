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
        "dataset_root": DATASET_ROOT,
        "checkpoint_path": CHECKPOINT_PATH,
        "feature_dim": FEATURE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "device": DEVICE,
        "seed": SEED,
        "experiment_log_dir": experiment_log_dir,
        "pretrained_encoder_path": PRETRAINED_ENCODER_PATH,
        "resume_checkpoint_path": RESUME_CHECKPOINT_PATH,
        "augment": AUGMENT,
        "min_grasp_score": MIN_GRASP_SCORE,
        "score_repeat_factor": SCORE_REPEAT_FACTOR,
        "score_repeat_power": SCORE_REPEAT_POWER,
    }

    if use_mlflow:
        with mlflow.start_run(run_name=mlflow_run_name):
            pipeline_fn(**pipeline_kwargs)
    else:
        pipeline_fn(**pipeline_kwargs)
