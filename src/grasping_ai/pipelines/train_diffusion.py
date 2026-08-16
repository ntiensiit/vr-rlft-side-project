"""Diffusion training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
from grasping_ai.data.training_pairs import (
    SupervisedGraspDataset,
    validate_grasp_dataset,
)
from grasping_ai.models.diffusion import GraspGeneratorModel
from grasping_ai.pipelines.supervised_training import build_conditioned_dataloader
from grasping_ai.training.checkpoint_io import load_torch_checkpoint
from grasping_ai.training.losses import build_diffusion_score_loss
from grasping_ai.training.trainer import (
    build_adam_optimizer,
    build_training_step,
    load_training_checkpoint,
    run_training_loop,
)
from grasping_ai.utils.path_validation import require_path

DATASET_ROOT = Path(FLATTENED_YAML_CONFIG.get("paths.dataset_root", "data/processed"))
CHECKPOINT_PATH = Path(
    FLATTENED_YAML_CONFIG.get("model.checkpoint", "artifacts/checkpoints/diffusion_grasp_generator.pt"),
)
FEATURE_DIM = int(FLATTENED_YAML_CONFIG.get("architecture.feature_dim", 32))
HIDDEN_DIM = int(FLATTENED_YAML_CONFIG.get("architecture.hidden_dim", 32))
NUM_LAYERS = int(FLATTENED_YAML_CONFIG.get("architecture.num_layers", 2))
LEARNING_RATE = float(FLATTENED_YAML_CONFIG.get("supervised.learning_rate", 0.001))
NUM_EPOCHS = int(FLATTENED_YAML_CONFIG.get("supervised.num_epochs", 3))
BATCH_SIZE = int(FLATTENED_YAML_CONFIG.get("supervised.batch_size", 2))
DEVICE = str(FLATTENED_YAML_CONFIG.get("device", "cpu"))
SEED = int(FLATTENED_YAML_CONFIG.get("seed", 42))
AUGMENT = bool(FLATTENED_YAML_CONFIG.get("training.augment", False))
LOG_EVERY = int(FLATTENED_YAML_CONFIG.get("training.log_every", 10))
MIN_GRASP_SCORE = float(FLATTENED_YAML_CONFIG.get("supervised.min_grasp_score", 0.0))
SCORE_REPEAT_FACTOR = int(FLATTENED_YAML_CONFIG.get("supervised.score_repeat_factor", 0))
SCORE_REPEAT_POWER = float(FLATTENED_YAML_CONFIG.get("supervised.score_repeat_power", 1.0))


def run_diffusion_training_pipeline(
    dataset_root: Path = DATASET_ROOT,
    checkpoint_path: Path = CHECKPOINT_PATH,
    *,
    experiment_log_dir: Path | None = None,
    pretrained_encoder_path: Path | None = None,
    resume_checkpoint_path: Path | None = None,
    **options: Any,
) -> None:
    """Run the end-to-end diffusion training pipeline for grasp generation.

    Args:
        dataset_root: Root directory of the grasp-pose dataset.
        checkpoint_path: Destination path for the final model checkpoint.
        feature_dim: Conditioning feature dimension for the generator.
        hidden_dim: Hidden width of the score network.
        num_layers: Number of hidden layers in the score network.
        learning_rate: Adam learning rate.
        num_epochs: Number of full passes over the training set.
        batch_size: Training batch size.
        device: Torch device identifier.
        seed: Optional global training seed.
        experiment_log_dir: Optional TensorBoard log directory.
        pretrained_encoder_path: Optional encoder checkpoint to warm-start from.
        resume_checkpoint_path: Optional checkpoint to resume optimizer state from.
        augment: When ``True``, apply SO(3)/translation jitter to training pairs.
        min_grasp_score: Drop grasps below this score when dataset records include scores.
        score_repeat_factor: Duplicate higher-scoring grasps when positive.
        score_repeat_power: Exponent for score-based pair repetition.
        options: Optional training overrides keyed by configuration name.

    Raises:
        TypeError: If ``dataset_root`` is not a ``pathlib.Path`` instance.
        FileNotFoundError: If ``dataset_root`` or required checkpoints are missing.
        ValueError: If the dataset cannot be converted into training pairs.
    """
    feature_dim = int(options.pop("feature_dim", FEATURE_DIM))
    hidden_dim = int(options.pop("hidden_dim", HIDDEN_DIM))
    num_layers = int(options.pop("num_layers", NUM_LAYERS))
    learning_rate = float(options.pop("learning_rate", LEARNING_RATE))
    num_epochs = int(options.pop("num_epochs", NUM_EPOCHS))
    batch_size = int(options.pop("batch_size", BATCH_SIZE))
    device = str(options.pop("device", DEVICE))
    seed = options.pop("seed", SEED)
    augment = bool(options.pop("augment", AUGMENT))
    min_grasp_score = float(options.pop("min_grasp_score", MIN_GRASP_SCORE))
    score_repeat_factor = int(options.pop("score_repeat_factor", SCORE_REPEAT_FACTOR))
    score_repeat_power = float(options.pop("score_repeat_power", SCORE_REPEAT_POWER))
    if options:
        unexpected = ", ".join(sorted(options))
        msg = f"Unexpected diffusion training options: {unexpected}"
        raise TypeError(msg)

    require_path(dataset_root, "dataset_root")
    if not dataset_root.exists():
        msg = f"Dataset root does not exist: {dataset_root}"
        raise FileNotFoundError(msg)

    validate_grasp_dataset(dataset_root)

    torch.manual_seed(seed) if seed is not None else None

    model = GraspGeneratorModel(feature_dim, hidden_dim, num_layers)
    model.to(device)
    optimizer = build_adam_optimizer(model.parameters(), learning_rate)

    resume_epoch = (
        load_training_checkpoint(resume_checkpoint_path, model, optimizer, device)
        if resume_checkpoint_path is not None
        else 0
    )

    if pretrained_encoder_path is not None:
        require_path(pretrained_encoder_path, "pretrained_encoder_path")
        checkpoint = load_torch_checkpoint(pretrained_encoder_path, device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        encoder_state: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                encoder_state[key[len("encoder.") :]] = value
            else:
                encoder_state[key] = value
        model.encoder.load_state_dict(encoder_state, strict=False)

    try:
        training_dataset = SupervisedGraspDataset(
            dataset_root,
            augment=augment,
            seed=seed,
            min_grasp_score=min_grasp_score,
            score_repeat_factor=score_repeat_factor,
            score_repeat_power=score_repeat_power,
        )
    except Exception as e:
        msg = f"Failed to build supervised training pairs: {e}"
        raise ValueError(msg) from e

    loss_fn = build_diffusion_score_loss()
    training_step = build_training_step(
        model,
        loss_fn,
        optimizer,
        device,
        seed=seed,
    )

    dataloader = build_conditioned_dataloader(
        training_dataset,
        batch_size,
        device,
        seed,
        model.encoder,
    )

    metadata = {
        "pipeline": "diffusion",
        "dataset_root": str(dataset_root),
        "checkpoint_path": str(checkpoint_path),
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "device": device,
        "augment": augment,
        **({"seed": seed} if seed is not None else {}),
        **({"resume_from_epoch": resume_epoch} if resume_epoch > 0 else {}),
    }

    run_training_loop(
        training_step,
        dataloader,
        num_epochs,
        checkpoint_path,
        log_every=LOG_EVERY,
        experiment_log_dir=experiment_log_dir,
        metadata=metadata,
        seed=seed,
    )
