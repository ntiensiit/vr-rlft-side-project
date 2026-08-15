from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from grasping_ai.data.training_pairs import (
    build_supervised_training_pairs,
    validate_grasp_dataset,
)
from grasping_ai.models.diffusion import GraspGeneratorModel
from grasping_ai.pipelines.supervised_training import iter_conditioned_training_batches
from grasping_ai.training import trainer as training_trainer
from grasping_ai.training.checkpoint_io import load_torch_checkpoint
from grasping_ai.training.losses import build_diffusion_score_loss
from grasping_ai.training.trainer import (
    BatchSource,
    build_adam_optimizer,
    build_training_step,
    load_training_checkpoint,
)
from grasping_ai.utils.path_validation import require_path


def run_diffusion_training_pipeline(
    dataset_root: Path,
    checkpoint_path: Path,
    feature_dim: int,
    hidden_dim: int,
    num_layers: int,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    device: str,
    seed: int | None = None,
    experiment_log_dir: Path | None = None,
    pretrained_encoder_path: Path | None = None,
    resume_checkpoint_path: Path | None = None,
    augment: bool = False,
    min_grasp_score: float = 0.0,
    score_repeat_factor: int = 0,
    score_repeat_power: float = 1.0,
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

    Raises:
        TypeError: If ``dataset_root`` is not a ``pathlib.Path`` instance.
        FileNotFoundError: If ``dataset_root`` or required checkpoints are missing.
        ValueError: If the dataset cannot be converted into training pairs.
    """
    require_path(dataset_root, "dataset_root")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    validate_grasp_dataset(dataset_root)

    if seed is not None:
        torch.manual_seed(seed)

    model = GraspGeneratorModel(feature_dim, hidden_dim, num_layers)
    model.to(device)
    optimizer = build_adam_optimizer(model.parameters(), learning_rate)

    resume_epoch = 0
    if resume_checkpoint_path is not None:
        resume_epoch = load_training_checkpoint(
            resume_checkpoint_path,
            model,
            optimizer,
            device,
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
        training_pairs = build_supervised_training_pairs(
            dataset_root,
            augment=augment,
            seed=seed,
            min_grasp_score=min_grasp_score,
            score_repeat_factor=score_repeat_factor,
            score_repeat_power=score_repeat_power,
        )
    except Exception as e:
        raise ValueError(f"Failed to build supervised training pairs: {e}") from e

    loss_fn = build_diffusion_score_loss()
    training_step = build_training_step(
        model, loss_fn, optimizer, device, seed=seed
    )

    dataloader: BatchSource = partial(
        iter_conditioned_training_batches,
        training_pairs,
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
    }
    if seed is not None:
        metadata["seed"] = seed
    if resume_epoch > 0:
        metadata["resume_from_epoch"] = resume_epoch

    training_trainer.run_training_loop(
        training_step,
        dataloader,
        num_epochs,
        checkpoint_path,
        log_every=10,
        experiment_log_dir=experiment_log_dir,
        metadata=metadata,
        seed=seed,
    )
