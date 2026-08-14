from functools import partial
from pathlib import Path
from typing import cast

import torch

from grasping_ai.data.training_pairs import (
    build_supervised_training_pairs,
    validate_grasp_dataset,
)
from grasping_ai.pipelines.supervised_training import iter_conditioned_training_batches
from grasping_ai.training.checkpoint_io import load_torch_checkpoint


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

    Raises:
        TypeError: If ``dataset_root`` is not a ``pathlib.Path`` instance.
        FileNotFoundError: If ``dataset_root`` or required checkpoints are missing.
        ValueError: If the dataset cannot be converted into training pairs.
    """
    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be a pathlib.Path instance")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    validate_grasp_dataset(dataset_root)

    if seed is not None:
        torch.manual_seed(seed)

    from grasping_ai.models.diffusion import GraspGeneratorModel
    from grasping_ai.training.losses import build_diffusion_score_loss
    from grasping_ai.training.trainer import (
        BatchSource,
        build_adam_optimizer,
        build_training_step,
        load_training_checkpoint,
        run_training_loop,
    )

    model = GraspGeneratorModel(feature_dim, hidden_dim, num_layers)
    model.to(device)
    optimizer = build_adam_optimizer(model.parameters(), learning_rate)

    resume_epoch = 0
    if resume_checkpoint_path is not None:
        resume_epoch = load_training_checkpoint(
            resume_checkpoint_path,
            cast(torch.nn.Module, model),
            cast(torch.optim.Optimizer, optimizer),
            device,
        )

    if pretrained_encoder_path is not None:
        if not isinstance(pretrained_encoder_path, Path):
            raise TypeError("pretrained_encoder_path must be a pathlib.Path instance")
        checkpoint = load_torch_checkpoint(pretrained_encoder_path, device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        encoder_state: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                encoder_state[key[len("encoder.") :]] = value
            else:
                encoder_state[key] = value
        cast(torch.nn.Module, model.encoder).load_state_dict(encoder_state, strict=False)

    try:
        training_pairs = build_supervised_training_pairs(
            dataset_root, augment=augment, seed=seed
        )
    except Exception as e:
        raise ValueError(f"Failed to build supervised training pairs: {e}") from e

    model_generator = cast(GraspGeneratorModel, model)
    loss_fn = build_diffusion_score_loss()
    training_step = build_training_step(
        model_generator, loss_fn, cast(torch.optim.Optimizer, optimizer), device, seed=seed
    )

    dataloader = cast(
        BatchSource,
        partial(
            iter_conditioned_training_batches,
            training_pairs,
            batch_size,
            device,
            seed,
            cast(torch.nn.Module, model_generator.encoder),
        ),
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

    run_training_loop(
        training_step,
        dataloader,
        num_epochs,
        checkpoint_path,
        log_every=10,
        experiment_log_dir=experiment_log_dir,
        metadata=metadata,
        seed=seed,
    )
