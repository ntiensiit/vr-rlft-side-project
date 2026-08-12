from pathlib import Path
from typing import cast

import torch

from grasping_ai.data.training_pairs import (
    build_supervised_training_pairs,
    validate_grasp_dataset,
)
from grasping_ai.pipelines.supervised_training import ConditionedTrainingDataloader
from grasping_ai.training.checkpoint_io import load_torch_checkpoint


def run_training_pipeline(
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
    """Run the end-to-end supervised training pipeline for grasp generation."""
    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be a pathlib.Path instance")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    validate_grasp_dataset(dataset_root)

    if seed is not None:
        torch.manual_seed(seed)

    components = build_supervised_training_components(
        feature_dim, hidden_dim, num_layers, learning_rate, device
    )
    model = components["model"]
    optimizer = components["optimizer"]

    resume_epoch = 0
    if resume_checkpoint_path is not None:
        from grasping_ai.training.trainer import load_training_checkpoint

        resume_epoch = load_training_checkpoint(
            resume_checkpoint_path,
            cast(torch.nn.Module, model),
            cast(torch.optim.Optimizer, optimizer),
            device,
        )

    if pretrained_encoder_path is not None:
        encoder_state = load_pretrained_encoder(pretrained_encoder_path, device)
        from grasping_ai.models.diffusion import GraspGeneratorModel

        encoder_module = cast(torch.nn.Module, cast(GraspGeneratorModel, model).encoder)
        encoder_module.load_state_dict(encoder_state, strict=False)

    try:
        training_pairs = build_supervised_training_pairs(
            dataset_root, augment=augment, seed=seed
        )
    except Exception as e:
        raise ValueError(f"Failed to build supervised training pairs: {e}") from e

    from grasping_ai.models.diffusion import GraspGeneratorModel
    from grasping_ai.training.losses import build_diffusion_score_loss
    from grasping_ai.training.trainer import build_training_step, run_training_loop

    model_generator = cast(GraspGeneratorModel, model)
    loss_fn = build_diffusion_score_loss()
    training_step = build_training_step(
        model_generator, loss_fn, cast(torch.optim.Optimizer, optimizer), device, seed=seed
    )

    dataloader = ConditionedTrainingDataloader(
        training_pairs,
        batch_size,
        device,
        seed,
        cast(torch.nn.Module, model_generator.encoder),
    )

    metadata = {
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


def build_supervised_training_components(
    feature_dim: int,
    hidden_dim: int,
    num_layers: int,
    learning_rate: float,
    device: str,
) -> dict[str, object]:
    """Construct the torch modules and optimizer used by supervised training."""
    from grasping_ai.models.diffusion import GraspGeneratorModel
    from grasping_ai.training.trainer import build_adam_optimizer

    model = GraspGeneratorModel(feature_dim, hidden_dim, num_layers)
    model.to(device)
    optimizer = build_adam_optimizer(model.parameters(), learning_rate)

    return {"model": model, "optimizer": optimizer}


def load_pretrained_encoder(
    checkpoint_path: Path, device: str,
) -> dict[str, torch.Tensor]:
    """Load a pretrained equivariant encoder from a checkpoint."""
    if not isinstance(checkpoint_path, Path):
        raise TypeError("checkpoint_path must be a pathlib.Path instance")

    checkpoint = load_torch_checkpoint(checkpoint_path, device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    encoder_state = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            encoder_state[k[len("encoder."):]] = v
        else:
            encoder_state[k] = v
    return encoder_state
