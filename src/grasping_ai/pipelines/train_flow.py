from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import cast

import torch

from grasping_ai.data.training_pairs import (
    build_supervised_training_pairs,
    validate_grasp_dataset,
)
from grasping_ai.models.flow import FlowGeneratorModel, load_flow_model_checkpoint
from grasping_ai.pipelines.supervised_training import iter_supervised_training_batches
from grasping_ai.training.losses import build_flow_matching_loss
from grasping_ai.training.trainer import (
    BatchSource,
    build_adam_optimizer,
    build_supervised_training_step,
    load_training_checkpoint,
    run_training_loop,
)


def build_flow_training_components(
    feature_dim: int,
    hidden_dim: int,
    num_layers: int,
    learning_rate: float,
    device: str,
) -> dict[str, object]:
    """Construct the flow model and a single joint optimizer for it.

    Args:
        feature_dim: Conditioning feature dimension from the encoder.
        hidden_dim: Width of the flow-field hidden layers.
        num_layers: Number of hidden layers in the flow field.
        learning_rate: Learning rate for the Adam optimizer.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.

    Returns:
        A dictionary containing the combined ``"model"`` (encoder + flow
        field on a single ``FlowGeneratorModel``) and the ``"optimizer"``
        that updates both jointly.
    """
    model = cast(torch.nn.Module, FlowGeneratorModel(feature_dim, hidden_dim, num_layers))
    model.to(device)
    optimizer = build_adam_optimizer(model.parameters(), learning_rate)
    return {"model": model, "optimizer": optimizer}


def build_flow_training_step(
    model: FlowGeneratorModel,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: str,
    seed: int | None = None,
) -> Callable[[torch.Tensor, torch.Tensor], dict[str, float]]:
    """Build a callable training step closure for a flow-matching model."""
    device_obj = torch.device(device)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device_obj).manual_seed(seed)

    def flow_forward(point_clouds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pcs = point_clouds
        x_1 = targets
        batch_size_val = x_1.shape[0]
        t = torch.rand(
            batch_size_val,
            dtype=x_1.dtype,
            device=device_obj,
            generator=generator,
        )
        x_0 = torch.randn(
            x_1.shape,
            dtype=x_1.dtype,
            device=device_obj,
            generator=generator,
        )
        t_view = t.view(batch_size_val, 1)
        x_t = (1.0 - t_view) * x_0 + t_view * x_1
        target_velocity = x_1 - x_0
        cond = model.condition(pcs)
        predicted_velocity = model.flow_field(x_t, cond)
        return loss_fn(predicted_velocity, target_velocity)

    return build_supervised_training_step(model, optimizer, device, flow_forward)


def run_flow_training_pipeline(
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
    """Run the end-to-end flow-matching training pipeline for grasp generation.

    Mirrors ``run_diffusion_training_pipeline`` but uses a continuous-time flow-matching
    objective on the canonical-frame 9D grasp vectors instead of the discrete
    diffusion score-matching loss.

    The encoder and flow field are trained jointly on a single
    ``FlowGeneratorModel`` and saved together by the checkpoint writer, so
    the train/inference model contract is explicit: ``state_dict()``
    contains both ``encoder.*`` and ``flow_field.*`` keys, and
    ``load_flow_model_checkpoint`` reconstructs the same model architecture.

    Args:
        dataset_root: Root directory of the grasp-pose dataset.
        checkpoint_path: Destination path for the trained flow checkpoint.
        feature_dim: Conditioning feature dimension from the encoder.
        hidden_dim: Width of the flow-field hidden layers.
        num_layers: Number of hidden layers in the flow field.
        learning_rate: Learning rate for the Adam optimizer.
        num_epochs: Number of training epochs to perform.
        batch_size: Training batch size.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.
        seed: Optional random seed for reproducible training.
        experiment_log_dir: Optional path to write TensorBoard experiment events.
        pretrained_encoder_path: Optional checkpoint whose encoder weights warm-start
            the flow model before training.
        resume_checkpoint_path: Optional path to a checkpoint to resume training from.
        augment: Whether to apply data augmentation to training samples.
        min_grasp_score: Minimum grasp score threshold for training samples.
        score_repeat_factor: Number of times to repeat samples based on score.
        score_repeat_power: Power scaling factor for score-based repetition.
    """
    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be a pathlib.Path instance")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    validate_grasp_dataset(dataset_root)

    if seed is not None:
        torch.manual_seed(seed)

    components = build_flow_training_components(feature_dim, hidden_dim, num_layers, learning_rate, device)
    model = cast(FlowGeneratorModel, components["model"])
    optimizer = cast(torch.optim.Optimizer, components["optimizer"])

    resume_epoch = 0
    if resume_checkpoint_path is not None:
        resume_epoch = load_training_checkpoint(
            resume_checkpoint_path,
            cast(torch.nn.Module, model),
            optimizer,
            device,
        )

    if pretrained_encoder_path is not None:
        if not isinstance(pretrained_encoder_path, Path):
            raise TypeError("pretrained_encoder_path must be a pathlib.Path instance")
        from grasping_ai.training.checkpoint_io import load_torch_checkpoint

        checkpoint = load_torch_checkpoint(pretrained_encoder_path, device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        encoder_state: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                encoder_state[key[len("encoder.") :]] = value
            else:
                encoder_state[key] = value
        cast(torch.nn.Module, model.encoder).load_state_dict(encoder_state, strict=False)

    pairs = build_supervised_training_pairs(
        dataset_root,
        augment=augment,
        seed=seed,
        min_grasp_score=min_grasp_score,
        score_repeat_factor=score_repeat_factor,
        score_repeat_power=score_repeat_power,
    )

    loss_fn = build_flow_matching_loss()
    training_step = build_flow_training_step(
        model,
        loss_fn,
        optimizer,
        device,
        seed=seed,
    )

    dataloader = cast(
        BatchSource,
        partial(iter_supervised_training_batches, pairs, batch_size, device, seed),
    )

    metadata = {
        "pipeline": "flow",
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
        cast(Callable[[torch.Tensor, torch.Tensor], dict[str, float]], training_step),
        dataloader,
        num_epochs,
        checkpoint_path,
        log_every=10,
        experiment_log_dir=experiment_log_dir,
        metadata=metadata,
        seed=seed,
    )


__all__ = [
    "build_flow_training_components",
    "build_flow_training_step",
    "load_flow_model_checkpoint",
    "run_flow_training_pipeline",
]
