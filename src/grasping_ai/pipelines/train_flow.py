import random
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import cast

import torch

from grasping_ai.data.training_pairs import build_supervised_training_pairs
from grasping_ai.models.flow import FlowGeneratorModel
from grasping_ai.training.checkpoint_io import load_torch_checkpoint
from grasping_ai.training.losses import build_flow_matching_loss
from grasping_ai.training.trainer import (
    build_adam_optimizer,
    run_training_loop,
    save_training_checkpoint,
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
        field on a single ``FlowGeneratorModel``), the same value as
        ``"flow_field"`` for backward compatibility, and the ``"optimizer"``
        that updates both jointly.
    """
    model = cast(torch.nn.Module, FlowGeneratorModel(feature_dim, hidden_dim, num_layers))
    model.to(device)
    optimizer = build_adam_optimizer(model.parameters(), learning_rate)
    return {"model": model, "flow_field": model, "optimizer": optimizer}


def build_flow_training_step(
    model: FlowGeneratorModel,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: str,
    seed: int | None = None,
) -> Callable[[torch.Tensor, torch.Tensor], dict[str, float]]:
    """Build a callable training step closure for a flow-matching model.

    The closure updates both the encoder and the flow field jointly so the
    checkpoint preserves the exact encoder state used during training.

    Args:
        model: ``FlowGeneratorModel`` instance being trained (encoder + flow
            field are jointly updated).
        loss_fn: Loss function returned by
            ``training.losses.build_flow_matching_loss``.
        optimizer: Optimizer returned by ``build_adam_optimizer``.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.
        seed: Optional random seed for reproducible interpolation sampling.

    Returns:
        A callable that consumes ``(point_clouds, targets)`` and returns a
        dictionary of training metrics for the step.
    """
    device_obj = torch.device(device)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device_obj).manual_seed(seed)

    def step(
        point_clouds: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, float]:
        model.train()
        optimizer.zero_grad()

        pcs = point_clouds.to(device_obj)
        x_1 = targets.to(device_obj)
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
        loss = loss_fn(predicted_velocity, target_velocity)

        loss.backward()
        optimizer.step()

        return {"loss": float(loss.item())}

    step.model = model  # type: ignore[attr-defined]
    step.optimizer = optimizer  # type: ignore[attr-defined]
    return step


class _FlowTrainingDataloader:
    """Iterable dataloader that emits ``(pcs, targets)`` batches per epoch."""

    def __init__(
        self,
        pairs: list[tuple[torch.Tensor, torch.Tensor]],
        batch_size: int,
        device: str,
        seed: int | None,
    ) -> None:
        self.pairs = pairs
        self.batch_size = batch_size
        self.device = device
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        num_samples = len(self.pairs)
        indices = list(range(num_samples))
        local_random = random.Random(self.seed if self.seed is not None else 42)
        local_random.shuffle(indices)

        for i in range(0, num_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            pcs = torch.stack([self.pairs[idx][0] for idx in batch_indices]).to(
                self.device
            )
            targets = torch.stack(
                [self.pairs[idx][1] for idx in batch_indices]
            ).to(self.device)
            yield pcs, targets


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
) -> None:
    """Run the end-to-end flow-matching training pipeline for grasp generation.

    Mirrors ``run_training_pipeline`` but uses a continuous-time flow-matching
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
    """
    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be a pathlib.Path instance")
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {dataset_root}"
        )

    if seed is not None:
        torch.manual_seed(seed)

    components = build_flow_training_components(
        feature_dim, hidden_dim, num_layers, learning_rate, device
    )
    model = cast(FlowGeneratorModel, components["model"])

    if pretrained_encoder_path is not None:
        from grasping_ai.pipelines.train import load_pretrained_encoder

        encoder_state = load_pretrained_encoder(pretrained_encoder_path, device)
        encoder_module = cast(torch.nn.Module, model.encoder)
        encoder_module.load_state_dict(encoder_state, strict=False)

    pairs = build_supervised_training_pairs(dataset_root)

    loss_fn = build_flow_matching_loss()
    training_step = build_flow_training_step(
        model,
        loss_fn,
        cast(torch.optim.Optimizer, components["optimizer"]),
        device,
        seed=seed,
    )

    dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]] = _FlowTrainingDataloader(
        pairs, batch_size, device, seed
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
    }
    if seed is not None:
        metadata["seed"] = seed

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

    save_training_checkpoint(
        cast(torch.nn.Module, model),
        cast(torch.optim.Optimizer, components["optimizer"]),
        num_epochs,
        checkpoint_path,
        seed=seed,
    )


def load_flow_model_checkpoint(
    checkpoint_path: Path,
    feature_dim: int,
    hidden_dim: int,
    num_layers: int,
    device: str,
) -> FlowGeneratorModel:
    """Reconstruct a ``FlowGeneratorModel`` from a joint train/inference checkpoint.

    Loads the combined ``state_dict`` (encoder + flow field) into a freshly
    constructed ``FlowGeneratorModel`` so the encoder state matches the
    encoder actually used during training.

    Args:
        checkpoint_path: Path to the flow checkpoint written by
            ``run_flow_training_pipeline``.
        feature_dim: Conditioning feature dimension used at training time.
        hidden_dim: Hidden width used at training time.
        num_layers: Number of hidden layers used at training time.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.

    Returns:
        A ``FlowGeneratorModel`` in evaluation mode on the requested device.
    """
    if not isinstance(checkpoint_path, Path):
        raise TypeError("checkpoint_path must be a pathlib.Path instance")
    model = cast(
        FlowGeneratorModel,
        FlowGeneratorModel(feature_dim, hidden_dim, num_layers),
    )
    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model
