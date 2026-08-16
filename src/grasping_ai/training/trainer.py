"""Generic supervised trainer implementation."""

from __future__ import annotations

from grasping_ai.config.diffusion import (
    DEFAULT_DIFFUSION_SCHEDULE,
    linear_beta_schedule,
)

from grasping_ai.training.checkpoint_io import load_torch_checkpoint

from grasping_ai.training.experiment_logging import (
    try_log_mlflow_artifact,
    try_log_mlflow_metric,
    try_log_mlflow_param,
)

from grasping_ai.utils.path_validation import require_path

from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from pathlib import Path

BatchSource = Iterable[tuple[torch.Tensor, torch.Tensor]] | Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]
OptimizerFactory = Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer]
LossForward = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

class SupervisedTrainingStep:
    """Callable supervised training step that retains model and optimizer references."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        forward_fn: LossForward,
    ) -> None:
        """Initialize a supervised step from model, optimizer, and forward logic."""
        self.model = model
        self.optimizer = optimizer
        self._device = torch.device(device)
        self._forward_fn = forward_fn

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """Run one supervised optimization step and return scalar metrics."""
        self.model.train()
        self.optimizer.zero_grad()
        loss = self._forward_fn(inputs.to(self._device), targets.to(self._device))
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.item())}

def build_supervised_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    forward_fn: LossForward,
) -> SupervisedTrainingStep:
    """Build a generic supervised step closure with pluggable forward/loss logic."""
    return SupervisedTrainingStep(model, optimizer, device, forward_fn)

def build_adam_optimizer(parameters: Iterator[torch.nn.Parameter], learning_rate: float) -> torch.optim.Optimizer:
    """Create an Adam optimizer for the given parameters.

    Args:
        parameters: Iterator over the model parameters to optimize.
        learning_rate: Learning rate for the optimizer.

    Returns:
        A configured ``torch.optim.Optimizer`` instance.
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    return torch.optim.Adam(list(parameters), lr=learning_rate)

def build_training_step(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: str,
    seed: int | None = None,
) -> SupervisedTrainingStep:
    """Build a callable training step closure over a model and optimizer.

    Args:
        model: The torch module being trained.
        loss_fn: Callable loss returned by ``training.losses``.
        optimizer: Optimizer returned by ``build_adam_optimizer``.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.
        seed: Optional random seed for reproducible sampling.

    Returns:
        A callable that consumes ``(inputs, targets)`` and returns a
        dictionary of training metrics for the step.
    """
    device_obj = torch.device(device)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device_obj).manual_seed(seed)

    def diffusion_forward(cond: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
        batch_size_val = x_0.shape[0]
        
        num_steps = DEFAULT_DIFFUSION_SCHEDULE.num_steps
        t = torch.randint(0, num_steps, (batch_size_val,), device=device_obj, generator=generator)
        noise = torch.randn(x_0.shape, dtype=x_0.dtype, device=device_obj, generator=generator)
        beta = linear_beta_schedule().to(device_obj)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        ab_t = alpha_bar[t].view(batch_size_val, 1)
        x_t = torch.sqrt(ab_t) * x_0 + torch.sqrt(1.0 - ab_t) * noise
        pred_noise = model(x_t, t, cond)
        return loss_fn(pred_noise, noise)

    return build_supervised_training_step(model, optimizer, device, diffusion_forward)

def run_training_loop(
    training_step: SupervisedTrainingStep,
    dataloader: BatchSource,
    num_epochs: int,
    checkpoint_path: Path,
    log_every: int,
    experiment_log_dir: Path | None = None,
    metadata: dict[str, object] | None = None,
    seed: int | None = None,
) -> None:
    """Run a supervised training loop over a dataloader.

    Args:
        training_step: Callable returned by ``build_training_step``.
        dataloader: Iterable yielding ``(inputs, targets)`` batches, refreshed for each epoch.
            May be an iterable or a zero-argument callable returning a fresh iterator.
        num_epochs: Number of full passes over the dataloader.
        checkpoint_path: Path where the final checkpoint should be written.
        log_every: Logging interval measured in training steps.
        experiment_log_dir: Optional path to write TensorBoard experiment events.
        metadata: Optional dictionary of experiment hyperparameters/run metadata.
        seed: Optional training seed to record in the checkpoint.
    """
    writer = None
    if experiment_log_dir is not None:
        writer = SummaryWriter(log_dir=str(experiment_log_dir))
        if metadata:
            for k, v in metadata.items():
                writer.add_text(f"metadata/{k}", str(v), global_step=0)
            for k, v in metadata.items():
                try_log_mlflow_param(k, str(v))

    try:
        step_count = 0
        # To support both iterator and iterable dataloaders
        for epoch in range(num_epochs):
            batches: Iterable[tuple[torch.Tensor, torch.Tensor]]
            if callable(dataloader):
                batches = dataloader()
            elif hasattr(dataloader, "__next__"):
                batches = dataloader
            else:
                batches = iter(dataloader)

            for inputs, targets in batches:
                metrics = training_step(inputs, targets)
                step_count += 1
                if log_every > 0 and step_count % log_every == 0:
                    loss_val = metrics.get("loss", 0.0)
                    logger.info("Epoch {}, Step {}: Loss = {:.4f}", epoch, step_count, loss_val)
                    if writer is not None:
                        writer.add_scalar("loss", float(loss_val), global_step=step_count)
                    try_log_mlflow_metric("loss", float(loss_val), step_count)
    finally:
        if writer is not None:
            writer.close()

    save_training_checkpoint(
        training_step.model,
        training_step.optimizer,
        num_epochs,
        checkpoint_path,
        seed,
    )
    try_log_mlflow_artifact(str(checkpoint_path))

def save_training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    checkpoint_path: Path,
    seed: int | None = None,
) -> None:
    """Persist a training checkpoint to disk.

    Args:
        model: The torch module whose parameters should be saved.
        optimizer: Optimizer whose state should be saved alongside the model.
        epoch: Current epoch index to record in the checkpoint.
        checkpoint_path: Destination file path for the checkpoint.
        seed: Optional training seed to record in the checkpoint.
    """
    require_path(checkpoint_path, "checkpoint_path")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    feature_dim = getattr(model, "feature_dim", 128)
    hidden_dim = getattr(model, "hidden_dim", 256)
    num_layers = getattr(model, "num_layers", 4)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }
    if seed is not None:
        checkpoint["seed"] = seed

    try:
        torch.save(checkpoint, checkpoint_path)
    except Exception as e:
        raise ValueError(f"Failed to save checkpoint: {e}") from e

def load_training_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> int:
    """Restore a model and optimizer from a training checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Module whose parameters should be restored.
        optimizer: Optional optimizer whose state should be restored.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.

    Returns:
        The epoch index recorded in the checkpoint.
    """
    require_path(checkpoint_path, "checkpoint_path")

    
    checkpoint = load_torch_checkpoint(checkpoint_path, device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return int(checkpoint.get("epoch", 0))
