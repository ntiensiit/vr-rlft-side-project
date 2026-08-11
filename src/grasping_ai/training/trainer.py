from collections.abc import Callable, Iterator
from pathlib import Path

import torch

OptimizerFactory = Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer]
TrainingStep = Callable[[torch.Tensor, torch.Tensor], dict[str, float]]


def build_adam_optimizer(
    parameters: Iterator[torch.nn.Parameter], learning_rate: float
) -> torch.optim.Optimizer:
    """Create an Adam optimizer for the given parameters.

    Args:
        parameters: Iterator over the model parameters to optimize.
        learning_rate: Learning rate for the optimizer.

    Returns:
        A configured ``torch.optim.Optimizer`` instance.
    """
    raise NotImplementedError


def build_training_step(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: str,
) -> TrainingStep:
    """Build a callable training step closure over a model and optimizer.

    Args:
        model: The torch module being trained.
        loss_fn: Callable loss returned by ``training.losses``.
        optimizer: Optimizer returned by ``build_adam_optimizer``.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.

    Returns:
        A callable that consumes ``(inputs, targets)`` and returns a
        dictionary of training metrics for the step.
    """
    raise NotImplementedError


def run_training_loop(
    training_step: TrainingStep,
    dataloader: Iterator[tuple[torch.Tensor, torch.Tensor]],
    num_epochs: int,
    checkpoint_path: Path,
    log_every: int,
) -> None:
    """Run a supervised training loop over a dataloader.

    Args:
        training_step: Callable returned by ``build_training_step``.
        dataloader: Iterator yielding ``(inputs, targets)`` batches.
        num_epochs: Number of full passes over the dataloader.
        checkpoint_path: Path where the final checkpoint should be written.
        log_every: Logging interval measured in training steps.
    """
    raise NotImplementedError


def save_training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    checkpoint_path: Path,
) -> None:
    """Persist a training checkpoint to disk.

    Args:
        model: The torch module whose parameters should be saved.
        optimizer: Optimizer whose state should be saved alongside the model.
        epoch: Current epoch index to record in the checkpoint.
        checkpoint_path: Destination file path for the checkpoint.
    """
    raise NotImplementedError


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
    raise NotImplementedError
