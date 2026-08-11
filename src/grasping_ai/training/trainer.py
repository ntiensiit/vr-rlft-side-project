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
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    return torch.optim.Adam(list(parameters), lr=learning_rate)


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
    def step(inputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        model.train()
        optimizer.zero_grad()

        device_obj = torch.device(device)
        cond = inputs.to(device_obj)
        x_0 = targets.to(device_obj)
        batch_size_val = x_0.shape[0]

        num_steps = 100
        t = torch.randint(0, num_steps, (batch_size_val,), device=device_obj)
        noise = torch.randn_like(x_0)

        beta = torch.linspace(1e-4, 0.02, num_steps, device=device_obj)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        ab_t = alpha_bar[t].view(batch_size_val, 1)
        x_t = torch.sqrt(ab_t) * x_0 + torch.sqrt(1.0 - ab_t) * noise

        pred_noise = model(x_t, t, cond)
        loss = loss_fn(pred_noise, noise)

        loss.backward()
        optimizer.step()

        return {"loss": float(loss.item())}

    # Attach model and optimizer as attributes to the function object for serialization in the loop
    step.model = model  # type: ignore[attr-defined]
    step.optimizer = optimizer  # type: ignore[attr-defined]
    return step


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
    step_count = 0
    # To support both iterator and iterable dataloaders
    for epoch in range(num_epochs):
        # Retrieve or copy the dataloader for this epoch if iterable
        batches = dataloader
        if not hasattr(dataloader, "__next__") and hasattr(dataloader, "__iter__"):
            batches = iter(dataloader)

        for inputs, targets in batches:
            metrics = training_step(inputs, targets)
            step_count += 1
            if step_count % log_every == 0:
                print(f"Epoch {epoch}, Step {step_count}: Loss = {metrics.get('loss', 0.0):.4f}")

    model = getattr(training_step, "model", None)
    optimizer = getattr(training_step, "optimizer", None)
    if model is not None and optimizer is not None:
        save_training_checkpoint(model, optimizer, num_epochs, checkpoint_path)


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
    if not isinstance(checkpoint_path, Path):
        raise TypeError("checkpoint_path must be a pathlib.Path instance")

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
    if not isinstance(checkpoint_path, Path):
        raise TypeError("checkpoint_path must be a pathlib.Path instance")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint: {e}") from e

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return int(checkpoint.get("epoch", 0))
