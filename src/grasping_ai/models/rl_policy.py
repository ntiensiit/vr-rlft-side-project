from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

PolicyNetwork = Callable[[torch.Tensor], torch.Tensor]
ValueNetwork = Callable[[torch.Tensor], torch.Tensor]

RL_CHECKPOINT_FORMAT_VERSION = 1
RL_POLICY_ARCHITECTURE = "mlp"


def save_rl_policy_checkpoint(
    policy: torch.nn.Module,
    policy_checkpoint_path: Path,
    epoch: int,
    observation_dim: int,
    action_dim: int,
    hidden_dim: int,
    num_layers: int,
    seed: int | None = None,
) -> None:
    """Persist an RL policy under the standard checkpoint schema.

    The checkpoint carries explicit architecture metadata
    (``observation_dim``, ``action_dim``, ``hidden_dim``, ``num_layers``,
    ``architecture``, ``format_version``) so that inference can load it
    without guessing hyperparameters from parameter names.

    Args:
        policy: Policy module whose parameters should be saved.
        policy_checkpoint_path: Destination file path for the checkpoint.
        epoch: Number of completed update steps recorded in the checkpoint.
        observation_dim: Dimensionality of the policy observation vector.
        action_dim: Dimensionality of the policy action vector.
        hidden_dim: Hidden width of the policy network.
        num_layers: Number of hidden layers in the policy network.
        seed: Optional training seed recorded in the checkpoint.

    Raises:
        ValueError: If any dimension is non-positive.
    """
    if observation_dim <= 0:
        raise ValueError("observation_dim must be positive")
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if not isinstance(policy_checkpoint_path, Path):
        raise TypeError("policy_checkpoint_path must be a pathlib.Path instance")

    checkpoint: dict[str, Any] = {
        "format_version": RL_CHECKPOINT_FORMAT_VERSION,
        "architecture": RL_POLICY_ARCHITECTURE,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "epoch": epoch,
        "model_state_dict": policy.state_dict(),
    }
    if seed is not None:
        checkpoint["seed"] = seed

    policy_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, policy_checkpoint_path)


def read_rl_policy_metadata(
    checkpoint: dict[str, object],
) -> tuple[int, int, int, int] | None:
    """Read standard policy metadata from a checkpoint.

    Args:
        checkpoint: Loaded RL policy checkpoint dictionary.

    Returns:
        A tuple ``(observation_dim, action_dim, hidden_dim, num_layers)`` when
        the checkpoint carries the standard metadata, otherwise ``None``.
    """
    if not isinstance(checkpoint, dict):
        return None
    keys = ("observation_dim", "action_dim", "hidden_dim", "num_layers")
    if not all(key in checkpoint for key in keys):
        return None
    try:
        dims = []
        for key in keys:
            value = checkpoint[key]
            if isinstance(value, torch.Tensor):
                value = value.item()
            dims.append(int(value))  # type: ignore[call-overload]
        return tuple(dims)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def build_policy_network(observation_dim: int, action_dim: int, hidden_dim: int, num_layers: int) -> PolicyNetwork:
    """Construct a policy network mapping observations to action distributions.

    Args:
        observation_dim: Dimensionality of the policy observation vector.
        action_dim: Dimensionality of the action vector.
        hidden_dim: Width of the hidden layers.
        num_layers: Number of hidden layers in the policy network.

    Returns:
        A callable policy mapping a batched observation tensor of shape
        ``(B, observation_dim)`` to action parameters of shape
        ``(B, action_dim)``.
    """
    if observation_dim <= 0:
        raise ValueError("observation_dim must be positive")
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    layers: list[torch.nn.Module] = []
    in_dim = observation_dim
    for _ in range(num_layers):
        layers.append(torch.nn.Linear(in_dim, hidden_dim))
        layers.append(torch.nn.Tanh())
        in_dim = hidden_dim
    layers.append(torch.nn.Linear(in_dim, action_dim))

    return torch.nn.Sequential(*layers)


def build_value_network(observation_dim: int, hidden_dim: int, num_layers: int) -> ValueNetwork:
    """Construct a value network for actor-critic style algorithms.

    Args:
        observation_dim: Dimensionality of the observation vector.
        hidden_dim: Width of the hidden layers.
        num_layers: Number of hidden layers in the value network.

    Returns:
        A callable value network mapping observations to scalar values.
    """
    if observation_dim <= 0:
        raise ValueError("observation_dim must be positive")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    layers: list[torch.nn.Module] = []
    in_dim = observation_dim
    for _ in range(num_layers):
        layers.append(torch.nn.Linear(in_dim, hidden_dim))
        layers.append(torch.nn.Tanh())
        in_dim = hidden_dim
    layers.append(torch.nn.Linear(in_dim, 1))

    return torch.nn.Sequential(*layers)


def select_action(
    policy: PolicyNetwork,
    observation: torch.Tensor,
    rng: torch.Generator,
    noise_scale: float = 0.1,
) -> torch.Tensor:
    """Sample an action from a stochastic policy given an observation.

    Args:
        policy: Policy network returned by ``build_policy_network``.
        observation: Observation tensor with shape ``(B, observation_dim)``.
        rng: Torch random generator used to sample actions.
        noise_scale: Standard deviation of the Gaussian exploration noise
            added to the deterministic policy output.

    Returns:
        A sampled action tensor with shape ``(B, action_dim)``.
    """
    if observation.ndim != 2:
        raise ValueError(f"observation must have shape (B, obs_dim), got {observation.shape}")
    if not isinstance(rng, torch.Generator):
        raise TypeError("rng must be a torch.Generator instance")
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be non-negative")

    action_mean = policy(observation)
    noise = torch.randn(
        action_mean.shape,
        generator=rng,
        device=action_mean.device,
        dtype=action_mean.dtype,
    )
    return action_mean + noise_scale * noise
