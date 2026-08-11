from collections.abc import Callable

import torch

PolicyNetwork = Callable[[torch.Tensor], torch.Tensor]
ValueNetwork = Callable[[torch.Tensor], torch.Tensor]


def build_policy_network(
    observation_dim: int, action_dim: int, hidden_dim: int, num_layers: int
) -> PolicyNetwork:
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
    policy: PolicyNetwork, observation: torch.Tensor, rng: torch.Generator
) -> torch.Tensor:
    """Sample an action from a stochastic policy given an observation.

    Args:
        policy: Policy network returned by ``build_policy_network``.
        observation: Observation tensor with shape ``(B, observation_dim)``.
        rng: Torch random generator used to sample actions.

    Returns:
        A sampled action tensor with shape ``(B, action_dim)``.
    """
    if observation.ndim != 2:
        raise ValueError(
            f"observation must have shape (B, obs_dim), got {observation.shape}"
        )
    if not isinstance(rng, torch.Generator):
        raise TypeError("rng must be a torch.Generator instance")

    action_mean = policy(observation)
    noise = torch.randn(
        action_mean.shape, generator=rng,
        device=action_mean.device, dtype=action_mean.dtype,
    )
    return action_mean + 0.1 * noise
