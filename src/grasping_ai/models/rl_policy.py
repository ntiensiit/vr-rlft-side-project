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
    raise NotImplementedError


def build_value_network(observation_dim: int, hidden_dim: int, num_layers: int) -> ValueNetwork:
    """Construct a value network for actor-critic style algorithms.

    Args:
        observation_dim: Dimensionality of the observation vector.
        hidden_dim: Width of the hidden layers.
        num_layers: Number of hidden layers in the value network.

    Returns:
        A callable value network mapping observations to scalar values.
    """
    raise NotImplementedError


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
    raise NotImplementedError
