from __future__ import annotations

import torch


def build_tanh_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_hidden_layers: int,
) -> torch.nn.Sequential:
    """Build a feed-forward MLP with ``Tanh`` activations between hidden layers.

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Width of each hidden layer.
        out_dim: Output feature dimension.
        num_hidden_layers: Number of hidden ``Linear + Tanh`` blocks.

    Returns:
        A ``torch.nn.Sequential`` mapping ``(B, in_dim)`` to ``(B, out_dim)``.
    """
    layers: list[torch.nn.Module] = []
    current_dim = in_dim
    for _ in range(num_hidden_layers):
        layers.append(torch.nn.Linear(current_dim, hidden_dim))
        layers.append(torch.nn.Tanh())
        current_dim = hidden_dim
    layers.append(torch.nn.Linear(current_dim, out_dim))
    return torch.nn.Sequential(*layers)


def build_mish_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int,
) -> torch.nn.Sequential:
    """Build a feed-forward MLP with ``Mish`` activations for generative heads.

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Width of each hidden layer.
        out_dim: Output feature dimension.
        num_layers: Total depth parameter matching score/flow network configs;
            builds ``num_layers - 1`` hidden blocks plus a final linear layer.

    Returns:
        A ``torch.nn.Sequential`` mapping ``(B, in_dim)`` to ``(B, out_dim)``.
    """
    layers: list[torch.nn.Module] = []
    current_dim = in_dim
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(current_dim, hidden_dim))
        layers.append(torch.nn.Mish())
        current_dim = hidden_dim
    layers.append(torch.nn.Linear(current_dim, out_dim))
    return torch.nn.Sequential(*layers)
