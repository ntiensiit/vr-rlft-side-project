"""Training loss functions for grasp models."""

from __future__ import annotations

from collections.abc import Callable

import torch

LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def build_diffusion_score_loss() -> LossFunction:
    """Construct a denoising score-matching loss for a diffusion grasp model.

    Returns:
        A callable loss mapping ``(predicted_score, target_score)`` to a scalar
        training loss tensor.
    """

    def loss(predicted_score: torch.Tensor, target_score: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(predicted_score, target_score)

    return loss


def build_flow_matching_loss() -> LossFunction:
    """Construct a flow-matching loss for a kinematic-flow grasp model.

    Returns:
        A callable loss mapping ``(predicted_velocity, target_velocity)`` to a
        scalar training loss tensor.
    """

    def loss(predicted_velocity: torch.Tensor, target_velocity: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(predicted_velocity, target_velocity)

    return loss
