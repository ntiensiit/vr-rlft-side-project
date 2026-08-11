from collections.abc import Callable

import torch

LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def build_diffusion_score_loss(_score_model: Callable[..., torch.Tensor]) -> LossFunction:
    """Construct a denoising score-matching loss for a diffusion grasp model.

    Args:
        _score_model: Score network produced by ``models.diffusion.build_score_network``.

    Returns:
        A callable loss mapping ``(predicted_score, target_score)`` to a scalar
        training loss tensor.
    """
    def loss(predicted_score: torch.Tensor, target_score: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(predicted_score, target_score)
    return loss


def build_flow_matching_loss(_flow_field: Callable[..., torch.Tensor]) -> LossFunction:
    """Construct a flow-matching loss for a kinematic-flow grasp model.

    Args:
        _flow_field: Flow field produced by ``models.flow.build_flow_field``.

    Returns:
        A callable loss mapping ``(predicted_velocity, target_velocity)`` to a
        scalar training loss tensor.
    """
    def loss(predicted_velocity: torch.Tensor, target_velocity: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
    return loss


def build_grasp_pose_regression_loss(loss_type: str) -> LossFunction:
    """Construct a grasp-pose regression loss for supervised training.

    Args:
        loss_type: Loss identifier such as ``"mse"`` or ``"smooth_l1"``.

    Returns:
        A callable loss mapping ``(predicted_pose, target_pose)`` to a scalar
        training loss tensor.
    """
    if loss_type.lower() == "mse":
        def loss_mse(predicted_pose: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.mse_loss(predicted_pose, target_pose)
        return loss_mse
    if loss_type.lower() == "smooth_l1":
        def loss_l1(predicted_pose: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.smooth_l1_loss(predicted_pose, target_pose)
        return loss_l1
    raise ValueError(f"Unsupported loss_type: '{loss_type}'")
