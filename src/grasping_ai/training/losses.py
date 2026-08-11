from collections.abc import Callable

import torch

LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def build_diffusion_score_loss(score_model: Callable[..., torch.Tensor]) -> LossFunction:
    """Construct a denoising score-matching loss for a diffusion grasp model.

    Args:
        score_model: Score network produced by ``models.diffusion.build_score_network``.

    Returns:
        A callable loss mapping ``(predicted_score, target_score)`` to a scalar
        training loss tensor.
    """
    raise NotImplementedError


def build_flow_matching_loss(flow_field: Callable[..., torch.Tensor]) -> LossFunction:
    """Construct a flow-matching loss for a kinematic-flow grasp model.

    Args:
        flow_field: Flow field produced by ``models.flow.build_flow_field``.

    Returns:
        A callable loss mapping ``(predicted_velocity, target_velocity)`` to a
        scalar training loss tensor.
    """
    raise NotImplementedError


def build_grasp_pose_regression_loss(loss_type: str) -> LossFunction:
    """Construct a grasp-pose regression loss for supervised training.

    Args:
        loss_type: Loss identifier such as ``"mse"`` or ``"smooth_l1"``.

    Returns:
        A callable loss mapping ``(predicted_pose, target_pose)`` to a scalar
        training loss tensor.
    """
    raise NotImplementedError
