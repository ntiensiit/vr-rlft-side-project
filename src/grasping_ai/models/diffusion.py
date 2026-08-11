from collections.abc import Callable

import torch

DiffusionScoreModel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
DiffusionSampler = Callable[[torch.Tensor, Callable[..., torch.Tensor], int], torch.Tensor]


def build_score_network(feature_dim: int, hidden_dim: int, num_layers: int) -> DiffusionScoreModel:
    """Construct a score network for grasp-pose diffusion.

    Args:
        feature_dim: Conditioning feature dimension from the encoder.
        hidden_dim: Width of the hidden layers.
        num_layers: Number of hidden layers in the score network.

    Returns:
        A callable score model accepting ``(x, t, conditioning)`` and returning
        a tensor with the same shape as ``x``.
    """
    raise NotImplementedError


def build_diffusion_sampler(num_steps: int) -> DiffusionSampler:
    """Construct a score-based diffusion sampler for grasp poses.

    Args:
        num_steps: Number of denoising steps used during sampling.

    Returns:
        A callable that maps ``(initial_noise, score_model, conditioning)``
        to a sampled grasp pose tensor.
    """
    raise NotImplementedError


def sample_grasps_with_diffusion(
    sampler: DiffusionSampler,
    score_model: DiffusionScoreModel,
    conditioning: torch.Tensor,
    grasp_dim: int,
    num_samples: int,
    rng: torch.Generator,
) -> torch.Tensor:
    """Draw candidate grasp poses using a score-based diffusion model.

    Args:
        sampler: Diffusion sampler returned by ``build_diffusion_sampler``.
        score_model: Score network produced by ``build_score_network``.
        conditioning: Object-level conditioning features with shape ``(B, F)``.
        grasp_dim: Dimensionality of a single grasp pose vector.
        num_samples: Number of grasp poses to sample per conditioning element.
        rng: Torch random generator used to draw initial noise.

    Returns:
        A tensor of sampled grasp poses with shape ``(B, num_samples, grasp_dim)``.
    """
    raise NotImplementedError
