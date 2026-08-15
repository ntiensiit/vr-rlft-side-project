from __future__ import annotations

from collections.abc import Callable

import torch


def batch_conditioned_grasp_samples(
    conditioning: torch.Tensor,
    grasp_dim: int,
    num_samples: int,
    rng: torch.Generator,
    sample_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Draw grasp samples for each conditioning row using a shared batching layout.

    Args:
        conditioning: Object-level conditioning features with shape ``(B, F)``.
        grasp_dim: Dimensionality of a single grasp pose vector.
        num_samples: Number of grasp poses to sample per conditioning element.
        rng: Torch random generator used to draw initial noise or flow states.
        sample_fn: Callable ``(initial_states, cond_flat) -> samples_flat`` where
            both tensors have leading dimension ``B * num_samples``.

    Returns:
        Sampled grasp poses with shape ``(B, num_samples, grasp_dim)``.
    """
    if conditioning.ndim != 2:
        raise ValueError(f"conditioning must have shape (B, F), got {conditioning.shape}")
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if not isinstance(rng, torch.Generator):
        raise TypeError("rng must be a torch.Generator instance")

    batch_size, feature_size = conditioning.shape
    device = conditioning.device
    dtype = conditioning.dtype

    total_samples = batch_size * num_samples
    cond_flat = conditioning.unsqueeze(1).repeat(1, num_samples, 1).view(total_samples, feature_size)
    initial_states = torch.randn(total_samples, grasp_dim, generator=rng, device=device, dtype=dtype)
    samples_flat = sample_fn(initial_states, cond_flat)
    return samples_flat.view(batch_size, num_samples, grasp_dim)
