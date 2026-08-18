"""Diffusion score networks and samplers for grasps."""

from __future__ import annotations

from collections.abc import Callable

import torch

from grasping_ai.config.diffusion import (
    DEFAULT_DIFFUSION_SCHEDULE,
    linear_beta_schedule,
)
from grasping_ai.models.equivariant_encoder import (
    build_equivariant_encoder,
    encode_point_cloud,
    pool_object_features,
)
from grasping_ai.models.grasp_sampling_batch import batch_conditioned_grasp_samples
from grasping_ai.models.mlp import build_mish_mlp

DiffusionScoreModel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
DiffusionSampler = Callable[
    [torch.Tensor, Callable[..., torch.Tensor], torch.Tensor, torch.Generator | None],
    torch.Tensor,
]


class ScoreNetwork(torch.nn.Module):
    """Neural network predicting score (noise) conditioned on features."""

    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int) -> None:
        """Build the score network MLP and time embedding."""
        super().__init__()
        self.time_emb = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.Mish(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

        mlp_in_dim = 9 + hidden_dim + feature_dim
        self.mlp = build_mish_mlp(mlp_in_dim, hidden_dim, 9, num_layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Forward pass of the score network."""
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        t = t.to(dtype=x.dtype)
        t_emb = self.time_emb(t)
        inputs = torch.cat([x, t_emb, conditioning], dim=-1)
        return self.mlp(inputs)


class GraspGeneratorModel(torch.nn.Module):
    """Complete generative model holding encoder and score network."""

    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int) -> None:
        """Initialize the diffusion grasp generator."""
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = build_equivariant_encoder(feature_dim, num_layers)
        self.score_net = ScoreNetwork(feature_dim, hidden_dim, num_layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Forward pass forwarding to the score network."""
        return self.score_net(x, t, conditioning)

    def condition(self, point_clouds: torch.Tensor) -> torch.Tensor:
        """Encode a batch of point clouds into pooled conditioning features.

        Runs the equivariant encoder and mean-pools per-point features into an
        SE(3)-invariant object descriptor for the diffusion score network.

        Args:
            point_clouds: Batched point cloud with shape ``(B, N, 3)``.

        Returns:
            Pooled conditioning features with shape ``(B, feature_dim)``.
        """
        features = encode_point_cloud(self.encoder, point_clouds)
        return pool_object_features(features)


def build_diffusion_sampler() -> DiffusionSampler:
    """Construct a score-based diffusion sampler for grasp poses.

    Uses ``DEFAULT_DIFFUSION_SCHEDULE`` for the noise schedule.

    Returns:
        A callable that maps ``(initial_noise, score_model, conditioning)``
        to a sampled grasp pose tensor.
    """
    schedule = DEFAULT_DIFFUSION_SCHEDULE
    num_steps = schedule.num_steps
    beta = linear_beta_schedule(schedule)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    def sampler(
        initial_noise: torch.Tensor,
        score_model: Callable[..., torch.Tensor],
        conditioning: torch.Tensor,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample grasp poses from initial noise using the score model.

        Args:
            initial_noise: Starting noise tensor.
            score_model: Callable representing the score network.
            conditioning: Conditioning tensor for the score model.
            rng: Optional PyTorch random number generator.

        Returns:
            A tensor containing the sampled grasp poses.
        """
        device = initial_noise.device
        dtype = initial_noise.dtype
        b_val = beta.to(device=device, dtype=dtype)
        a_val = alpha.to(device=device, dtype=dtype)
        ab_val = alpha_bar.to(device=device, dtype=dtype)

        x = initial_noise
        for t_idx in reversed(range(num_steps)):
            t_tensor = torch.full((x.shape[0], 1), t_idx, device=device, dtype=dtype)
            pred_noise = score_model(x, t_tensor, conditioning)

            beta_t = b_val[t_idx]
            alpha_t = a_val[t_idx]
            alpha_bar_t = ab_val[t_idx]

            mean = (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise) / torch.sqrt(alpha_t)

            if t_idx > 0:
                if rng is not None:
                    noise = torch.randn(x.shape, generator=rng, device=device, dtype=dtype)
                else:
                    noise = torch.randn_like(x)
                sigma = torch.sqrt(beta_t)
                x = mean + sigma * noise
            else:
                x = mean

        return x

    return sampler


# Tests call this with six positional arguments.
def sample_grasps_with_diffusion(  # noqa: PLR0913,PLR0917
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
        score_model: Score network used for denoising.
        conditioning: Object-level conditioning features with shape ``(B, F)``.
        grasp_dim: Dimensionality of a single grasp pose vector.
        num_samples: Number of grasp poses to sample per conditioning element.
        rng: Torch random generator used to draw initial noise.

    Returns:
        A tensor of sampled grasp poses with shape ``(B, num_samples, grasp_dim)``.
    """
    return batch_conditioned_grasp_samples(
        conditioning,
        grasp_dim,
        num_samples,
        rng,
        lambda noise, cond_flat: sampler(noise, score_model, cond_flat, rng),
    )
