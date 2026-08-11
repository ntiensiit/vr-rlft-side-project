from collections.abc import Callable

import torch

DiffusionScoreModel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
DiffusionSampler = Callable[
    [torch.Tensor, Callable[..., torch.Tensor], torch.Tensor, torch.Generator | None],
    torch.Tensor,
]


class ScoreNetwork(torch.nn.Module):
    """Neural network predicting score (noise) conditioned on features."""

    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.time_emb = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.Mish(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

        layers: list[torch.nn.Module] = []
        in_dim = 9 + hidden_dim + feature_dim
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.Mish())
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, 9))
        self.mlp = torch.nn.Sequential(*layers)

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
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        from grasping_ai.models.equivariant_encoder import build_equivariant_encoder
        self.encoder = build_equivariant_encoder(feature_dim, num_layers)
        self.score_net = ScoreNetwork(feature_dim, hidden_dim, num_layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Forward pass forwarding to the score network."""
        return self.score_net(x, t, conditioning)


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
    return ScoreNetwork(feature_dim, hidden_dim, num_layers)


def build_diffusion_sampler(num_steps: int) -> DiffusionSampler:
    """Construct a score-based diffusion sampler for grasp poses.

    Args:
        num_steps: Number of denoising steps used during sampling.

    Returns:
        A callable that maps ``(initial_noise, score_model, conditioning)``
        to a sampled grasp pose tensor.
    """
    beta = torch.linspace(1e-4, 0.02, num_steps)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    def sampler(
        initial_noise: torch.Tensor,
        score_model: Callable[..., torch.Tensor],
        conditioning: torch.Tensor,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
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
    if conditioning.ndim != 2:
        raise ValueError(f"conditioning must have shape (B, F), got {conditioning.shape}")
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if not isinstance(rng, torch.Generator):
        raise TypeError("rng must be a torch.Generator instance")

    b_size, f_size = conditioning.shape
    device = conditioning.device
    dtype = conditioning.dtype

    n_total = b_size * num_samples
    cond_flat = conditioning.unsqueeze(1).repeat(1, num_samples, 1).view(n_total, f_size)
    noise = torch.randn(n_total, grasp_dim, generator=rng, device=device, dtype=dtype)
    samples_flat = sampler(noise, score_model, cond_flat, rng)
    return samples_flat.view(b_size, num_samples, grasp_dim)
