from collections.abc import Callable

import torch

FlowField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
FlowIntegrator = Callable[[FlowField, torch.Tensor, torch.Tensor], torch.Tensor]


class FlowFieldNet(torch.nn.Module):
    """Neural network predicting flow velocity conditioned on features."""

    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        in_dim = 9 + feature_dim
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.Mish())
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, 9))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting velocity."""
        inputs = torch.cat([x, conditioning], dim=-1)
        return self.mlp(inputs)


def build_flow_field(feature_dim: int, hidden_dim: int, num_layers: int) -> FlowField:
    """Construct a continuous flow field for grasp-pose generation.

    Args:
        feature_dim: Conditioning feature dimension from the encoder.
        hidden_dim: Width of the hidden layers.
        num_layers: Number of hidden layers in the flow field.

    Returns:
        A callable mapping ``(x, conditioning)`` to a velocity tensor with the
        same shape as ``x``.
    """
    return FlowFieldNet(feature_dim, hidden_dim, num_layers)


def build_flow_integrator(num_steps: int) -> FlowIntegrator:
    """Construct a numerical integrator for sampling along the flow field.

    Args:
        num_steps: Number of integration steps used to evolve samples.

    Returns:
        A callable that integrates ``(flow_field, x0, conditioning)`` forward
        in time and returns terminal samples with the same shape as ``x0``.
    """
    def integrator(
        flow_field: FlowField,
        x0: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        dt = 1.0 / num_steps
        x = x0
        for _ in range(num_steps):
            v = flow_field(x, conditioning)
            x = x + dt * v
        return x

    return integrator


def sample_grasps_with_flow(
    integrator: FlowIntegrator,
    flow_field: FlowField,
    conditioning: torch.Tensor,
    grasp_dim: int,
    num_samples: int,
    rng: torch.Generator,
) -> torch.Tensor:
    """Sample grasp poses by integrating the learned flow field.

    Args:
        integrator: Integrator returned by ``build_flow_integrator``.
        flow_field: Flow field returned by ``build_flow_field``.
        conditioning: Object-level conditioning features with shape ``(B, F)``.
        grasp_dim: Dimensionality of a single grasp pose vector.
        num_samples: Number of grasp poses to sample per conditioning element.
        rng: Torch random generator used to draw initial samples.

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
    x0 = torch.randn(n_total, grasp_dim, generator=rng, device=device, dtype=dtype)
    samples_flat = integrator(flow_field, x0, cond_flat)
    return samples_flat.view(b_size, num_samples, grasp_dim)
