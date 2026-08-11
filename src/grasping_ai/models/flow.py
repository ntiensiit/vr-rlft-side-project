from collections.abc import Callable

import torch

FlowField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
FlowIntegrator = Callable[[FlowField, torch.Tensor, int], torch.Tensor]


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
    raise NotImplementedError


def build_flow_integrator(num_steps: int) -> FlowIntegrator:
    """Construct a numerical integrator for sampling along the flow field.

    Args:
        num_steps: Number of integration steps used to evolve samples.

    Returns:
        A callable that integrates ``(flow_field, x0, conditioning)`` forward
        in time and returns terminal samples with the same shape as ``x0``.
    """
    raise NotImplementedError


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
    raise NotImplementedError
