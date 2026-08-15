"""Flow-matching models for conditional grasp generation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch

from grasping_ai.models.grasp_sampling_batch import batch_conditioned_grasp_samples
from grasping_ai.models.mlp import build_mish_mlp
from grasping_ai.training.checkpoint_io import load_torch_checkpoint

if TYPE_CHECKING:
    from pathlib import Path

FlowField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
FlowIntegrator = Callable[[FlowField, torch.Tensor, torch.Tensor], torch.Tensor]


class FlowFieldNet(torch.nn.Module):
    """Neural network predicting flow velocity conditioned on features."""

    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int) -> None:
        """Build the flow-field MLP."""
        super().__init__()
        self.mlp = build_mish_mlp(9 + feature_dim, hidden_dim, 9, num_layers)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting velocity."""
        inputs = torch.cat([x, conditioning], dim=-1)
        return self.mlp(inputs)


class FlowGeneratorModel(torch.nn.Module):
    """Complete generative model holding encoder and flow field.

    Owning both the encoder and the flow field on the same ``nn.Module``
    ensures that:

    * a single optimizer updates both jointly;
    * a single ``state_dict`` covers the encoder parameters used at
      training time and the flow-field parameters;
    * inference can reconstruct the exact model by loading the combined
      checkpoint without separately reinitializing or reloading an encoder.

    This makes the train/inference model contract explicit and avoids the
    train/inference inconsistency that arises when the encoder is built
    and optimized separately from the flow field.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int) -> None:
        """Initialize the flow grasp generator."""
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        from grasping_ai.models.equivariant_encoder import build_equivariant_encoder

        self.encoder = build_equivariant_encoder(feature_dim, num_layers)
        self.flow_field = FlowFieldNet(feature_dim, hidden_dim, num_layers)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Forward pass forwarding to the flow field."""
        return self.flow_field(x, conditioning)

    def condition(self, point_clouds: torch.Tensor) -> torch.Tensor:
        """Encode a batch of point clouds into pooled conditioning features.

        Runs the equivariant encoder and mean-pools per-point features into an
        SE(3)-invariant object descriptor for the flow field network.

        Args:
            point_clouds: Batched point cloud with shape ``(B, N, 3)``.

        Returns:
            Pooled conditioning features with shape ``(B, feature_dim)``.
        """
        from grasping_ai.models.equivariant_encoder import (
            encode_point_cloud,
            pool_object_features,
        )

        features = encode_point_cloud(self.encoder, point_clouds)
        return pool_object_features(features)


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
    return batch_conditioned_grasp_samples(
        conditioning,
        grasp_dim,
        num_samples,
        rng,
        lambda initial_states, cond_flat: integrator(flow_field, initial_states, cond_flat),
    )


def load_flow_model_from_state(
    checkpoint: dict[str, object],
    feature_dim: int,
    hidden_dim: int,
    num_layers: int,
    device: str,
) -> FlowGeneratorModel:
    """Reconstruct a ``FlowGeneratorModel`` from an in-memory checkpoint dict."""
    model = FlowGeneratorModel(feature_dim, hidden_dim, num_layers)
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint['model_state_dict'] must be a dictionary")
    model.load_state_dict(cast("dict[str, Any]", state_dict))
    model.to(torch.device(device))
    model.eval()
    return model


def load_flow_model_checkpoint(
    checkpoint_path: Path,
    feature_dim: int,
    hidden_dim: int,
    num_layers: int,
    device: str,
) -> FlowGeneratorModel:
    """Reconstruct a ``FlowGeneratorModel`` from a joint train/inference checkpoint.

    Args:
        checkpoint_path: Path to the flow checkpoint written by flow training.
        feature_dim: Conditioning feature dimension used at training time.
        hidden_dim: Hidden width used at training time.
        num_layers: Number of hidden layers used at training time.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.

    Returns:
        A ``FlowGeneratorModel`` in evaluation mode on the requested device.
    """
    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    return load_flow_model_from_state(checkpoint, feature_dim, hidden_dim, num_layers, device)
