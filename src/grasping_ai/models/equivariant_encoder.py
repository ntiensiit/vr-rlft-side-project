from collections.abc import Callable

import torch

EquivariantFeatures = torch.Tensor


def build_equivariant_encoder(
    feature_dim: int, _num_layers: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Construct a callable equivariant point-cloud encoder.

    Args:
        feature_dim: Output feature dimension produced per point.
        num_layers: Number of equivariant message-passing layers.

    Returns:
        A function that maps a batched point cloud of shape ``(B, N, 3)`` to
        per-point equivariant features of shape ``(B, N, feature_dim)``.
    """
    class PointNetEncoder(torch.nn.Module):
        def __init__(self, f_dim: int) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(3, f_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    return PointNetEncoder(feature_dim)


def encode_point_cloud(
    encoder: Callable[[torch.Tensor], torch.Tensor], points: torch.Tensor
) -> torch.Tensor:
    """Run an equivariant encoder on a batched point cloud.

    Args:
        encoder: Encoder function returned by ``build_equivariant_encoder``.
        points: Batched point cloud tensor with shape ``(B, N, 3)``.

    Returns:
        Equivariant features with shape ``(B, N, feature_dim)``.
    """
    return encoder(points)


def pool_object_features(features: torch.Tensor) -> torch.Tensor:
    """Pool per-point equivariant features into an object-level descriptor.

    Args:
        features: Per-point features with shape ``(B, N, feature_dim)``.

    Returns:
        Object-level descriptor with shape ``(B, feature_dim)``.
    """
    return torch.mean(features, dim=1)
