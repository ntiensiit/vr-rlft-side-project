"""SE(3)-equivariant point-cloud encoders."""

from __future__ import annotations

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG

import torch

DEGENERATE_COMPONENT_EPS = float(FLATTENED_YAML_CONFIG.get("tolerances.degenerate_component_eps", 1e-9))
DEGENERATE_SPAN_EPS = float(FLATTENED_YAML_CONFIG.get("tolerances.degenerate_span_eps", 1e-12))
GRASP_POSES_NDIM = int(FLATTENED_YAML_CONFIG.get("grasp.poses_ndim", 3))
MIN_ANTIPODAL_CONTACTS = int(FLATTENED_YAML_CONFIG.get("grasp.min_antipodal_contacts", 2))
SPATIAL_DIM = int(FLATTENED_YAML_CONFIG.get("geometry.spatial_dim", 3))
TORCH_DEGENERATE_CLAMP_MIN = float(FLATTENED_YAML_CONFIG.get("grasp.torch_degenerate_clamp_min", 1e-12))

def _compute_se3_frame(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a deterministic SE(3)-equivariant frame and centroid.

    The frame is built from the direction of the point farthest from the
    centroid and the direction of the largest residual orthogonal to it, then
    completed with their cross product. This Gram-Schmidt construction is
    SO(3)-equivariant (a rotated input rotates the frame covariantly) and free
    of the eigenvector-sign ambiguity of PCA frames.

    Args:
        points: Batched point cloud with shape ``(B, N, 3)``.

    Returns:
        A tuple ``(frame, centroid)`` where ``frame`` has shape ``(B, 3, 3)``
        and ``centroid`` has shape ``(B, 3)``.
    """
    b_size = points.shape[0]
    centroid = points.mean(dim=1, keepdim=True)  # (B, 1, 3)
    rel = points - centroid  # (B, N, 3)
    norms = torch.norm(rel, dim=-1)  # (B, N)
    span = norms.sum(dim=1)  # (B,)
    degenerate_cloud = span < DEGENERATE_SPAN_EPS

    # u1: direction of the point farthest from the centroid.
    idx_a = norms.argmax(dim=1)  # (B,)
    idx_a_3d = idx_a.view(b_size, 1, 1).expand(-1, 1, 3)
    a = rel.gather(1, idx_a_3d).squeeze(1)  # (B, 3)
    u1 = a / torch.norm(a, dim=-1, keepdim=True).clamp(min=TORCH_DEGENERATE_CLAMP_MIN)

    # u2: largest residual component orthogonal to u1.
    proj = torch.einsum("bnk,bk->bn", rel, u1).unsqueeze(-1)  # (B, N, 1)
    residual = rel - proj * u1.unsqueeze(1)  # (B, N, 3)
    res_norm = torch.norm(residual, dim=-1)  # (B, N)
    idx_b = res_norm.argmax(dim=1)  # (B,)
    idx_b_3d = idx_b.view(b_size, 1, 1).expand(-1, 1, 3)
    b_vec = residual.gather(1, idx_b_3d).squeeze(1)  # (B, 3)
    b_norm = torch.norm(b_vec, dim=-1)  # (B,)
    u2 = b_vec / b_norm.clamp(min=TORCH_DEGENERATE_CLAMP_MIN).unsqueeze(-1)

    degenerate = b_norm < DEGENERATE_COMPONENT_EPS
    if bool(degenerate.any()):
        # Degenerate (collinear) elements: pick the reference axis least
        # aligned with u1 and orthonormalize it against u1.
        axes = torch.eye(3, device=points.device, dtype=points.dtype)  # (3, 3)
        abs_dot = torch.abs(u1 @ axes)  # (B, 3)
        ref = axes[abs_dot.argmin(dim=1)]  # (B, 3)
        ref = ref - (ref * u1).sum(-1, keepdim=True) * u1
        fallback = ref / torch.norm(ref, dim=-1, keepdim=True).clamp(min=TORCH_DEGENERATE_CLAMP_MIN)
        u2 = torch.where(degenerate.unsqueeze(-1), fallback, u2)

    u3 = torch.cross(u1, u2, dim=-1)
    frame = torch.stack([u1, u2, u3], dim=-1)  # (B, 3, 3)

    # Fully degenerate (single-point) clouds receive the identity frame.
    identity_frame = torch.eye(3, device=points.device, dtype=points.dtype).expand(b_size, 3, 3)
    frame = torch.where(degenerate_cloud.view(b_size, 1, 1), identity_frame, frame)
    return frame, centroid.squeeze(1)

def compute_se3_frame(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a deterministic SE(3)-equivariant frame and centroid.

    The canonical coordinates ``R^T (p - c)`` derived from this frame are
    invariant under rotations and translations of the input point cloud.

    Args:
        points: Batched point cloud with shape ``(B, N, 3)``.

    Returns:
        A tuple ``(frame, centroid)`` where ``frame`` has shape ``(B, 3, 3)``
        and ``centroid`` has shape ``(B, 3)``.

    Raises:
        TypeError: If ``points`` is not a ``torch.Tensor``.
        ValueError: If ``points`` does not have shape ``(B, N, 3)`` or
            contains fewer than two points.
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError("points must be a torch.Tensor")
    if points.ndim != GRASP_POSES_NDIM or points.shape[2] != SPATIAL_DIM:
        raise ValueError(f"points must have shape (B, N, 3), got {points.shape}")
    if points.shape[1] < MIN_ANTIPODAL_CONTACTS:
        raise ValueError("point cloud must contain at least two points")
    return _compute_se3_frame(points)

def world_transform_from_frame(frame: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
    """Build the SE(3) transform mapping canonical coordinates to input coordinates.

    Canonical coordinates ``p_c = R^T (p - c)`` map back to the input frame via
    ``p = R p_c + c``, which is the homogeneous transform ``[[R, c], [0, 1]]``
    where ``R`` is the equivariant frame and ``c`` the centroid.

    Args:
        frame: Equivariant frame with shape ``(B, 3, 3)``.
        centroid: Cloud centroid with shape ``(B, 3)``.

    Returns:
        The world transform with shape ``(B, 4, 4)``.
    """
    b_size = frame.shape[0]
    world = torch.eye(4, device=frame.device, dtype=frame.dtype).expand(b_size, 4, 4).clone()
    world[:, :3, :3] = frame
    world[:, :3, 3] = centroid
    return world

def invert_rigid_transform_batch(transforms: torch.Tensor) -> torch.Tensor:
    """Invert batched rigid ``(4, 4)`` SE(3) transformation matrices on device.

    Args:
        transforms: Homogeneous transforms with shape ``(B, 4, 4)``.

    Returns:
        Batch of inverse transforms with shape ``(B, 4, 4)``.
    """
    rotation = transforms[:, :3, :3]
    translation = transforms[:, :3, 3]
    rotation_inv = rotation.mT
    translation_inv = -(rotation_inv @ translation.unsqueeze(-1)).squeeze(-1)
    result = torch.zeros_like(transforms)
    result[:, :3, :3] = rotation_inv
    result[:, :3, 3] = translation_inv
    result[:, 3, 3] = 1.0
    return result

def compose_with_se3_frame(transforms: torch.Tensor, frame: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
    """Express canonical-frame grasp poses in the input point-cloud frame.

    Grasps sampled in the canonical object frame must be conjugated by the
    frame world transform ``M = [[R, c], [0, 1]]`` so that they are returned
    in the frame of the original point cloud: ``T_input = M T_canonical M^-1``.

    Args:
        transforms: Canonical-frame grasp transforms with shape ``(B, 4, 4)``.
        frame: Equivariant frame with shape ``(B, 3, 3)``.
        centroid: Cloud centroid with shape ``(B, 3)``.

    Returns:
        Input-frame grasp transforms with shape ``(B, 4, 4)``.
    """
    world = world_transform_from_frame(frame, centroid)
    world_inv = invert_rigid_transform_batch(world)
    return torch.matmul(torch.matmul(world, transforms), world_inv)

class SE3EquivariantPointNet(torch.nn.Module):
    """Point-cloud encoder with SE(3)-equivariant features.

    Each point is expressed in canonical coordinates by centering the cloud
    and projecting onto a deterministic SE(3)-equivariant frame
    (``compute_se3_frame``). Because the canonical coordinates are
    SE(3)-invariant, the per-point features are equivariant under the trivial
    (identity) action, and the pooled object descriptor is SE(3)-invariant.
    """

    def __init__(self, f_dim: int, num_layers: int) -> None:
        """Initialize the encoder from its feature dimension and depth.

        Args:
            f_dim: Output feature dimension produced per point.
            num_layers: Number of feature-transform layers.
        """
        super().__init__()
        self.linear = torch.nn.Linear(3, f_dim)
        layers: list[torch.nn.Module] = []
        for _ in range(max(0, num_layers - 1)):
            layers.append(torch.nn.Mish())
            layers.append(torch.nn.Linear(f_dim, f_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Map a batched point cloud to per-point canonical features."""
        frame, centroid = _compute_se3_frame(points)
        rel = points - centroid.unsqueeze(1)
        canonical = torch.matmul(rel, frame)  # (B, N, 3) SE(3)-invariant
        features = self.linear(canonical)
        if len(self.mlp) > 0:
            features = self.mlp(features)
        return features

def build_equivariant_encoder(feature_dim: int, num_layers: int) -> SE3EquivariantPointNet:
    """Construct a callable SE(3)-equivariant point-cloud encoder.

    The encoder projects each centered point onto a deterministic
    SE(3)-equivariant frame and feeds the resulting invariant canonical
    coordinates through a feature transform, yielding per-point features that
    are equivariant under the trivial (identity) action on the features.

    Args:
        feature_dim: Output feature dimension produced per point.
        num_layers: Number of feature-transform layers.

    Returns:
        A function that maps a batched point cloud of shape ``(B, N, 3)`` to
        per-point equivariant features of shape ``(B, N, feature_dim)``.
    """
    return SE3EquivariantPointNet(feature_dim, num_layers)

def encode_point_cloud(encoder: torch.nn.Module, points: torch.Tensor) -> torch.Tensor:
    """Run an SE(3)-equivariant encoder on a batched point cloud.

    Args:
        encoder: Encoder function returned by ``build_equivariant_encoder``.
        points: Batched point cloud tensor with shape ``(B, N, 3)``.

    Returns:
        Equivariant features with shape ``(B, N, feature_dim)``.
    """
    return encoder(points)

def pool_object_features(features: torch.Tensor) -> torch.Tensor:
    """Pool per-point equivariant features into an object-level descriptor.

    Mean pooling of the (invariant) per-point features yields an
    SE(3)-invariant object descriptor suitable for conditioning the diffusion
    and flow heads, which then produce grasps in the canonical object frame.

    Args:
        features: Per-point features with shape ``(B, N, feature_dim)``.

    Returns:
        Object-level descriptor with shape ``(B, feature_dim)``.
    """
    return torch.mean(features, dim=1)
