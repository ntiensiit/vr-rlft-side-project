from collections.abc import Callable

import numpy as np
import torch

from grasping_ai.data.grasp_vector import vec_to_se3
from grasping_ai.models.equivariant_encoder import (
    compose_with_se3_frame,
    compute_se3_frame,
    encode_point_cloud,
    pool_object_features,
)


def prepare_point_cloud_tensor(point_cloud: np.ndarray, device: str) -> torch.Tensor:
    """Validate and batch a numpy point cloud for grasp inference.

    Args:
        point_cloud: Object point cloud with shape ``(N, 3)``.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.

    Returns:
        Point cloud tensor with shape ``(1, N, 3)`` on ``device``.
    """
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        raise ValueError(f"point_cloud must have shape (N, 3), got {point_cloud.shape}")
    return torch.from_numpy(point_cloud).float().to(device).unsqueeze(0)


def encode_grasp_conditioning(
    encoder: Callable[[torch.Tensor], torch.Tensor], point_cloud: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute SE(3) frame features and pooled conditioning for grasp sampling.

    Args:
        encoder: Equivariant point-cloud encoder module.
        point_cloud: Batched point cloud with shape ``(1, N, 3)``.

    Returns:
        Tuple ``(conditioning, frame, centroid)`` used by diffusion/flow samplers.
    """
    frame, centroid = compute_se3_frame(point_cloud)
    features = encode_point_cloud(encoder, point_cloud)
    conditioning = pool_object_features(features)
    return conditioning, frame, centroid


def sample_to_world_frame(
    samples: torch.Tensor,
    frame: torch.Tensor,
    centroid: torch.Tensor,
) -> np.ndarray:
    """Map canonical 9D grasp samples back to the input point-cloud frame.

    Args:
        samples: Sample tensor whose last dimension is 9 (or flattenable to 9).
        frame: Equivariant frame with shape ``(1, 3, 3)``.
        centroid: Cloud centroid with shape ``(1, 3)``.

    Returns:
        World-frame grasp transforms as a numpy array with shape ``(K, 4, 4)``.
    """
    samples_flat = samples.reshape(-1, 9)
    canonical = vec_to_se3(samples_flat)
    transforms = compose_with_se3_frame(canonical, frame, centroid)
    return transforms.cpu().numpy()
