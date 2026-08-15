"""Encode and decode fixed-size grasp pose vectors."""

from __future__ import annotations

import numpy as np
import torch

from grasping_ai.utils.constants import GRASP_VECTOR_DIM, POINT_CLOUD_NDIM, SE3_MATRIX_SHAPE, TORCH_NORM_CLAMP_MIN


def se3_to_vec(t_matrix: np.ndarray) -> np.ndarray:
    """Convert a ``(4, 4)`` SE(3) matrix to a 9D position + rotation-column vector.

    Args:
        t_matrix: Rigid transform with shape ``(4, 4)``.

    Returns:
        A vector ``[t, r1, r2]`` with shape ``(9,)`` where ``t`` is translation
        and ``r1``/``r2`` are the first two rotation columns.
    """
    if not isinstance(t_matrix, np.ndarray) or t_matrix.shape != SE3_MATRIX_SHAPE:
        raise ValueError("t_matrix must be a (4, 4) numpy array")
    t = t_matrix[:3, 3]
    r1 = t_matrix[:3, 0]
    r2 = t_matrix[:3, 1]
    return np.concatenate([t, r1, r2])


def vec_to_se3(x: torch.Tensor) -> torch.Tensor:
    """Convert ``(M, 9)`` grasp vectors to ``(M, 4, 4)`` SE(3) transforms.

    Each row contains translation ``(3)`` plus a 6D continuous rotation
    representation formed by two 3-vectors that are orthonormalized with
    Gram-Schmidt.

    Args:
        x: Grasp vectors with shape ``(M, 9)``.

    Returns:
        Batch of homogeneous transforms with shape ``(M, 4, 4)``.
    """
    if x.ndim != POINT_CLOUD_NDIM or x.shape[1] != GRASP_VECTOR_DIM:
        raise ValueError(f"x must have shape (M, 9), got {x.shape}")

    m = x.shape[0]
    t = x[:, :3]
    v = x[:, 3:9]

    a1 = v[:, :3]
    a2 = v[:, 3:]

    b1 = a1 / torch.norm(a1, dim=-1, keepdim=True).clamp(min=TORCH_NORM_CLAMP_MIN)
    dot = torch.sum(b1 * a2, dim=-1, keepdim=True)
    u2 = a2 - dot * b1
    b2 = u2 / torch.norm(u2, dim=-1, keepdim=True).clamp(min=TORCH_NORM_CLAMP_MIN)
    b3 = torch.cross(b1, b2, dim=-1)

    rot = torch.stack([b1, b2, b3], dim=-1)

    se3 = torch.eye(4, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(m, 1, 1)
    se3[:, :3, :3] = rot
    se3[:, :3, 3] = t
    return se3
