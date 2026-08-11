from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

GraspPoseGenerator = Callable[[np.ndarray], np.ndarray]


def load_grasp_model_checkpoint(checkpoint_path: Path, device: str) -> dict[str, torch.Tensor]:
    """Load a grasp-generation model checkpoint from disk.

    Args:
        checkpoint_path: Path to the checkpoint file produced during training.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.

    Returns:
        A mapping from parameter names to tensors representing the trained
        grasp-generation model.
    """
    raise NotImplementedError


def build_diffusion_grasp_generator(
    checkpoint: dict[str, torch.Tensor],
    feature_dim: int,
    num_diffusion_steps: int,
    device: str,
) -> GraspPoseGenerator:
    """Create a callable that generates grasps using a diffusion model.

    Args:
        checkpoint: Loaded model parameters from ``load_grasp_model_checkpoint``.
        feature_dim: Conditioning feature dimension expected by the score model.
        num_diffusion_steps: Number of denoising steps for the sampler.
        device: Device identifier on which inference runs.

    Returns:
        A function that takes a point cloud ``(N, 3)`` and returns a set of
        candidate grasp poses as a numpy array.
    """
    raise NotImplementedError


def build_flow_grasp_generator(
    checkpoint: dict[str, torch.Tensor],
    feature_dim: int,
    num_flow_steps: int,
    device: str,
) -> GraspPoseGenerator:
    """Create a callable that generates grasps using a flow model.

    Args:
        checkpoint: Loaded model parameters from ``load_grasp_model_checkpoint``.
        feature_dim: Conditioning feature dimension expected by the flow field.
        num_flow_steps: Number of integration steps for the flow sampler.
        device: Device identifier on which inference runs.

    Returns:
        A function that takes a point cloud ``(N, 3)`` and returns a set of
        candidate grasp poses as a numpy array.
    """
    raise NotImplementedError


def generate_candidate_grasps(
    generator: GraspPoseGenerator, point_cloud: np.ndarray, num_grasps: int
) -> np.ndarray:
    """Produce a fixed number of grasp candidates for a point cloud.

    Args:
        generator: Callable grasp generator produced by ``build_*_grasp_generator``.
        point_cloud: Object point cloud with shape ``(N, 3)``.
        num_grasps: Number of candidate grasps to produce.

    Returns:
        Candidate grasp poses represented as an array.
    """
    raise NotImplementedError
