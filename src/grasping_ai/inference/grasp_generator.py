from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from grasping_ai.inference.grasp_sampling import (
    encode_grasp_conditioning,
    prepare_point_cloud_tensor,
    sample_to_world_frame,
)
from grasping_ai.training.checkpoint_io import load_torch_checkpoint

GraspPoseGenerator = Callable[[np.ndarray, int], np.ndarray]


def load_grasp_model_checkpoint(checkpoint_path: Path, device: str) -> dict[str, torch.Tensor]:
    """Load a grasp-generation model checkpoint from disk.

    Args:
        checkpoint_path: Path to the checkpoint file produced during training.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.

    Returns:
        A mapping from parameter names to tensors representing the trained
        grasp-generation model.
    """
    if not isinstance(checkpoint_path, Path):
        raise TypeError("checkpoint_path must be a pathlib.Path instance")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    return cast(dict[str, torch.Tensor], checkpoint)


def _ckpt_int(checkpoint: dict[str, Any], key: str) -> int:
    """Extract an integer value from a checkpoint dict."""
    val = checkpoint[key]
    if isinstance(val, torch.Tensor):
        return int(val.item())
    return int(val)


def build_diffusion_grasp_generator(
    checkpoint: dict[str, torch.Tensor],
    feature_dim: int,
    num_diffusion_steps: int,
    device: str,
    seed: int = 42,
) -> GraspPoseGenerator:
    """Create a callable that generates grasps using a diffusion model.

    Args:
        checkpoint: Loaded model parameters from ``load_grasp_model_checkpoint``.
        feature_dim: Conditioning feature dimension expected by the score model.
        num_diffusion_steps: Number of denoising steps for the sampler.
        device: Device identifier on which inference runs.
        seed: Random seed used to draw initial diffusion noise. Sampling is
            reproducible for a fixed seed.

    Returns:
        A function that takes a point cloud ``(N, 3)`` and returns a set of
        candidate grasp poses as a numpy array.
    """
    hidden_dim = _ckpt_int(checkpoint, "hidden_dim")
    num_layers = _ckpt_int(checkpoint, "num_layers")

    from grasping_ai.models.diffusion import GraspGeneratorModel, build_diffusion_sampler
    model = GraspGeneratorModel(feature_dim, hidden_dim, num_layers)
    model.load_state_dict(cast(dict[str, Any], checkpoint["model_state_dict"]))
    model.to(device)
    model.eval()

    sampler = build_diffusion_sampler(num_diffusion_steps)

    def generator(point_cloud: np.ndarray, num_grasps: int = 10) -> np.ndarray:
        pc_tensor = prepare_point_cloud_tensor(point_cloud, device)

        with torch.no_grad():
            cond, frame, centroid = encode_grasp_conditioning(
                model.encoder, pc_tensor
            )

            from grasping_ai.models.diffusion import sample_grasps_with_diffusion
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

            samples = sample_grasps_with_diffusion(
                sampler=sampler,
                score_model=model.score_net,
                conditioning=cond,
                grasp_dim=9,
                num_samples=num_grasps,
                rng=rng,
            )

            return sample_to_world_frame(samples, frame, centroid)

    return generator


def build_flow_grasp_generator(
    checkpoint: dict[str, torch.Tensor],
    feature_dim: int,
    num_flow_steps: int,
    device: str,
    seed: int = 42,
) -> GraspPoseGenerator:
    """Create a callable that generates grasps using a flow model.

    Args:
        checkpoint: Loaded model parameters from ``load_grasp_model_checkpoint``.
        feature_dim: Conditioning feature dimension expected by the flow field.
        num_flow_steps: Number of integration steps for the flow sampler.
        device: Device identifier on which inference runs.
        seed: Random seed used to draw initial flow samples. Sampling is
            reproducible for a fixed seed.

    Returns:
        A function that takes a point cloud ``(N, 3)`` and returns a set of
        candidate grasp poses as a numpy array.
    """
    hidden_dim = _ckpt_int(checkpoint, "hidden_dim")
    num_layers = _ckpt_int(checkpoint, "num_layers")

    class FlowModelWrapper(torch.nn.Module):
        def __init__(self, f_dim: int, h_dim: int, n_layers: int) -> None:
            super().__init__()
            self.feature_dim = f_dim
            self.hidden_dim = h_dim
            self.num_layers = n_layers
            from grasping_ai.models.equivariant_encoder import build_equivariant_encoder
            from grasping_ai.models.flow import build_flow_field
            self.encoder = build_equivariant_encoder(f_dim, n_layers)
            self.flow_field = build_flow_field(f_dim, h_dim, n_layers)

    model = FlowModelWrapper(feature_dim, hidden_dim, num_layers)
    model.load_state_dict(cast(dict[str, Any], checkpoint["model_state_dict"]))
    model.to(device)
    model.eval()

    from grasping_ai.models.flow import build_flow_integrator, sample_grasps_with_flow
    integrator = build_flow_integrator(num_flow_steps)

    def generator(point_cloud: np.ndarray, num_grasps: int = 10) -> np.ndarray:
        pc_tensor = prepare_point_cloud_tensor(point_cloud, device)

        with torch.no_grad():
            cond, frame, centroid = encode_grasp_conditioning(
                model.encoder, pc_tensor
            )

            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

            samples = sample_grasps_with_flow(
                integrator=integrator,
                flow_field=model.flow_field,
                conditioning=cond,
                grasp_dim=9,
                num_samples=num_grasps,
                rng=rng,
            )

            return sample_to_world_frame(samples, frame, centroid)

    return generator


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
    return generator(point_cloud, num_grasps)
