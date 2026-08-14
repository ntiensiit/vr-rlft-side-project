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
from grasping_ai.models.flow import load_flow_model_from_state
from grasping_ai.training.checkpoint_io import (
    checkpoint_scalar_int,
    load_torch_checkpoint,
)

GraspPoseGenerator = Callable[[np.ndarray, int], np.ndarray]


def load_grasp_model_checkpoint(checkpoint_path: Path, device: str) -> dict[str, Any]:
    """Load a grasp-generation model checkpoint from disk.

    Args:
        checkpoint_path: Path to the checkpoint file produced during training.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.

    Returns:
        Deserialized checkpoint dictionary for grasp-generation models.
    """
    return load_torch_checkpoint(checkpoint_path, device)


def build_diffusion_grasp_generator(
    checkpoint: dict[str, Any],
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
    hidden_dim = checkpoint_scalar_int(checkpoint["hidden_dim"])
    num_layers = checkpoint_scalar_int(checkpoint["num_layers"])

    from grasping_ai.models.diffusion import GraspGeneratorModel, build_diffusion_sampler

    model = GraspGeneratorModel(feature_dim, hidden_dim, num_layers)
    model.load_state_dict(cast(dict[str, Any], checkpoint["model_state_dict"]))
    model.to(device)
    model.eval()

    sampler = build_diffusion_sampler(num_diffusion_steps)

    def generator(point_cloud: np.ndarray, num_grasps: int = 10) -> np.ndarray:
        pc_tensor = prepare_point_cloud_tensor(point_cloud, device)

        with torch.no_grad():
            cond, frame, centroid = encode_grasp_conditioning(model.encoder, pc_tensor)

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
    checkpoint: dict[str, Any],
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
    hidden_dim = checkpoint_scalar_int(checkpoint["hidden_dim"])
    num_layers = checkpoint_scalar_int(checkpoint["num_layers"])

    model = load_flow_model_from_state(checkpoint, feature_dim, hidden_dim, num_layers, device)

    from grasping_ai.models.flow import build_flow_integrator, sample_grasps_with_flow

    integrator = build_flow_integrator(num_flow_steps)

    def generator(point_cloud: np.ndarray, num_grasps: int = 10) -> np.ndarray:
        pc_tensor = prepare_point_cloud_tensor(point_cloud, device)

        with torch.no_grad():
            cond, frame, centroid = encode_grasp_conditioning(model.encoder, pc_tensor)

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


def generate_candidate_grasps(generator: GraspPoseGenerator, point_cloud: np.ndarray, num_grasps: int) -> np.ndarray:
    """Produce a fixed number of grasp candidates for a point cloud.

    Args:
        generator: Callable grasp generator produced by ``build_*_grasp_generator``.
        point_cloud: Object point cloud with shape ``(N, 3)``.
        num_grasps: Number of candidate grasps to produce.

    Returns:
        Candidate grasp poses represented as an array.
    """
    return generator(point_cloud, num_grasps)
