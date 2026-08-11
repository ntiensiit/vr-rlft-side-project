from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

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
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint: {e}") from e
    from typing import cast
    return cast(dict[str, torch.Tensor], checkpoint)


def _ckpt_int(checkpoint: dict[str, Any], key: str) -> int:
    """Extract an integer value from a checkpoint dict."""
    val = checkpoint[key]
    if isinstance(val, torch.Tensor):
        return int(val.item())
    return int(val)


def vec_to_se3(x: torch.Tensor) -> torch.Tensor:
    """Convert (M, 9) vectors to SE(3) transforms (M, 4, 4).

    Each row contains translation (3) + 6D rotation representation.
    """
    m = x.shape[0]
    t = x[:, :3]
    v = x[:, 3:9]

    a1 = v[:, :3]
    a2 = v[:, 3:]

    b1 = a1 / torch.norm(a1, dim=-1, keepdim=True).clamp(min=1e-8)
    dot = torch.sum(b1 * a2, dim=-1, keepdim=True)
    u2 = a2 - dot * b1
    b2 = u2 / torch.norm(u2, dim=-1, keepdim=True).clamp(min=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    rot = torch.stack([b1, b2, b3], dim=-1)

    se3 = torch.eye(4, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(m, 1, 1)
    se3[:, :3, :3] = rot
    se3[:, :3, 3] = t
    return se3


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
    hidden_dim = _ckpt_int(checkpoint, "hidden_dim")
    num_layers = _ckpt_int(checkpoint, "num_layers")

    from grasping_ai.models.diffusion import GraspGeneratorModel, build_diffusion_sampler
    model = GraspGeneratorModel(feature_dim, hidden_dim, num_layers)
    model.load_state_dict(cast(dict[str, Any], checkpoint["model_state_dict"]))
    model.to(device)
    model.eval()

    sampler = build_diffusion_sampler(num_diffusion_steps)

    def generator(point_cloud: np.ndarray, num_grasps: int = 10) -> np.ndarray:
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            raise ValueError(f"point_cloud must have shape (N, 3), got {point_cloud.shape}")

        from grasping_ai.perception.pointcloud import normalize_point_cloud
        pc_norm = normalize_point_cloud(point_cloud)
        pc_tensor = torch.from_numpy(pc_norm).float().to(device).unsqueeze(0)

        with torch.no_grad():
            from grasping_ai.models.equivariant_encoder import (
                encode_point_cloud,
                pool_object_features,
            )
            features = encode_point_cloud(model.encoder, pc_tensor)
            cond = pool_object_features(features)

            from grasping_ai.models.diffusion import sample_grasps_with_diffusion
            rng = torch.Generator(device=device)
            rng.manual_seed(42)

            samples = sample_grasps_with_diffusion(
                sampler=sampler,
                score_model=model.score_net,
                conditioning=cond,
                grasp_dim=9,
                num_samples=num_grasps,
                rng=rng,
            )

            samples_flat = samples.view(-1, 9)
            transforms = vec_to_se3(samples_flat)

        return transforms.cpu().numpy()

    return generator


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
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            raise ValueError(f"point_cloud must have shape (N, 3), got {point_cloud.shape}")

        from grasping_ai.perception.pointcloud import normalize_point_cloud
        pc_norm = normalize_point_cloud(point_cloud)
        pc_tensor = torch.from_numpy(pc_norm).float().to(device).unsqueeze(0)

        with torch.no_grad():
            from grasping_ai.models.equivariant_encoder import (
                encode_point_cloud,
                pool_object_features,
            )
            features = encode_point_cloud(model.encoder, pc_tensor)
            cond = pool_object_features(features)

            rng = torch.Generator(device=device)
            rng.manual_seed(42)

            samples = sample_grasps_with_flow(
                integrator=integrator,
                flow_field=model.flow_field,
                conditioning=cond,
                grasp_dim=9,
                num_samples=num_grasps,
                rng=rng,
            )

            samples_flat = samples.view(-1, 9)
            transforms = vec_to_se3(samples_flat)

        return transforms.cpu().numpy()

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
