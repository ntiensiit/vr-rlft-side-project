from __future__ import annotations

from pathlib import Path

import numpy as np

from grasping_ai.data.pointcloud_dataset import resolve_ycb_object_id
from grasping_ai.inference.grasp_generator import (
    GraspPoseGenerator,
    build_diffusion_grasp_generator,
    build_flow_grasp_generator,
    load_grasp_model_checkpoint,
)
from grasping_ai.sensors.pointcloud_sensor import sample_point_cloud_from_mesh


def build_grasp_generator_from_checkpoint(
    method: str,
    checkpoint: dict[str, object],
    feature_dim: int,
    num_steps: int,
    device: str,
    seed: int = 42,
) -> GraspPoseGenerator:
    """Construct a diffusion or flow grasp generator from a loaded checkpoint."""
    if method == "diffusion":
        return build_diffusion_grasp_generator(
            checkpoint, feature_dim, num_steps, device, seed
        )
    if method == "flow":
        return build_flow_grasp_generator(
            checkpoint, feature_dim, num_steps, device, seed
        )
    raise ValueError(f"method must be 'diffusion' or 'flow', got '{method}'")


def load_inference_point_cloud(
    observation_path: Path | None,
    ycb_root: Path | None,
    object_id: str | None,
    num_grasps: int,
    seed: int,
) -> np.ndarray:
    """Resolve a single-object point cloud for grasp inference."""
    if observation_path is None and (ycb_root is None or object_id is None):
        raise ValueError(
            "Provide either observation_path or both ycb_root and object_id"
        )
    if observation_path is not None and (ycb_root is not None or object_id is not None):
        raise ValueError("Pass observation_path or ycb_root/object_id, not both")

    if observation_path is not None:
        if not observation_path.is_file():
            raise FileNotFoundError(f"Observation file not found: {observation_path}")
        point_cloud = np.load(observation_path)
    else:
        if ycb_root is None or object_id is None:
            raise ValueError(
                "Provide both ycb_root and object_id when observation_path is not given"
            )
        if not ycb_root.is_dir():
            raise FileNotFoundError(f"YCB root directory not found: {ycb_root}")
        mesh_path = resolve_ycb_object_id(ycb_root, object_id)
        rng = np.random.default_rng(seed)
        point_cloud = sample_point_cloud_from_mesh(mesh_path, num_grasps * 8, rng)

    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        raise ValueError(
            f"point_cloud must have shape (N, 3), got {point_cloud.shape}"
        )
    return point_cloud


def run_single_object_grasp_inference(
    checkpoint_path: Path,
    output_path: Path,
    method: str,
    feature_dim: int,
    num_steps: int,
    num_grasps: int,
    device: str,
    seed: int,
    observation_path: Path | None = None,
    ycb_root: Path | None = None,
    object_id: str | None = None,
) -> np.ndarray:
    """Generate grasp candidates for one object and optionally persist them."""
    from grasping_ai.inference.grasp_generator import generate_candidate_grasps

    point_cloud = load_inference_point_cloud(
        observation_path, ycb_root, object_id, num_grasps, seed
    )
    checkpoint = load_grasp_model_checkpoint(checkpoint_path, device)
    generator = build_grasp_generator_from_checkpoint(
        method, checkpoint, feature_dim, num_steps, device, seed
    )
    grasp_poses = generate_candidate_grasps(generator, point_cloud, num_grasps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, grasp_poses)
    return grasp_poses
