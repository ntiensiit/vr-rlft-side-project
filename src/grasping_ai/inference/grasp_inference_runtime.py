from __future__ import annotations

from pathlib import Path

import numpy as np

from grasping_ai.data.pointcloud_dataset import resolve_ycb_object_id
from grasping_ai.inference.grasp_generator import (
    build_diffusion_grasp_generator,
    build_flow_grasp_generator,
    generate_candidate_grasps,
    load_grasp_model_checkpoint,
)
from grasping_ai.sensors.pointcloud_sensor import sample_point_cloud_from_mesh


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
    """Generate grasp candidates for one object and optionally persist them.

    Args:
        checkpoint_path: Trained grasp-generation checkpoint on disk.
        output_path: Destination ``.npy`` path for generated grasp poses.
        method: ``"diffusion"`` or ``"flow"``; must match the checkpoint type.
        feature_dim: Conditioning feature dimension expected by the model.
        num_steps: Diffusion denoising steps or flow integration steps.
        num_grasps: Number of candidate grasps to sample.
        device: Torch device identifier such as ``"cpu"`` or ``"cuda"``.
        seed: Random seed for point-cloud sampling and grasp generation.
        observation_path: Optional precomputed object point cloud ``.npy``.
        ycb_root: Optional YCB root used when ``observation_path`` is omitted.
        object_id: YCB object identifier required with ``ycb_root``.

    Returns:
        Generated grasp poses with shape ``(K, 4, 4)``.

    Raises:
        ValueError: If inputs are inconsistent or ``method`` is unsupported.
        FileNotFoundError: If a required observation or YCB path is missing.
    """
    if observation_path is None and (ycb_root is None or object_id is None):
        raise ValueError("Provide either observation_path or both ycb_root and object_id")
    if observation_path is not None and (ycb_root is not None or object_id is not None):
        raise ValueError("Pass observation_path or ycb_root/object_id, not both")

    if observation_path is not None:
        if not observation_path.is_file():
            raise FileNotFoundError(f"Observation file not found: {observation_path}")
        point_cloud = np.load(observation_path)
    else:
        if ycb_root is None or object_id is None:
            raise ValueError("Provide both ycb_root and object_id when observation_path is not given")
        if not ycb_root.is_dir():
            raise FileNotFoundError(f"YCB root directory not found: {ycb_root}")
        mesh_path = resolve_ycb_object_id(ycb_root, object_id)
        rng = np.random.default_rng(seed)
        point_cloud = sample_point_cloud_from_mesh(mesh_path, num_grasps * 8, rng)

    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        raise ValueError(f"point_cloud must have shape (N, 3), got {point_cloud.shape}")

    checkpoint = load_grasp_model_checkpoint(checkpoint_path, device)
    if method == "diffusion":
        generator = build_diffusion_grasp_generator(checkpoint, feature_dim, num_steps, device, seed)
    elif method == "flow":
        generator = build_flow_grasp_generator(checkpoint, feature_dim, num_steps, device, seed)
    else:
        raise ValueError(f"method must be 'diffusion' or 'flow', got '{method}'")

    grasp_poses = generate_candidate_grasps(generator, point_cloud, num_grasps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, grasp_poses)
    return grasp_poses
