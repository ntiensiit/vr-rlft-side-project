from __future__ import annotations

import argparse
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


def run_inference_main(
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
) -> None:
    """Generate grasp candidates from a trained grasp-generation checkpoint.

    Single-object runtime entry point that branches on ``method`` so the same
    script serves both the diffusion and flow training pipelines. The output is
    a plain ``(num_grasps, 4, 4)`` numpy array so it can be fed directly to
    ``scripts/run_simulation.py`` (which loads plain arrays via
    ``np.load(grasps_path)`` without ``allow_pickle=True``).

    Args:
        checkpoint_path: Path to a checkpoint produced by ``scripts/train.py`` or
            ``scripts/train_flow.py``. Must contain ``feature_dim``,
            ``hidden_dim``, ``num_layers``, ``epoch``, and
            ``model_state_dict``.
        output_path: Destination path for the generated-grasp ``.npy`` file.
        method: ``"diffusion"`` or ``"flow"``.
        feature_dim: Conditioning feature dimension expected by the model.
        num_steps: Number of denoising steps (diffusion) or flow integration
            steps (flow).
        num_grasps: Number of candidate grasps to generate.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.
        seed: Random seed used to draw initial noise.
        observation_path: Optional path to a serialized point-cloud observation
            (``.npy``). Exactly one of ``observation_path`` or
            ``(ycb_root, object_id)`` must be provided.
        ycb_root: Optional root directory of the raw YCB object set.
        object_id: Optional YCB object identifier; together with ``ycb_root``
            samples a fresh point cloud from the YCB mesh.
    """
    if method not in ("diffusion", "flow"):
        raise ValueError(
            f"method must be 'diffusion' or 'flow', got '{method}'"
        )
    if observation_path is None and (ycb_root is None or object_id is None):
        raise ValueError(
            "Provide either --observation or both --ycb-root and --object-id"
        )
    if observation_path is not None and (
        ycb_root is not None or object_id is not None
    ):
        raise ValueError(
            "Pass --observation or --ycb-root/--object-id, not both"
        )

    if observation_path is not None:
        if not observation_path.is_file():
            raise FileNotFoundError(
                f"Observation file not found: {observation_path}"
            )
        point_cloud = np.load(observation_path)
    else:
        if ycb_root is None or object_id is None:
            raise ValueError(
                "Provide both --ycb-root and --object-id when --observation is not given"
            )
        if not ycb_root.is_dir():
            raise FileNotFoundError(
                f"YCB root directory not found: {ycb_root}"
            )
        mesh_path = resolve_ycb_object_id(ycb_root, object_id)
        rng = np.random.default_rng(seed)
        point_cloud = sample_point_cloud_from_mesh(mesh_path, num_grasps * 8, rng)

    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        raise ValueError(
            f"point_cloud must have shape (N, 3), got {point_cloud.shape}"
        )

    checkpoint = load_grasp_model_checkpoint(checkpoint_path, device)

    if method == "diffusion":
        generator = build_diffusion_grasp_generator(
            checkpoint, feature_dim, num_steps, device, seed
        )
    else:
        generator = build_flow_grasp_generator(
            checkpoint, feature_dim, num_steps, device, seed
        )

    grasp_poses = generate_candidate_grasps(generator, point_cloud, num_grasps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, grasp_poses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate grasp candidates from a trained checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method",
        type=str,
        choices=["diffusion", "flow"],
        default="diffusion",
        help="Inference method matching the trained checkpoint",
    )
    parser.add_argument("--feature-dim", type=int, required=True)
    parser.add_argument(
        "--num-steps",
        type=int,
        required=True,
        help="Diffusion denoising steps or flow integration steps",
    )
    parser.add_argument("--num-grasps", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--observation", type=Path, default=None)
    parser.add_argument("--ycb-root", type=Path, default=None)
    parser.add_argument("--object-id", type=str, default=None)
    args = parser.parse_args()
    run_inference_main(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        method=args.method,
        feature_dim=args.feature_dim,
        num_steps=args.num_steps,
        num_grasps=args.num_grasps,
        device=args.device,
        seed=args.seed,
        observation_path=args.observation,
        ycb_root=args.ycb_root,
        object_id=args.object_id,
    )
