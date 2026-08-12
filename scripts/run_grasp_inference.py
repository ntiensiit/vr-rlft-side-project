from __future__ import annotations

import argparse
from pathlib import Path

from grasping_ai.inference.grasp_inference_runtime import run_single_object_grasp_inference

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
    run_single_object_grasp_inference(
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
