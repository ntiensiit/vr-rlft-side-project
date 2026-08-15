"""Run grasp inference on saved point clouds."""

from __future__ import annotations

import argparse
from pathlib import Path

from grasping_ai.config.yaml_loader import (
    config_get,
    config_int,
    config_path,
    load_project_yaml_config,
    parse_clean_argv,
    parse_config_dir_from_argv,
)
from grasping_ai.inference.grasp_inference_runtime import run_single_object_grasp_inference

if __name__ == "__main__":
    config_dir = parse_config_dir_from_argv()
    cfg = load_project_yaml_config(config_dir)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-dir", type=Path, default=config_dir)
    parser = argparse.ArgumentParser(
        description="Generate grasp candidates from a trained checkpoint",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config_path(cfg, "model", "checkpoint")
        or config_path(cfg, "diffusion", "checkpoint")
        or config_path(cfg, "flow", "checkpoint"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method",
        type=str,
        choices=["diffusion", "flow"],
        default=str(config_get(cfg, "default_method")),
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=int(config_get(cfg, "architecture", "feature_dim")),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=config_int(cfg, "model", "inference_steps", default=0)
        or config_int(cfg, "diffusion", "inference_steps", default=0)
        or config_int(cfg, "flow", "inference_steps", default=0)
        or 5,
        help="Diffusion denoising steps or flow integration steps",
    )
    parser.add_argument(
        "--num-grasps",
        type=int,
        default=int(config_get(cfg, "architecture", "num_grasps")),
    )
    parser.add_argument("--device", type=str, default=str(config_get(cfg, "device")))
    parser.add_argument("--seed", type=int, default=int(config_get(cfg, "seed")))
    parser.add_argument("--observation", type=Path, default=None)
    parser.add_argument(
        "--ycb-root",
        type=Path,
        default=config_path(cfg, "paths", "ycb_root"),
    )
    parser.add_argument("--object-id", type=str, default=None)
    args = parser.parse_args(parse_clean_argv())
    if args.checkpoint is None:
        parser.error(
            "--checkpoint is required (set in configs/model/diffusion.yaml or "
            "configs/model/flow.yaml or pass explicitly)",
        )
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
        ycb_root=args.ycb_root if args.observation is None else None,
        object_id=args.object_id,
    )
