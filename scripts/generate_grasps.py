from pathlib import Path

from grasping_ai.config.yaml_loader import (
    config_get,
    config_path,
    load_project_yaml_config,
    parse_config_dir_from_argv,
)
from grasping_ai.inference.grasp_generator import (
    build_diffusion_grasp_generator,
    generate_candidate_grasps,
    load_grasp_model_checkpoint,
)
from grasping_ai.pipelines.generate_grasps import write_generated_grasps
from grasping_ai.sensors.pointcloud_sensor import acquire_point_cloud_stream

if __name__ == "__main__":
    import argparse

    config_dir = parse_config_dir_from_argv()
    cfg = load_project_yaml_config(config_dir, "base", "model")
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-dir", type=Path, default=config_dir)
    parser = argparse.ArgumentParser(
        description="Generate grasp poses using a trained model",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config_path(cfg, "diffusion", "checkpoint"),
    )
    parser.add_argument("--observations", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=config_path(cfg, "diffusion", "exports", "grasp_candidates"),
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=int(config_get(cfg, "architecture", "feature_dim")),
    )
    parser.add_argument(
        "--num-diffusion-steps",
        type=int,
        default=int(config_get(cfg, "diffusion", "inference_steps")),
    )
    parser.add_argument(
        "--num-grasps",
        type=int,
        default=int(config_get(cfg, "architecture", "num_grasps")),
    )
    parser.add_argument("--device", type=str, default=str(config_get(cfg, "device")))
    parser.add_argument("--seed", type=int, default=int(config_get(cfg, "seed")))
    args = parser.parse_args()
    if args.checkpoint is None:
        parser.error(
            "--checkpoint is required (set in configs/model/default.yaml diffusion.checkpoint or pass explicitly)"
        )
    if args.output is None:
        parser.error(
            "--output is required (set in configs/model/default.yaml "
            "diffusion.exports.grasp_candidates or pass explicitly)"
        )

    checkpoint = load_grasp_model_checkpoint(args.checkpoint, args.device)
    generator = build_diffusion_grasp_generator(
        checkpoint,
        args.feature_dim,
        args.num_diffusion_steps,
        args.device,
        args.seed,
    )
    point_clouds = list(acquire_point_cloud_stream(args.observations))
    grasps = [generate_candidate_grasps(generator, point_cloud, args.num_grasps) for point_cloud in point_clouds]
    write_generated_grasps(args.output, {f"object_{i}": grasp for i, grasp in enumerate(grasps)})
