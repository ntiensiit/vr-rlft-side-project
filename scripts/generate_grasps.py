from pathlib import Path

from grasping_ai.inference.grasp_generator import (
    build_diffusion_grasp_generator,
    load_grasp_model_checkpoint,
)
from grasping_ai.pipelines.generate_grasps import (
    generate_grasps_for_dataset,
    write_generated_grasps,
)
from grasping_ai.sensors.pointcloud_sensor import acquire_point_cloud_stream


def generate_grasps_main(
    checkpoint_path: Path,
    observation_paths: list[Path],
    output_path: Path,
    feature_dim: int,
    num_diffusion_steps: int,
    num_grasps: int,
    device: str,
) -> None:
    """Load a diffusion grasp model and generate grasps for a set of objects.

    Args:
        checkpoint_path: Path to the trained diffusion model checkpoint.
        observation_paths: Paths to per-object sensor observations.
        output_path: Destination path for the generated-grasp output file.
        feature_dim: Conditioning feature dimension expected by the model.
        num_diffusion_steps: Number of denoising steps used at inference.
        num_grasps: Number of candidate grasps to generate per object.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.
    """
    checkpoint = load_grasp_model_checkpoint(checkpoint_path, device)
    generator = build_diffusion_grasp_generator(
        checkpoint, feature_dim, num_diffusion_steps, device
    )
    point_clouds = list(acquire_point_cloud_stream(observation_paths))
    grasps = generate_grasps_for_dataset(point_clouds, generator, num_grasps)
    write_generated_grasps(output_path, {f"object_{i}": grasp for i, grasp in enumerate(grasps)})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate grasp poses using a trained model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-dim", type=int, required=True)
    parser.add_argument("--num-diffusion-steps", type=int, required=True)
    parser.add_argument("--num-grasps", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()
    generate_grasps_main(
        args.checkpoint,
        args.observations,
        args.output,
        args.feature_dim,
        args.num_diffusion_steps,
        args.num_grasps,
        args.device,
    )
