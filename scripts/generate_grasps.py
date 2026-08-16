"""CLI entry point for analytical grasp generation."""

from __future__ import annotations

from grasping_ai.config import SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig

from grasping_ai.inference.grasp_generator import (
    build_diffusion_grasp_generator,
    generate_candidate_grasps,
    load_grasp_model_checkpoint,
)

from grasping_ai.pipelines.generate_grasps import write_generated_grasps

from grasping_ai.sensors.pointcloud_sensor import acquire_point_cloud_stream

from pathlib import Path

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/generate_grasps")
def main(cfg: DictConfig) -> None:
    yaml_config = FlattenedYAMLConfig(cfg)
    checkpoint = yaml_config.value("checkpoint", "model", "checkpoint", value_type=Path, script_or=True, required=True)
    output = yaml_config.value(
        "output", "model", "exports", "grasp_candidates", value_type=Path, script_or=True, required=True
    )
    observations = yaml_config.value("observations", "observations", "files", value_type=list[Path], script_or=True)
    device = str(yaml_config.get("device"))
    seed = yaml_config.value("seed", value_type=int)

    model_checkpoint = load_grasp_model_checkpoint(checkpoint, device)
    generator = build_diffusion_grasp_generator(
        model_checkpoint,
        yaml_config.value("architecture", "feature_dim", value_type=int),
        device,
        seed,
    )
    point_clouds = list(acquire_point_cloud_stream(observations))
    grasps = [
        generate_candidate_grasps(generator, point_cloud, yaml_config.value("architecture", "num_grasps", value_type=int))
        for point_cloud in point_clouds
    ]
    write_generated_grasps(output, {f"object_{i}": grasp for i, grasp in enumerate(grasps)})

if __name__ == "__main__":
    main()
