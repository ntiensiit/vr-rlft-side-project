"""Prepare processed grasp datasets from raw YCB assets."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import open3d as _open3d  # noqa: F401
import hydra
import numpy as np

from grasping_ai.config import SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.pipelines.prepare_synthetic_data import (
    generate_synthetic_dataset,
    prepare_data_index,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/prepare_data")
def main(cfg: DictConfig) -> None:
    """Prepare processed grasp datasets or rebuild the dataset index."""
    yaml_config = FlattenedYAMLConfig(cfg)
    mode = str(yaml_config.value("mode", "prepare", "mode", value_type=object, script_or=True))
    dataset_root = yaml_config.value("paths", "dataset_root", value_type=Path)
    output_dir = yaml_config.value("output_dir", "prepare", "output_dir", value_type=Path, script_or=True)
    output_index = yaml_config.value("paths", "output_index", value_type=Path, required=True)
    target_dir = output_dir if output_dir is not None else dataset_root

    if mode == "synthetic":
        if dataset_root is None and output_dir is None:
            msg = "prepare.output_dir or paths.dataset_root is required when prepare.mode is synthetic"
            raise ValueError(msg)
        ycb_root = yaml_config.value("paths", "ycb_root", value_type=Path, required=True)
        if target_dir is None:
            msg = "prepare.output_dir or paths.dataset_root is required when prepare.mode is synthetic"
            raise ValueError(msg)

        gripper_close = yaml_config.get_path("robot", "gripper", "close_command")
        close_command = np.asarray(gripper_close, dtype=np.float64) if isinstance(gripper_close, list) else None
        sim_position = yaml_config.value("synthetic", "sim_object_position", value_type=list[float], required=True)
        sim_object_position = np.asarray(sim_position, dtype=np.float64)
        sim_table = yaml_config.get_path("synthetic", "sim_table_xml")
        table_xml = Path(str(sim_table)) if isinstance(sim_table, str) else None
        if table_xml is not None and not table_xml.is_absolute():
            table_xml = Path(__file__).resolve().parents[1] / table_xml

        generate_synthetic_dataset(
            ycb_root=ycb_root,
            output_dir=target_dir,
            num_samples=yaml_config.value("synthetic", "num_samples", value_type=int),
            num_grasps=yaml_config.value("synthetic", "num_grasps", value_type=int),
            gripper_width=yaml_config.value("synthetic", "gripper_width", value_type=float),
            seed=yaml_config.value("synthetic", "seed", value_type=int),
            required_objects=yaml_config.value("objects", "ids", value_type=list[str]),
            oversample_factor=yaml_config.value("synthetic", "oversample_factor", value_type=int),
            oversample_extra=yaml_config.value("synthetic", "oversample_extra", value_type=int),
            neighborhood_size=yaml_config.value("synthetic", "neighborhood_size", value_type=int),
            voxel_size=yaml_config.value("synthetic", "voxel_size", value_type=float),
            strict_antipodal_dot=yaml_config.value("synthetic", "strict_antipodal_dot", value_type=float),
            strict_alignment_dot=yaml_config.value("synthetic", "strict_alignment_dot", value_type=float),
            relaxed_antipodal_dot=yaml_config.value("synthetic", "relaxed_antipodal_dot", value_type=float),
            allow_relaxed=yaml_config.value("synthetic", "allow_relaxed", value_type=bool),
            search_multiplier=yaml_config.value("synthetic", "search_multiplier", value_type=int),
            candidate_multiplier=yaml_config.value("synthetic", "candidate_multiplier", value_type=int),
            min_grasp_translation=yaml_config.value("synthetic", "min_grasp_translation", value_type=float),
            min_grasp_rotation=yaml_config.value("synthetic", "min_grasp_rotation", value_type=float),
            min_quality_score=yaml_config.value("synthetic", "min_quality_score", value_type=float),
            friction_coefficient=yaml_config.value("synthetic", "friction_coefficient", value_type=float),
            collision_clearance=yaml_config.value("synthetic", "collision_clearance", value_type=float),
            sim_validate=yaml_config.value("synthetic", "sim_validate", value_type=bool),
            mjcf_root=yaml_config.value("paths", "ycb_mjcf", value_type=Path),
            robot_xml=yaml_config.value("robot", "description", value_type=Path),
            num_simulation_steps=yaml_config.value("synthetic", "num_simulation_steps", value_type=int),
            gripper_close_command=close_command,
            lift_height_threshold=yaml_config.value("metrics", "lift_height_threshold", value_type=float),
            max_linear_velocity=yaml_config.value("limits", "max_linear_velocity", value_type=float),
            max_angular_velocity=yaml_config.value("limits", "max_angular_velocity", value_type=float),
            quality_report_path=yaml_config.value(
                "quality_report", "prepare", "quality_report", value_type=Path, script_or=True,
            ),
            sim_object_position=sim_object_position,
            sim_validate_require_lift=yaml_config.value("synthetic", "sim_validate_require_lift", value_type=bool),
            sim_validate_require_ik=yaml_config.value("synthetic", "sim_validate_require_ik", value_type=bool),
            sim_validate_min_contacts=yaml_config.value("synthetic", "sim_validate_min_contacts", value_type=float),
            sim_validate_fallback_analytical=yaml_config.value(
                "synthetic", "sim_validate_fallback_analytical", value_type=bool,
            ),
            table_xml=table_xml,
        )
        prepare_data_index(target_dir, output_index)
    else:
        if target_dir is None:
            msg = "paths.dataset_root or prepare.output_dir is required when prepare.mode is index"
            raise ValueError(msg)
        prepare_data_index(target_dir, output_index)


if __name__ == "__main__":
    main()
