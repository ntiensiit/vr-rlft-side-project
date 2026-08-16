"""Prepare processed grasp datasets from raw YCB assets."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import open3d as _open3d  # noqa: F401
import hydra
import numpy as np

from grasping_ai.config import FLATTENED_YAML_CONFIG, SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.pipelines.prepare_synthetic_data import generate_synthetic_dataset, prepare_data_index

DATASET_ROOT = Path(str(FLATTENED_YAML_CONFIG.get("script.dataset_root", "data/processed")))
OBJECT_IDS = tuple(FLATTENED_YAML_CONFIG.get("script.object_ids", []))
ROBOT_XML_PATH = Path(str(FLATTENED_YAML_CONFIG.get("script.robot_xml", "deploy/robot.xml")))
YCB_ROOT = Path(str(FLATTENED_YAML_CONFIG.get("script.ycb_root", "data/raw/ycb")))
ALLOW_RELAXED = bool(FLATTENED_YAML_CONFIG.get("script.allow_relaxed", True))
CANDIDATE_MULTIPLIER = int(FLATTENED_YAML_CONFIG.get("script.candidate_multiplier", 3))
COLLISION_CLEARANCE = float(FLATTENED_YAML_CONFIG.get("script.collision_clearance", 0.005))
FRICTION_COEFFICIENT = float(FLATTENED_YAML_CONFIG.get("script.friction_coefficient", 0.5))
GRIPPER_WIDTH = float(FLATTENED_YAML_CONFIG.get("script.gripper_width", 0.08))
LIFT_HEIGHT_THRESHOLD = float(FLATTENED_YAML_CONFIG.get("script.lift_height_threshold", 0.05))
MAX_ANGULAR_VELOCITY = float(FLATTENED_YAML_CONFIG.get("script.max_angular_velocity", 0.1))
MAX_LINEAR_VELOCITY = float(FLATTENED_YAML_CONFIG.get("script.max_linear_velocity", 0.05))
MIN_GRASP_ROTATION = float(FLATTENED_YAML_CONFIG.get("script.min_grasp_rotation", 0.2))
MIN_GRASP_TRANSLATION = float(FLATTENED_YAML_CONFIG.get("script.min_grasp_translation", 0.01))
MIN_QUALITY_SCORE = float(FLATTENED_YAML_CONFIG.get("script.min_quality_score", 0.0))
NEIGHBORHOOD_SIZE = int(FLATTENED_YAML_CONFIG.get("script.neighborhood_size", 30))
NUM_GRASPS = int(FLATTENED_YAML_CONFIG.get("script.num_grasps", 16))
NUM_SAMPLES = int(FLATTENED_YAML_CONFIG.get("script.num_samples", 512))
NUM_SIMULATION_STEPS = int(FLATTENED_YAML_CONFIG.get("script.num_simulation_steps", 500))
OVERSAMPLE_EXTRA = int(FLATTENED_YAML_CONFIG.get("script.oversample_extra", 256))
OVERSAMPLE_FACTOR = int(FLATTENED_YAML_CONFIG.get("script.oversample_factor", 2))
RELAXED_ANTIPODAL_DOT = float(FLATTENED_YAML_CONFIG.get("script.relaxed_antipodal_dot", 0.3))
SEARCH_MULTIPLIER = int(FLATTENED_YAML_CONFIG.get("script.search_multiplier", 50))
SEED = int(FLATTENED_YAML_CONFIG.get("script.seed", 42))
SIM_OBJECT_POSITION = tuple(FLATTENED_YAML_CONFIG.get_path("synthetic", "sim_object_position"))
SIM_VALIDATE = bool(FLATTENED_YAML_CONFIG.get("script.sim_validate", False))
SIM_VALIDATE_FALLBACK_ANALYTICAL = bool(
    FLATTENED_YAML_CONFIG.get("script.sim_validate_fallback_analytical", True),
)
SIM_VALIDATE_MIN_CONTACTS = float(FLATTENED_YAML_CONFIG.get("script.sim_validate_min_contacts", 1.0))
SIM_VALIDATE_REQUIRE_IK = bool(FLATTENED_YAML_CONFIG.get("script.sim_validate_require_ik", True))
SIM_VALIDATE_REQUIRE_LIFT = bool(FLATTENED_YAML_CONFIG.get("script.sim_validate_require_lift", False))
STRICT_ALIGNMENT_DOT = float(FLATTENED_YAML_CONFIG.get("script.strict_alignment_dot", 0.5))
STRICT_ANTIPODAL_DOT = float(FLATTENED_YAML_CONFIG.get("script.strict_antipodal_dot", 0.5))
VOXEL_SIZE = float(FLATTENED_YAML_CONFIG.get("script.voxel_size", 1.0e-5))

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/prepare_data")
def main(cfg: DictConfig) -> None:
    """Prepare processed grasp datasets or rebuild the dataset index."""
    yaml_config = FlattenedYAMLConfig(cfg)
    mode = str(yaml_config.value("mode", "prepare", "mode", value_type=object, script_or=True, default="index"))
    dataset_root = yaml_config.value(
        "dataset_root", "paths", "dataset_root", value_type=Path, script_or=True, default=DATASET_ROOT,
    )
    output_dir = yaml_config.value("output_dir", "prepare", "output_dir", value_type=Path, script_or=True)
    output_index = yaml_config.value(
        "output_index", "paths", "output_index", value_type=Path, script_or=True, required=True,
    )
    target_dir = output_dir if output_dir is not None else dataset_root

    if mode == "synthetic":
        if dataset_root is None and output_dir is None:
            msg = "prepare.output_dir or paths.dataset_root is required when prepare.mode is synthetic"
            raise ValueError(msg)
        ycb_root = yaml_config.value("ycb_root", "paths", "ycb_root", value_type=Path, script_or=True, default=YCB_ROOT)
        if target_dir is None:
            msg = "prepare.output_dir or paths.dataset_root is required when prepare.mode is synthetic"
            raise ValueError(msg)

        gripper_close = yaml_config.get_path("robot", "gripper", "close_command")
        close_command = np.asarray(gripper_close, dtype=np.float64) if isinstance(gripper_close, list) else None
        sim_position = yaml_config.value(
            "sim_object_position",
            "synthetic",
            "sim_object_position",
            value_type=list[float],
            script_or=True,
            default=list(SIM_OBJECT_POSITION),
        )
        sim_object_position = np.asarray(sim_position, dtype=np.float64)
        sim_table = yaml_config.get_path("synthetic", "sim_table_xml")
        table_xml = Path(str(sim_table)) if isinstance(sim_table, str) else None
        if table_xml is not None and not table_xml.is_absolute():
            table_xml = Path(__file__).resolve().parents[1] / table_xml

        generate_synthetic_dataset(
            ycb_root=ycb_root,
            output_dir=target_dir,
            num_samples=yaml_config.value(
                "num_samples", "synthetic", "num_samples", value_type=int, script_or=True, default=NUM_SAMPLES,
            ),
            num_grasps=yaml_config.value(
                "num_grasps", "synthetic", "num_grasps", value_type=int, script_or=True, default=NUM_GRASPS,
            ),
            gripper_width=yaml_config.value(
                "gripper_width", "synthetic", "gripper_width", value_type=float, script_or=True, default=GRIPPER_WIDTH,
            ),
            seed=yaml_config.value("seed", "synthetic", "seed", value_type=int, script_or=True, default=SEED),
            required_objects=yaml_config.value(
                "required_objects", "objects", "ids", value_type=list[str], script_or=True, default=list(OBJECT_IDS),
            ),
            oversample_factor=yaml_config.value(
                "oversample_factor",
                "synthetic",
                "oversample_factor",
                value_type=int,
                script_or=True,
                default=OVERSAMPLE_FACTOR,
            ),
            oversample_extra=yaml_config.value(
                "oversample_extra",
                "synthetic",
                "oversample_extra",
                value_type=int,
                script_or=True,
                default=OVERSAMPLE_EXTRA,
            ),
            neighborhood_size=yaml_config.value(
                "neighborhood_size",
                "synthetic",
                "neighborhood_size",
                value_type=int,
                script_or=True,
                default=NEIGHBORHOOD_SIZE,
            ),
            voxel_size=yaml_config.value(
                "voxel_size", "synthetic", "voxel_size", value_type=float, script_or=True, default=VOXEL_SIZE,
            ),
            strict_antipodal_dot=yaml_config.value(
                "strict_antipodal_dot",
                "synthetic",
                "strict_antipodal_dot",
                value_type=float,
                script_or=True,
                default=STRICT_ANTIPODAL_DOT,
            ),
            strict_alignment_dot=yaml_config.value(
                "strict_alignment_dot",
                "synthetic",
                "strict_alignment_dot",
                value_type=float,
                script_or=True,
                default=STRICT_ALIGNMENT_DOT,
            ),
            relaxed_antipodal_dot=yaml_config.value(
                "relaxed_antipodal_dot",
                "synthetic",
                "relaxed_antipodal_dot",
                value_type=float,
                script_or=True,
                default=RELAXED_ANTIPODAL_DOT,
            ),
            allow_relaxed=yaml_config.value(
                "allow_relaxed", "synthetic", "allow_relaxed", value_type=bool, script_or=True, default=ALLOW_RELAXED,
            ),
            search_multiplier=yaml_config.value(
                "search_multiplier",
                "synthetic",
                "search_multiplier",
                value_type=int,
                script_or=True,
                default=SEARCH_MULTIPLIER,
            ),
            candidate_multiplier=yaml_config.value(
                "candidate_multiplier",
                "synthetic",
                "candidate_multiplier",
                value_type=int,
                script_or=True,
                default=CANDIDATE_MULTIPLIER,
            ),
            min_grasp_translation=yaml_config.value(
                "min_grasp_translation",
                "synthetic",
                "min_grasp_translation",
                value_type=float,
                script_or=True,
                default=MIN_GRASP_TRANSLATION,
            ),
            min_grasp_rotation=yaml_config.value(
                "min_grasp_rotation",
                "synthetic",
                "min_grasp_rotation",
                value_type=float,
                script_or=True,
                default=MIN_GRASP_ROTATION,
            ),
            min_quality_score=yaml_config.value(
                "min_quality_score",
                "synthetic",
                "min_quality_score",
                value_type=float,
                script_or=True,
                default=MIN_QUALITY_SCORE,
            ),
            friction_coefficient=yaml_config.value(
                "friction_coefficient",
                "synthetic",
                "friction_coefficient",
                value_type=float,
                script_or=True,
                default=FRICTION_COEFFICIENT,
            ),
            collision_clearance=yaml_config.value(
                "collision_clearance",
                "synthetic",
                "collision_clearance",
                value_type=float,
                script_or=True,
                default=COLLISION_CLEARANCE,
            ),
            sim_validate=yaml_config.value(
                "sim_validate", "synthetic", "sim_validate", value_type=bool, script_or=True, default=SIM_VALIDATE,
            ),
            mjcf_root=yaml_config.value("mjcf_root", "paths", "ycb_mjcf", value_type=Path, script_or=True),
            robot_xml=yaml_config.value(
                "robot_xml", "robot", "description", value_type=Path, script_or=True, default=ROBOT_XML_PATH,
            ),
            num_simulation_steps=yaml_config.value(
                "num_simulation_steps",
                "synthetic",
                "num_simulation_steps",
                value_type=int,
                script_or=True,
                default=NUM_SIMULATION_STEPS,
            ),
            gripper_close_command=close_command,
            lift_height_threshold=yaml_config.value(
                "lift_height_threshold",
                "metrics",
                "lift_height_threshold",
                value_type=float,
                script_or=True,
                default=LIFT_HEIGHT_THRESHOLD,
            ),
            max_linear_velocity=yaml_config.value(
                "max_linear_velocity",
                "limits",
                "max_linear_velocity",
                value_type=float,
                script_or=True,
                default=MAX_LINEAR_VELOCITY,
            ),
            max_angular_velocity=yaml_config.value(
                "max_angular_velocity",
                "limits",
                "max_angular_velocity",
                value_type=float,
                script_or=True,
                default=MAX_ANGULAR_VELOCITY,
            ),
            quality_report_path=yaml_config.value(
                "quality_report", "prepare", "quality_report", value_type=Path, script_or=True,
            ),
            sim_object_position=sim_object_position,
            sim_validate_require_lift=yaml_config.value(
                "sim_validate_require_lift",
                "synthetic",
                "sim_validate_require_lift",
                value_type=bool,
                script_or=True,
                default=SIM_VALIDATE_REQUIRE_LIFT,
            ),
            sim_validate_require_ik=yaml_config.value(
                "sim_validate_require_ik",
                "synthetic",
                "sim_validate_require_ik",
                value_type=bool,
                script_or=True,
                default=SIM_VALIDATE_REQUIRE_IK,
            ),
            sim_validate_min_contacts=yaml_config.value(
                "sim_validate_min_contacts",
                "synthetic",
                "sim_validate_min_contacts",
                value_type=float,
                script_or=True,
                default=SIM_VALIDATE_MIN_CONTACTS,
            ),
            sim_validate_fallback_analytical=yaml_config.value(
                "sim_validate_fallback_analytical",
                "synthetic",
                "sim_validate_fallback_analytical",
                value_type=bool,
                script_or=True,
                default=SIM_VALIDATE_FALLBACK_ANALYTICAL,
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
