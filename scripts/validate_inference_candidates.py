"""Physically validate raw model grasp candidates and write a validated NPZ archive."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
from loguru import logger

from grasping_ai.config import FLATTENED_YAML_CONFIG, SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.pipelines.validate_inference_candidates import validate_inference_candidates

CANDIDATE_FILE = FLATTENED_YAML_CONFIG.get("script.candidate_file")
OBSERVATION = FLATTENED_YAML_CONFIG.get("script.observation")
OUTPUT = FLATTENED_YAML_CONFIG.get("script.output")
OBJECT_ID = FLATTENED_YAML_CONFIG.get("script.object_id")
ROBOT_XML = Path(str(FLATTENED_YAML_CONFIG.get("script.robot_xml", "deploy/robot.xml")))
YCB_ROOT = Path(str(FLATTENED_YAML_CONFIG.get("script.ycb_root", "data/processed/ycb_mjcf")))
TABLE_XML = Path(str(FLATTENED_YAML_CONFIG.get("script.table_xml", "deploy/table.xml")))
OBJECT_POSITION = tuple(FLATTENED_YAML_CONFIG.get("script.object_position", [0.5, 0.0, 0.28]))
NUM_SIMULATION_STEPS = int(FLATTENED_YAML_CONFIG.get("script.num_simulation_steps", 2000))
GRIPPER_WIDTH = float(FLATTENED_YAML_CONFIG.get("script.gripper_width", 0.08))
LIFT_HEIGHT_THRESHOLD = float(FLATTENED_YAML_CONFIG.get("script.lift_height_threshold", 0.05))
LIFT_DISTANCE = float(FLATTENED_YAML_CONFIG.get("script.lift_distance", 0.1))
MAX_LINEAR_VELOCITY = float(FLATTENED_YAML_CONFIG.get("script.max_linear_velocity", 0.05))
MAX_ANGULAR_VELOCITY = float(FLATTENED_YAML_CONFIG.get("script.max_angular_velocity", 0.1))
MIN_CONTACTS = float(FLATTENED_YAML_CONFIG.get("script.min_contacts", 1.0))
REQUIRE_IK = bool(FLATTENED_YAML_CONFIG.get("script.require_ik", True))
REQUIRE_LIFT = bool(FLATTENED_YAML_CONFIG.get("script.require_lift", True))

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path=SCRIPTS_CONFIG_PATH,
    config_name="scripts/validate_inference_candidates",
)
def main(cfg: DictConfig) -> None:
    """Run physics validation for the exact candidates produced by inference."""
    config = FlattenedYAMLConfig(cfg)
    close_command = config.value(
        "close_command", "robot", "gripper", "close_command",
        value_type=list[float], script_or=True, default=[0.0],
    )
    if not isinstance(close_command, list):
        raise TypeError("robot.gripper.close_command must be a list")
    table_xml = config.value("table_xml", value_type=Path, script_or=True, default=TABLE_XML)
    sample = validate_inference_candidates(
        candidate_path=config.value(
            "candidate_file", value_type=Path, script_or=True, default=CANDIDATE_FILE, required=True,
        ),
        observation_path=config.value(
            "observation", value_type=Path, script_or=True, default=OBSERVATION, required=True,
        ),
        output_path=config.value(
            "output", value_type=Path, script_or=True, default=OUTPUT, required=True,
        ),
        object_id=str(
            config.value("object_id", value_type=object, script_or=True, default=OBJECT_ID, required=True),
        ),
        ycb_root=config.value(
            "ycb_root", value_type=Path, script_or=True, default=YCB_ROOT, required=True,
        ),
        robot_xml=config.value(
            "robot_xml", value_type=Path, script_or=True, default=ROBOT_XML, required=True,
        ),
        table_xml=table_xml,
        object_position=np.asarray(
            config.value(
                "object_position", value_type=list[float], script_or=True,
                default=list(OBJECT_POSITION), required=True,
            ),
            dtype=np.float64,
        ),
        num_simulation_steps=config.value(
            "num_simulation_steps", value_type=int, script_or=True,
            default=NUM_SIMULATION_STEPS, required=True,
        ),
        gripper_close_command=np.asarray(close_command, dtype=np.float64),
        gripper_width=config.value(
            "gripper_width", value_type=float, script_or=True, default=GRIPPER_WIDTH, required=True,
        ),
        lift_height_threshold=config.value(
            "lift_height_threshold", value_type=float, script_or=True,
            default=LIFT_HEIGHT_THRESHOLD, required=True,
        ),
        lift_distance=config.value(
            "lift_distance", value_type=float, script_or=True, default=LIFT_DISTANCE, required=True,
        ),
        max_linear_velocity=config.value(
            "max_linear_velocity", value_type=float, script_or=True,
            default=MAX_LINEAR_VELOCITY, required=True,
        ),
        max_angular_velocity=config.value(
            "max_angular_velocity", value_type=float, script_or=True,
            default=MAX_ANGULAR_VELOCITY, required=True,
        ),
        min_contacts=config.value(
            "min_contacts", value_type=float, script_or=True, default=MIN_CONTACTS, required=True,
        ),
        require_ik=config.value(
            "require_ik", value_type=bool, script_or=True, default=REQUIRE_IK, required=True,
        ),
        require_lift=config.value(
            "require_lift", value_type=bool, script_or=True, default=REQUIRE_LIFT, required=True,
        ),
    )
    logger.info(
        "Validated {}/{} inference candidates into {}",
        int(np.count_nonzero(sample["sim_validated"])),
        len(sample["grasp_poses"]),
        config.value("output", value_type=Path, script_or=True, required=True),
    )
    rejection_counts = Counter(str(reason) for reason in sample["validation_failure_reasons"] if str(reason))
    if rejection_counts:
        logger.info(
            "Rejection summary: {}",
            "; ".join(f"{reason}: {count}" for reason, count in sorted(rejection_counts.items())),
        )


if __name__ == "__main__":
    main()
