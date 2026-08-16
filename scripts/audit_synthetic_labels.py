"""Audit synthetic grasp labels for consistency and coverage."""

from __future__ import annotations

from grasping_ai.config import SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.pipelines.synthetic_audit import audit_synthetic_labels

from pathlib import Path

import hydra
import json
from loguru import logger
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/audit_synthetic_labels")
def main(cfg: DictConfig) -> None:
    yaml_config = FlattenedYAMLConfig(cfg)
    report = audit_synthetic_labels(
        dataset_root=yaml_config.value("paths", "dataset_root", value_type=Path, required=True),
        friction_coefficient=yaml_config.value(
            "synthetic",
            "friction_coefficient",
            value_type=float,
            default=yaml_config.value("metrics", "friction_coefficient", value_type=float),
        ),
        collision_clearance=yaml_config.value(
            "synthetic",
            "collision_clearance",
            value_type=float,
            default=yaml_config.value("metrics", "collision_clearance", value_type=float),
        ),
        output_path=yaml_config.value("output", value_type=Path, script_or=True),
    )
    for entry in report:
        logger.info("{}", json.dumps(entry))


if __name__ == "__main__":
    main()
