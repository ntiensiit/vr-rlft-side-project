"""Audit synthetic grasp labels for consistency and coverage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from loguru import logger

from grasping_ai.config import FLATTENED_YAML_CONFIG, SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.pipelines.synthetic_audit import audit_synthetic_labels

DATASET_ROOT = Path(str(FLATTENED_YAML_CONFIG.get("script.dataset_root", "data/processed")))
FRICTION_COEFFICIENT = float(FLATTENED_YAML_CONFIG.get("script.friction_coefficient", 0.5))
COLLISION_CLEARANCE = float(FLATTENED_YAML_CONFIG.get("script.collision_clearance", 0.005))

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/audit_synthetic_labels")
def main(cfg: DictConfig) -> None:
    """Run the synthetic label audit and log each report entry."""
    yaml_config = FlattenedYAMLConfig(cfg)
    report = audit_synthetic_labels(
        dataset_root=yaml_config.value(
            "dataset_root", "paths", "dataset_root", value_type=Path, script_or=True, default=DATASET_ROOT,
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
        output_path=yaml_config.value("output", value_type=Path, script_or=True),
    )
    for entry in report:
        logger.info("{}", json.dumps(entry))


if __name__ == "__main__":
    main()
