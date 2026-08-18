"""Run fair physical comparisons between pose execution and an RL policy."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra

from grasping_ai.config import FLATTENED_YAML_CONFIG, SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.pipelines.grasp_experiments import run_grasp_experiments

GRASP_FILE = Path(str(FLATTENED_YAML_CONFIG.get("script.grasp_file", "data/processed/003_cracker_box.npz")))
OBJECT_ID = str(FLATTENED_YAML_CONFIG.get("script.object_id", "003_cracker_box"))
GRASP_INDEX = int(FLATTENED_YAML_CONFIG.get("script.grasp_index", 0))
ROBOT_XML_PATH = Path(str(FLATTENED_YAML_CONFIG.get("script.robot_xml", "deploy/robot.xml")))
YCB_ROOT = Path(str(FLATTENED_YAML_CONFIG.get("script.ycb_root", "data/processed/ycb_mjcf")))
TABLE_XML_PATH = Path(str(FLATTENED_YAML_CONFIG.get("script.table_xml", "deploy/table.xml")))
POLICY_CHECKPOINT_PATH = Path(
    str(FLATTENED_YAML_CONFIG.get("script.policy_checkpoint", "artifacts/checkpoints/rl_grasp_policy.pt")),
)
OUTPUT_PATH = Path(str(FLATTENED_YAML_CONFIG.get("script.output", "artifacts/reports/003_grasp_experiments.jsonl")))
EPISODES = int(FLATTENED_YAML_CONFIG.get("script.episodes", 20))
MAX_STEPS = int(FLATTENED_YAML_CONFIG.get("script.max_steps", 500))
BASELINE_SIMULATION_STEPS = int(FLATTENED_YAML_CONFIG.get("script.baseline_simulation_steps", 2000))
LIFT_THRESHOLD = float(FLATTENED_YAML_CONFIG.get("script.lift_threshold", 0.05))
PREGRASP_DISTANCE = float(FLATTENED_YAML_CONFIG.get("script.pregrasp_distance", 0.05))
DEVICE = str(FLATTENED_YAML_CONFIG.get("device", "cpu"))
SEED = int(FLATTENED_YAML_CONFIG.get("seed", 42))

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/run_grasp_experiments")
def main(cfg: DictConfig) -> None:
    """Run the configured three-way grasp comparison."""
    config = FlattenedYAMLConfig(cfg)
    run_grasp_experiments(
        grasp_file=config.value("grasp_file", value_type=Path, default=GRASP_FILE, script_or=True),
        object_id=str(config.value("object_id", value_type=object, default=OBJECT_ID, script_or=True)),
        grasp_index=config.value("grasp_index", value_type=int, default=GRASP_INDEX, script_or=True),
        robot_xml=config.value("robot_xml", value_type=Path, default=ROBOT_XML_PATH, script_or=True),
        ycb_root=config.value("ycb_root", value_type=Path, default=YCB_ROOT, script_or=True),
        table_xml=config.value("table_xml", value_type=Path, default=TABLE_XML_PATH, script_or=True),
        policy_checkpoint=config.value(
            "policy_checkpoint", value_type=Path, default=POLICY_CHECKPOINT_PATH, script_or=True,
        ),
        output=config.value("output", value_type=Path, default=OUTPUT_PATH, script_or=True),
        episodes=config.value("episodes", value_type=int, default=EPISODES, script_or=True),
        max_steps=config.value("max_steps", value_type=int, default=MAX_STEPS, script_or=True),
        baseline_simulation_steps=config.value(
            "baseline_simulation_steps",
            value_type=int,
            default=BASELINE_SIMULATION_STEPS,
            script_or=True,
        ),
        lift_threshold=config.value("lift_threshold", value_type=float, default=LIFT_THRESHOLD, script_or=True),
        pregrasp_distance=config.value(
            "pregrasp_distance", value_type=float, default=PREGRASP_DISTANCE, script_or=True,
        ),
        device=str(config.value("device", "device", value_type=object, default=DEVICE, script_or=True)),
        seed=config.value("seed", "seed", value_type=int, default=SEED, script_or=True),
    )


if __name__ == "__main__":
    main()
