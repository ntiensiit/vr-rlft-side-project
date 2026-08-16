"""Train RL grasp policies from the command line."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra

from grasping_ai.config import FLATTENED_YAML_CONFIG, SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.pipelines.train_rl import run_rl_training_pipeline

ROBOT_XML_PATH = Path(str(FLATTENED_YAML_CONFIG.get("script.robot_xml", "deploy/robot.xml")))
YCB_ROOT = Path(str(FLATTENED_YAML_CONFIG.get("script.ycb_root", "data/processed/ycb_mjcf")))
OBJECT_IDS = tuple(FLATTENED_YAML_CONFIG.get("script.object_ids", []))
POLICY_CHECKPOINT_PATH = Path(
    str(FLATTENED_YAML_CONFIG.get("script.policy_checkpoint", "artifacts/checkpoints/rl_grasp_policy.pt")),
)
OBSERVATION_DIM = int(FLATTENED_YAML_CONFIG.get("script.observation_dim", 31))
ACTION_DIM = int(FLATTENED_YAML_CONFIG.get("script.action_dim", 8))
HIDDEN_DIM = int(FLATTENED_YAML_CONFIG.get("script.hidden_dim", 32))
LEARNING_RATE = float(FLATTENED_YAML_CONFIG.get("script.learning_rate", 0.0003))
NUM_UPDATES = int(FLATTENED_YAML_CONFIG.get("script.num_updates", 10))
GAMMA = float(FLATTENED_YAML_CONFIG.get("script.gamma", 0.99))
DEVICE = str(FLATTENED_YAML_CONFIG.get("script.device", "cpu"))
SEED = int(FLATTENED_YAML_CONFIG.get("script.seed", 42))
N_STEPS = int(FLATTENED_YAML_CONFIG.get("script.n_steps", 64))
BATCH_SIZE = int(FLATTENED_YAML_CONFIG.get("script.batch_size", 64))
N_EPOCHS = int(FLATTENED_YAML_CONFIG.get("script.n_epochs", 1))
POLICY_NUM_LAYERS = int(FLATTENED_YAML_CONFIG.get("script.policy_num_layers", 2))

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/train_rl")
def main(cfg: DictConfig) -> None:
    """Train the RL grasp policy from the hydra configuration."""
    yaml_config = FlattenedYAMLConfig(cfg)
    configured_objects = yaml_config.value(
        "object_ids", "objects", "ids", value_type=list[str], script_or=True, default=list(OBJECT_IDS),
    )
    run_rl_training_pipeline(
        robot_xml_path=yaml_config.value(
            "robot_xml", "robot", "description", value_type=Path, script_or=True, default=ROBOT_XML_PATH,
        ),
        ycb_root=yaml_config.value(
            "ycb_root", "paths", "ycb_mjcf", value_type=Path, script_or=True, default=YCB_ROOT,
        ),
        # The environment has one tracked object; use the first configured
        # object while the library API still rejects ambiguous direct calls.
        object_ids=configured_objects[:1],
        policy_checkpoint_path=yaml_config.value(
            "policy_checkpoint", "rl", "checkpoint", value_type=Path, script_or=True, default=POLICY_CHECKPOINT_PATH,
        ),
        observation_dim=yaml_config.value(
            "observation_dim", "rl", "observation_dim", value_type=int, script_or=True, default=OBSERVATION_DIM,
        ),
        action_dim=yaml_config.value(
            "action_dim", "rl", "action_dim", value_type=int, script_or=True, default=ACTION_DIM,
        ),
        hidden_dim=yaml_config.value(
            "hidden_dim", "rl", "hidden_dim", value_type=int, script_or=True, default=HIDDEN_DIM,
        ),
        learning_rate=yaml_config.value(
            "learning_rate", "rl", "learning_rate", value_type=float, script_or=True, default=LEARNING_RATE,
        ),
        num_updates=yaml_config.value(
            "num_updates", "rl", "num_updates", value_type=int, script_or=True, default=NUM_UPDATES,
        ),
        gamma=yaml_config.value("gamma", "rl", "gamma", value_type=float, script_or=True, default=GAMMA),
        device=str(yaml_config.value("device", "device", value_type=object, script_or=True, default=DEVICE)),
        seed=yaml_config.value("seed", "seed", value_type=int, script_or=True, default=SEED),
        experiment_log_dir=yaml_config.value(
            "experiment_log_dir", "rl", "tensorboard", value_type=Path, script_or=True,
        ),
        n_steps=yaml_config.value("n_steps", "rl", "n_steps", value_type=int, script_or=True, default=N_STEPS),
        batch_size=yaml_config.value(
            "batch_size", "rl", "batch_size", value_type=int, script_or=True, default=BATCH_SIZE,
        ),
        n_epochs=yaml_config.value("n_epochs", "rl", "n_epochs", value_type=int, script_or=True, default=N_EPOCHS),
        policy_num_layers=yaml_config.value(
            "policy_num_layers",
            "rl",
            "policy_num_layers",
            value_type=int,
            script_or=True,
            default=POLICY_NUM_LAYERS,
        ),
    )

if __name__ == "__main__":
    main()
