"""Train RL grasp policies from the command line."""

from __future__ import annotations

from grasping_ai.config import SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig

from grasping_ai.pipelines.train_rl import run_rl_training_pipeline

from pathlib import Path

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/train_rl")
def main(cfg: DictConfig) -> None:
    yaml_config = FlattenedYAMLConfig(cfg)
    run_rl_training_pipeline(
        robot_xml_path=yaml_config.value("robot", "description", value_type=Path, required=True),
        ycb_root=yaml_config.value("paths", "ycb_mjcf", value_type=Path, required=True),
        object_ids=yaml_config.value("objects", "ids", value_type=list[str]),
        policy_checkpoint_path=yaml_config.value("rl", "checkpoint", value_type=Path, required=True),
        observation_dim=yaml_config.value("rl", "observation_dim", value_type=int),
        action_dim=yaml_config.value("rl", "action_dim", value_type=int),
        hidden_dim=yaml_config.value("rl", "hidden_dim", value_type=int),
        learning_rate=yaml_config.value("rl", "learning_rate", value_type=float),
        num_updates=yaml_config.value("rl", "num_updates", value_type=int),
        gamma=yaml_config.value("rl", "gamma", value_type=float),
        device=str(yaml_config.get("device")),
        seed=yaml_config.value("seed", value_type=int),
        experiment_log_dir=yaml_config.value("rl", "tensorboard", value_type=Path),
        n_steps=yaml_config.value("rl", "n_steps", value_type=int),
        batch_size=yaml_config.value("rl", "batch_size", value_type=int),
        n_epochs=yaml_config.value("rl", "n_epochs", value_type=int),
        policy_num_layers=yaml_config.value("rl", "policy_num_layers", value_type=int),
    )

if __name__ == "__main__":
    main()
