"""Train RL grasp policies from the command line."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from grasping_ai.config.config import SCRIPTS_CONFIG_PATH, config_get, config_value
from grasping_ai.pipelines.train_rl import run_rl_training_pipeline


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/train_rl")
def main(cfg: DictConfig) -> None:
    run_rl_training_pipeline(
        robot_xml_path=config_value(cfg, "robot", "description", value_type=Path, required=True),
        ycb_root=config_value(cfg, "paths", "ycb_mjcf", value_type=Path, required=True),
        object_ids=config_value(cfg, "objects", "ids", value_type=list[str]),
        policy_checkpoint_path=config_value(cfg, "rl", "checkpoint", value_type=Path, required=True),
        observation_dim=config_value(cfg, "rl", "observation_dim", value_type=int),
        action_dim=config_value(cfg, "rl", "action_dim", value_type=int),
        hidden_dim=config_value(cfg, "rl", "hidden_dim", value_type=int),
        learning_rate=config_value(cfg, "rl", "learning_rate", value_type=float),
        num_updates=config_value(cfg, "rl", "num_updates", value_type=int),
        gamma=config_value(cfg, "rl", "gamma", value_type=float),
        device=str(config_get(cfg, "device")),
        seed=config_value(cfg, "seed", value_type=int),
        experiment_log_dir=config_value(cfg, "rl", "tensorboard", value_type=Path),
        n_steps=config_value(cfg, "rl", "n_steps", value_type=int),
        batch_size=config_value(cfg, "rl", "batch_size", value_type=int),
        n_epochs=config_value(cfg, "rl", "n_epochs", value_type=int),
        policy_num_layers=config_value(cfg, "rl", "policy_num_layers", value_type=int),
    )


if __name__ == "__main__":
    main()
