from __future__ import annotations

from pathlib import Path

from grasping_ai.config.yaml_loader import (
    config_get,
    config_path,
    config_str_list,
    load_project_yaml_config,
    parse_config_dir_from_argv,
)
from grasping_ai.pipelines.train_rl import run_rl_training_pipeline

if __name__ == "__main__":
    import argparse

    config_dir = parse_config_dir_from_argv()
    cfg = load_project_yaml_config(config_dir)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-dir", type=Path, default=config_dir)
    parser = argparse.ArgumentParser(
        description="Train an RL grasping policy",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--robot-xml",
        type=Path,
        default=config_path(cfg, "robot", "description"),
    )
    parser.add_argument(
        "--ycb-root",
        type=Path,
        default=config_path(cfg, "paths", "ycb_mjcf"),
    )
    parser.add_argument(
        "--object-ids",
        type=str,
        nargs="+",
        default=config_str_list(cfg, "objects", "ids"),
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=Path,
        default=config_path(cfg, "rl", "checkpoint"),
    )
    parser.add_argument(
        "--observation-dim",
        type=int,
        default=int(config_get(cfg, "rl", "observation_dim")),
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=int(config_get(cfg, "rl", "action_dim")),
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=int(config_get(cfg, "rl", "hidden_dim")),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(config_get(cfg, "rl", "learning_rate")),
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=int(config_get(cfg, "rl", "num_updates")),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=float(config_get(cfg, "rl", "gamma")),
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=int(config_get(cfg, "rl", "n_steps")),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(config_get(cfg, "rl", "batch_size")),
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=int(config_get(cfg, "rl", "n_epochs")),
    )
    parser.add_argument(
        "--policy-num-layers",
        type=int,
        default=int(config_get(cfg, "rl", "policy_num_layers")),
    )
    parser.add_argument("--device", type=str, default=str(config_get(cfg, "device")))
    parser.add_argument("--seed", type=int, default=int(config_get(cfg, "seed")))
    parser.add_argument(
        "--experiment-log-dir",
        type=Path,
        default=config_path(cfg, "rl", "tensorboard"),
    )
    args = parser.parse_args()
    if args.robot_xml is None:
        parser.error(
            "--robot-xml is required (set in configs/gripper/franka_emika_panda.yaml "
            "robot.description or pass explicitly)"
        )
    if args.ycb_root is None:
        parser.error("--ycb-root is required (set in configs/base.yaml paths.ycb_mjcf or pass explicitly)")
    if args.object_ids is None:
        parser.error("--object-ids is required (set in configs/object/default.yaml objects.ids or pass explicitly)")
    if args.policy_checkpoint is None:
        parser.error(
            "--policy-checkpoint is required (set in configs/rl/default.yaml rl.checkpoint or pass explicitly)"
        )
    run_rl_training_pipeline(
        robot_xml_path=args.robot_xml,
        ycb_root=args.ycb_root,
        object_ids=args.object_ids,
        policy_checkpoint_path=args.policy_checkpoint,
        observation_dim=args.observation_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        num_updates=args.num_updates,
        gamma=args.gamma,
        device=args.device,
        seed=args.seed,
        experiment_log_dir=args.experiment_log_dir,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        policy_num_layers=args.policy_num_layers,
    )
