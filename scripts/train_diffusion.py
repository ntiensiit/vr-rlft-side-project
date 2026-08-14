from pathlib import Path

from grasping_ai.config.yaml_loader import (
    config_get,
    config_path,
    load_project_yaml_config,
    parse_config_dir_from_argv,
    parse_config_overrides_from_argv,
)
from grasping_ai.pipelines.train_diffusion import run_diffusion_training_pipeline

if __name__ == "__main__":
    import argparse

    config_dir = parse_config_dir_from_argv()
    overrides = parse_config_overrides_from_argv()
    if not any(item.startswith("training=") for item in overrides):
        overrides = ["training=diffusion", *overrides]
    cfg = load_project_yaml_config(config_dir, overrides=overrides)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-dir", type=Path, default=config_dir)
    parser = argparse.ArgumentParser(
        description="Train a diffusion grasp-generation model",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=config_path(cfg, "paths", "dataset_root"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config_path(cfg, "diffusion", "checkpoint"),
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=int(config_get(cfg, "architecture", "feature_dim")),
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=int(config_get(cfg, "architecture", "hidden_dim")),
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=int(config_get(cfg, "architecture", "num_layers")),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(config_get(cfg, "supervised", "learning_rate")),
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=int(config_get(cfg, "supervised", "num_epochs")),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(config_get(cfg, "supervised", "batch_size")),
    )
    parser.add_argument("--device", type=str, default=str(config_get(cfg, "device")))
    parser.add_argument("--seed", type=int, default=int(config_get(cfg, "seed")))
    parser.add_argument(
        "--experiment-log-dir",
        type=Path,
        default=config_path(cfg, "diffusion", "tensorboard"),
    )
    parser.add_argument("--pretrained-encoder", type=Path, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint to resume model and optimizer state from",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply SO(3)/translation jitter during supervised pair construction",
    )
    parser.add_argument(
        "--min-grasp-score",
        type=float,
        default=float(config_get(cfg, "supervised", "min_grasp_score", default=0.0)),
    )
    parser.add_argument(
        "--score-repeat-factor",
        type=int,
        default=int(config_get(cfg, "supervised", "score_repeat_factor", default=0)),
    )
    parser.add_argument(
        "--score-repeat-power",
        type=float,
        default=float(config_get(cfg, "supervised", "score_repeat_power", default=1.0)),
    )
    args = parser.parse_args()
    if args.dataset_root is None:
        parser.error(
            "--dataset-root is required (set in configs/data/default.yaml paths.dataset_root or pass explicitly)"
        )
    if args.checkpoint is None:
        parser.error(
            "--checkpoint is required (set in configs/model/diffusion.yaml diffusion.checkpoint or pass explicitly)"
        )
    from grasping_ai.utils.logging_utils import init_mlflow, setup_logging
    setup_logging(module_name="train_diffusion")
    use_mlflow = init_mlflow(cfg)

    if use_mlflow:
        import mlflow
        with mlflow.start_run(run_name="diffusion_training"):
            run_diffusion_training_pipeline(
                dataset_root=args.dataset_root,
                checkpoint_path=args.checkpoint,
                feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                device=args.device,
                seed=args.seed,
                experiment_log_dir=args.experiment_log_dir,
                pretrained_encoder_path=args.pretrained_encoder,
                resume_checkpoint_path=args.resume,
                augment=args.augment,
                min_grasp_score=args.min_grasp_score,
                score_repeat_factor=args.score_repeat_factor,
                score_repeat_power=args.score_repeat_power,
            )
    else:
        run_diffusion_training_pipeline(
            dataset_root=args.dataset_root,
            checkpoint_path=args.checkpoint,
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            experiment_log_dir=args.experiment_log_dir,
            pretrained_encoder_path=args.pretrained_encoder,
            resume_checkpoint_path=args.resume,
            augment=args.augment,
            min_grasp_score=args.min_grasp_score,
            score_repeat_factor=args.score_repeat_factor,
            score_repeat_power=args.score_repeat_power,
        )
