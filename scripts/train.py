from pathlib import Path

from grasping_ai.pipelines.train import run_training_pipeline


def train_main(
    dataset_root: Path,
    checkpoint_path: Path,
    feature_dim: int,
    hidden_dim: int,
    num_layers: int,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    device: str,
    seed: int | None = None,
    experiment_log_dir: Path | None = None,
    pretrained_encoder_path: Path | None = None,
    resume_checkpoint_path: Path | None = None,
    augment: bool = False,
) -> None:
    """Run the supervised training pipeline and persist a model checkpoint."""
    run_training_pipeline(
        dataset_root=dataset_root,
        checkpoint_path=checkpoint_path,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        seed=seed,
        experiment_log_dir=experiment_log_dir,
        pretrained_encoder_path=pretrained_encoder_path,
        resume_checkpoint_path=resume_checkpoint_path,
        augment=augment,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a grasp-generation model")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-dim", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--experiment-log-dir", type=Path, default=None)
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
    args = parser.parse_args()
    train_main(
        args.dataset_root,
        args.checkpoint,
        args.feature_dim,
        args.hidden_dim,
        args.num_layers,
        args.learning_rate,
        args.num_epochs,
        args.batch_size,
        args.device,
        args.seed,
        args.experiment_log_dir,
        args.pretrained_encoder,
        args.resume,
        args.augment,
    )
