from pathlib import Path

from grasping_ai.pipelines.train_flow import run_flow_training_pipeline


def train_flow_main(
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
) -> None:
    """Run the flow-matching grasp-generation training pipeline.

    Args:
        dataset_root: Root directory of the grasp-pose dataset.
        checkpoint_path: Destination path for the trained flow checkpoint.
        feature_dim: Conditioning feature dimension from the encoder.
        hidden_dim: Width of the flow-field hidden layers.
        num_layers: Number of hidden layers in the flow field.
        learning_rate: Learning rate for the Adam optimizer.
        num_epochs: Number of training epochs to perform.
        batch_size: Training batch size.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.
        seed: Optional random seed for reproducible training.
        experiment_log_dir: Optional path to write TensorBoard experiment events.
    """
    run_flow_training_pipeline(
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
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a flow-matching grasp-generation model"
    )
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
    args = parser.parse_args()
    train_flow_main(
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
    )
