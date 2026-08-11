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
) -> None:
    """Run the supervised training pipeline and persist a model checkpoint.

    Args:
        dataset_root: Root directory of the grasp-pose dataset.
        checkpoint_path: Destination path for the trained model checkpoint.
        feature_dim: Conditioning feature dimension used by the encoder.
        hidden_dim: Hidden width of the grasp-generation model.
        num_layers: Number of layers in the grasp-generation model.
        learning_rate: Learning rate for the optimizer.
        num_epochs: Number of training epochs to perform.
        batch_size: Training batch size.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.
    """
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
    )
