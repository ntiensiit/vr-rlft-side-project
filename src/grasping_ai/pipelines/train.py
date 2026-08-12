from collections.abc import Iterator
from pathlib import Path
from typing import cast

import torch


def run_training_pipeline(
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
    """Run the end-to-end supervised training pipeline for grasp generation.

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
        seed: Optional random seed for reproducible training.
        experiment_log_dir: Optional path to write TensorBoard experiment events.
        pretrained_encoder_path: Optional checkpoint whose encoder weights warm-start
            the grasp-generation model before supervised training.
    """
    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be a pathlib.Path instance")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    if seed is not None:
        torch.manual_seed(seed)

    # Build components
    components = build_supervised_training_components(
        feature_dim, hidden_dim, num_layers, learning_rate, device
    )
    model = components["model"]
    optimizer = components["optimizer"]

    if pretrained_encoder_path is not None:
        encoder_state = load_pretrained_encoder(pretrained_encoder_path, device)
        from grasping_ai.models.diffusion import GraspGeneratorModel

        encoder_module = cast(torch.nn.Module, cast(GraspGeneratorModel, model).encoder)
        encoder_module.load_state_dict(encoder_state, strict=False)

    from grasping_ai.data.training_pairs import build_supervised_training_pairs

    try:
        training_pairs = build_supervised_training_pairs(dataset_root)
    except Exception as e:
        raise ValueError(f"Failed to build supervised training pairs: {e}") from e

    # Construct the training step and training loop
    from grasping_ai.models.diffusion import GraspGeneratorModel
    from grasping_ai.training.losses import (
        build_diffusion_score_loss,
        build_grasp_pose_regression_loss,
    )
    from grasping_ai.training.trainer import build_training_step, run_training_loop

    model_generator = cast(GraspGeneratorModel, model)
    loss_fn = build_diffusion_score_loss()
    regression_loss_fn = build_grasp_pose_regression_loss("mse")
    training_step = build_training_step(
        model_generator, loss_fn, cast(torch.optim.Optimizer, optimizer), device, seed=seed
    )

    # Iterable loader helper
    class TrainingDataloader:
        def __init__(
            self,
            pairs: list[tuple[torch.Tensor, torch.Tensor]],
            b_size: int,
            dev: str,
            s: int | None,
        ) -> None:
            self.pairs = pairs
            self.b_size = b_size
            self.dev = dev
            self.seed = s

        def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            num_samples = len(self.pairs)
            indices = list(range(num_samples))
            import random
            local_random = random.Random(self.seed if self.seed is not None else 42)
            local_random.shuffle(indices)

            for i in range(0, num_samples, self.b_size):
                batch_indices = indices[i : i + self.b_size]
                pcs = torch.stack([self.pairs[idx][0] for idx in batch_indices]).to(self.dev)
                targets = torch.stack([self.pairs[idx][1] for idx in batch_indices]).to(self.dev)

                # Compute conditioning features online
                from grasping_ai.models.equivariant_encoder import (
                    encode_point_cloud,
                    pool_object_features,
                )
                features = encode_point_cloud(model_generator.encoder, pcs)
                cond = pool_object_features(features)

                yield cond, targets

    dataloader = TrainingDataloader(training_pairs, batch_size, device, seed)

    with torch.no_grad():
        first_batch = next(iter(dataloader))
        baseline_regression_loss = regression_loss_fn(first_batch[1], first_batch[1])

    metadata = {
        "dataset_root": str(dataset_root),
        "checkpoint_path": str(checkpoint_path),
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "device": device,
        "baseline_regression_loss": float(baseline_regression_loss.item()),
    }
    if seed is not None:
        metadata["seed"] = seed

    run_training_loop(
        training_step,
        dataloader,
        num_epochs,
        checkpoint_path,
        log_every=10,
        experiment_log_dir=experiment_log_dir,
        metadata=metadata,
        seed=seed,
    )


def build_supervised_training_components(
    feature_dim: int,
    hidden_dim: int,
    num_layers: int,
    learning_rate: float,
    device: str,
) -> dict[str, object]:
    """Construct the torch modules and optimizer used by supervised training.

    Args:
        feature_dim: Conditioning feature dimension used by the encoder.
        hidden_dim: Hidden width of the grasp-generation model.
        num_layers: Number of layers in the grasp-generation model.
        learning_rate: Learning rate for the optimizer.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.

    Returns:
        A dictionary containing ``"model"``, ``"optimizer"`` and any other
        components required by the supervised training loop.
    """
    from grasping_ai.models.diffusion import GraspGeneratorModel
    from grasping_ai.training.trainer import build_adam_optimizer

    model = GraspGeneratorModel(feature_dim, hidden_dim, num_layers)
    model.to(device)
    optimizer = build_adam_optimizer(model.parameters(), learning_rate)

    return {"model": model, "optimizer": optimizer}


def load_pretrained_encoder(
    checkpoint_path: Path, device: str,
) -> dict[str, torch.Tensor]:
    """Load a pretrained equivariant encoder from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint containing encoder weights.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.

    Returns:
        A mapping from parameter names to tensors describing the loaded
        encoder state.
    """
    if not isinstance(checkpoint_path, Path):
        raise TypeError("checkpoint_path must be a pathlib.Path instance")
    from grasping_ai.training.checkpoint_io import load_torch_checkpoint

    checkpoint = load_torch_checkpoint(checkpoint_path, device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    encoder_state = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            encoder_state[k[len("encoder."):]] = v
        else:
            encoder_state[k] = v
    return encoder_state
