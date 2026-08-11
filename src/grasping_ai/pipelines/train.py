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
    """
    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be a pathlib.Path instance")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    import numpy as np

    # Build components
    components = build_supervised_training_components(
        feature_dim, hidden_dim, num_layers, learning_rate, device
    )
    model = components["model"]
    optimizer = components["optimizer"]

    # Discover files using Phase 3 data contract
    from grasping_ai.data.pointcloud_dataset import discover_dataset_files, load_grasp_sample
    try:
        records = discover_dataset_files(dataset_root)
    except Exception as e:
        raise ValueError(f"Failed to discover dataset files: {e}") from e

    if not records:
        raise ValueError("Dataset is empty")

    # Load and check samples
    training_pairs = []
    for record in records:
        sample = load_grasp_sample(record)
        pc = sample["point_cloud"]
        grasp_poses = sample["grasp_poses"]
        if grasp_poses is None or len(grasp_poses) == 0:
            raise ValueError(f"Record {record} has no target grasp poses")

        # Helper to convert T (4, 4) to 9D representation
        def se3_to_vec(t_matrix: np.ndarray) -> np.ndarray:
            t = t_matrix[:3, 3]
            r1 = t_matrix[:3, 0]
            r2 = t_matrix[:3, 1]
            return np.concatenate([t, r1, r2])

        pc_t = torch.from_numpy(pc).float()
        for t_matrix in grasp_poses:
            t_vec = se3_to_vec(cast(np.ndarray, t_matrix))
            t_tensor = torch.from_numpy(t_vec).float()
            training_pairs.append((pc_t, t_tensor))

    # Construct the training step and training loop
    from grasping_ai.models.diffusion import GraspGeneratorModel
    from grasping_ai.training.losses import build_diffusion_score_loss
    from grasping_ai.training.trainer import build_training_step, run_training_loop

    model_generator = cast(GraspGeneratorModel, model)
    loss_fn = build_diffusion_score_loss(model_generator.score_net)
    training_step = build_training_step(
        model_generator, loss_fn, cast(torch.optim.Optimizer, optimizer), device
    )

    # Iterable loader helper
    class TrainingDataloader:
        def __init__(
            self, pairs: list[tuple[torch.Tensor, torch.Tensor]], b_size: int, dev: str
        ) -> None:
            self.pairs = pairs
            self.b_size = b_size
            self.dev = dev

        def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            num_samples = len(self.pairs)
            indices = list(range(num_samples))
            import random
            random.seed(42)
            random.shuffle(indices)

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

    dataloader = TrainingDataloader(training_pairs, batch_size, device)
    dataloader_iter = cast(Iterator[tuple[torch.Tensor, torch.Tensor]], iter(dataloader))
    run_training_loop(training_step, dataloader_iter, num_epochs, checkpoint_path, log_every=10)


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


def load_pretrained_encoder(checkpoint_path: Path, device: str) -> torch.Tensor:
    """Load a pretrained equivariant encoder from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint containing encoder weights.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.

    Returns:
        The loaded encoder parameters as a state-dict-like tensor container.
    """
    if not isinstance(checkpoint_path, Path):
        raise TypeError("checkpoint_path must be a pathlib.Path instance")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise ValueError(f"Failed to load encoder: {e}") from e

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    encoder_state = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            encoder_state[k[len("encoder."):]] = v
        else:
            encoder_state[k] = v
    from typing import cast
    return cast(torch.Tensor, encoder_state)
