import random
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import cast

import numpy as np
import torch

from grasping_ai.data.pointcloud_dataset import (
    discover_dataset_files,
    load_grasp_sample,
)
from grasping_ai.models.equivariant_encoder import (
    build_equivariant_encoder,
    compute_se3_frame,
    encode_point_cloud,
    pool_object_features,
    world_transform_from_frame,
)
from grasping_ai.models.flow import build_flow_field
from grasping_ai.training.losses import build_flow_matching_loss
from grasping_ai.training.trainer import (
    build_adam_optimizer,
    run_training_loop,
    save_training_checkpoint,
)


def build_flow_training_components(
    feature_dim: int,
    hidden_dim: int,
    num_layers: int,
    learning_rate: float,
    device: str,
) -> dict[str, object]:
    """Construct the flow model and optimizer used by flow-based training.

    Args:
        feature_dim: Conditioning feature dimension from the encoder.
        hidden_dim: Width of the flow-field hidden layers.
        num_layers: Number of hidden layers in the flow field.
        learning_rate: Learning rate for the Adam optimizer.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.

    Returns:
        A dictionary containing the flow ``"model"``, the ``"optimizer"`` and
        the ``"flow_field"`` callable produced by ``models.flow``.
    """
    flow_field = cast(torch.nn.Module, build_flow_field(feature_dim, hidden_dim, num_layers))
    flow_field.to(device)
    optimizer = build_adam_optimizer(flow_field.parameters(), learning_rate)
    return {"model": flow_field, "flow_field": flow_field, "optimizer": optimizer}


def build_flow_training_step(
    flow_field: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: str,
    seed: int | None = None,
) -> Callable[[torch.Tensor, torch.Tensor], dict[str, float]]:
    """Build a callable training step closure for a flow-matching model.

    Args:
        flow_field: ``FlowFieldNet`` instance being trained.
        loss_fn: Loss function returned by ``training.losses.build_flow_matching_loss``.
        optimizer: Optimizer returned by ``build_adam_optimizer``.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.
        seed: Optional random seed for reproducible interpolation sampling.

    Returns:
        A callable that consumes ``(conditioning, targets)`` and returns a
        dictionary of training metrics for the step.
    """
    device_obj = torch.device(device)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device_obj).manual_seed(seed)

    def step(
        conditioning: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, float]:
        flow_field.train()
        optimizer.zero_grad()

        cond = conditioning.to(device_obj)
        x_1 = targets.to(device_obj)
        batch_size_val = x_1.shape[0]

        t = torch.rand(
            batch_size_val,
            dtype=x_1.dtype,
            device=device_obj,
            generator=generator,
        )
        x_0 = torch.randn(
            x_1.shape,
            dtype=x_1.dtype,
            device=device_obj,
            generator=generator,
        )
        t_view = t.view(batch_size_val, 1)
        x_t = (1.0 - t_view) * x_0 + t_view * x_1
        target_velocity = x_1 - x_0

        predicted_velocity = flow_field(x_t, cond)
        loss = loss_fn(predicted_velocity, target_velocity)

        loss.backward()
        optimizer.step()

        return {"loss": float(loss.item())}

    step.model = flow_field  # type: ignore[attr-defined]
    step.optimizer = optimizer  # type: ignore[attr-defined]
    return step


def _se3_to_vec(t_matrix: np.ndarray) -> np.ndarray:
    """Convert a 4x4 SE(3) transform into a 9D position+rotation-column vector."""
    t = t_matrix[:3, 3]
    r1 = t_matrix[:3, 0]
    r2 = t_matrix[:3, 1]
    return np.concatenate([t, r1, r2])


def _flow_dataset_pairs(
    dataset_root: Path,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Load the synthetic-grasp dataset and convert each grasp to canonical 9D.

    Args:
        dataset_root: Root directory of the grasp-pose dataset.

    Returns:
        List of ``(point_cloud, grasp_vector)`` pairs in canonical frame.
    """
    records = discover_dataset_files(dataset_root)
    if not records:
        raise ValueError("Dataset is empty")

    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for record in records:
        sample = load_grasp_sample(record)
        pc = sample["point_cloud"]
        grasp_poses = sample["grasp_poses"]
        if grasp_poses is None or len(grasp_poses) == 0:
            raise ValueError(
                f"Record {record} has no target grasp poses"
            )

        pc_t = torch.from_numpy(pc).float()
        frame, centroid = compute_se3_frame(pc_t.unsqueeze(0))
        world = world_transform_from_frame(frame, centroid)[0]
        world_inv = torch.linalg.inv(world)
        for t_matrix in grasp_poses:
            t_tensor = torch.from_numpy(cast(np.ndarray, t_matrix)).float()
            canonical = world_inv @ t_tensor @ world
            t_vec = _se3_to_vec(canonical.numpy())
            pairs.append((pc_t, torch.from_numpy(t_vec).float()))

    return pairs


class _FlowTrainingDataloader:
    """Iterable dataloader that emits ``(cond, targets)`` batches per epoch."""

    def __init__(
        self,
        pairs: list[tuple[torch.Tensor, torch.Tensor]],
        batch_size: int,
        device: str,
        seed: int | None,
    ) -> None:
        self.pairs = pairs
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.encoder: torch.nn.Module | None = None

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        num_samples = len(self.pairs)
        indices = list(range(num_samples))
        local_random = random.Random(self.seed if self.seed is not None else 42)
        local_random.shuffle(indices)

        for i in range(0, num_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            pcs = torch.stack([self.pairs[idx][0] for idx in batch_indices]).to(
                self.device
            )
            targets = torch.stack(
                [self.pairs[idx][1] for idx in batch_indices]
            ).to(self.device)

            if self.encoder is None:
                raise RuntimeError(
                    "Flow dataloader requires encoder set via set_encoder()"
                )
            features = encode_point_cloud(self.encoder, pcs)
            cond = pool_object_features(features)

            yield cond, targets

    def set_encoder(self, encoder: torch.nn.Module) -> None:
        self.encoder = encoder


def run_flow_training_pipeline(
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
) -> None:
    """Run the end-to-end flow-matching training pipeline for grasp generation.

    Mirrors ``run_training_pipeline`` but uses a continuous-time flow-matching
    objective on the canonical-frame 9D grasp vectors instead of the discrete
    diffusion score-matching loss.

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
    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be a pathlib.Path instance")
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {dataset_root}"
        )

    if seed is not None:
        torch.manual_seed(seed)

    components = build_flow_training_components(
        feature_dim, hidden_dim, num_layers, learning_rate, device
    )
    flow_field = components["model"]
    optimizer = components["optimizer"]

    encoder = cast(torch.nn.Module, build_equivariant_encoder(feature_dim, 2))
    encoder.to(device)

    pairs = _flow_dataset_pairs(dataset_root)

    loss_fn = build_flow_matching_loss(cast(Callable[..., torch.Tensor], flow_field))
    training_step = build_flow_training_step(
        cast(torch.nn.Module, flow_field),
        loss_fn,
        cast(torch.optim.Optimizer, optimizer),
        device,
        seed=seed,
    )

    dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]] = _FlowTrainingDataloader(
        pairs, batch_size, device, seed
    )
    if isinstance(dataloader, _FlowTrainingDataloader):
        dataloader.set_encoder(encoder)

    metadata = {
        "pipeline": "flow",
        "dataset_root": str(dataset_root),
        "checkpoint_path": str(checkpoint_path),
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "device": device,
    }
    if seed is not None:
        metadata["seed"] = seed

    run_training_loop(
        cast(Callable[[torch.Tensor, torch.Tensor], dict[str, float]], training_step),
        dataloader,
        num_epochs,
        checkpoint_path,
        log_every=10,
        experiment_log_dir=experiment_log_dir,
        metadata=metadata,
        seed=seed,
    )

    save_training_checkpoint(
        cast(torch.nn.Module, flow_field),
        cast(torch.optim.Optimizer, optimizer),
        num_epochs,
        checkpoint_path,
        seed=seed,
    )
