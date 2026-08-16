"""Shared supervised training loop utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Dataset

from grasping_ai.inference.grasp_sampling import encode_grasp_conditioning

if TYPE_CHECKING:
    from collections.abc import Iterator

TrainingDataset = Dataset[tuple[torch.Tensor, torch.Tensor]] | list[tuple[torch.Tensor, torch.Tensor]]


def build_supervised_dataloader(
    dataset: TrainingDataset,
    batch_size: int,
    seed: int | None,
    *,
    num_workers: int = 0,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build a reproducibly shuffled DataLoader for supervised pairs."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    generator = torch.Generator().manual_seed(42 if seed is None else seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
    )


def iter_supervised_training_batches(
    pairs: TrainingDataset,
    batch_size: int,
    device: str,
    seed: int | None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield shuffled ``(point_clouds, targets)`` batches for supervised training.

    Args:
        pairs: Training samples as ``(point_cloud, grasp_vector)`` tuples.
        batch_size: Maximum number of samples per yielded batch.
        device: Torch device string passed to ``Tensor.to``.
        seed: Optional shuffle seed; defaults to ``42`` when omitted.

    Yields:
        Batched ``(point_clouds, targets)`` tensors on ``device``.
    """
    for point_clouds, targets in build_supervised_dataloader(pairs, batch_size, seed):
        yield point_clouds.to(device), targets.to(device)


def iter_conditioned_training_batches(
    pairs: TrainingDataset,
    batch_size: int,
    device: str,
    seed: int | None,
    encoder: torch.nn.Module,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield shuffled ``(conditioning, targets)`` batches for diffusion training.

    Args:
        pairs: Training samples as ``(point_cloud, grasp_vector)`` tuples.
        batch_size: Maximum number of samples per yielded batch.
        device: Torch device string passed to ``Tensor.to``.
        seed: Optional shuffle seed forwarded to ``iter_supervised_training_batches``.
        encoder: Equivariant encoder used to precompute conditioning features.

    Yields:
        Batched ``(conditioning, targets)`` tensors on ``device``.
    """
    for point_clouds, targets in iter_supervised_training_batches(pairs, batch_size, device, seed):
        conditioning, _, _ = encode_grasp_conditioning(encoder, point_clouds)
        yield conditioning, targets
