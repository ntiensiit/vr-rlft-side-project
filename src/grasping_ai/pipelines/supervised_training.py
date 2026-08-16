"""Shared supervised training loop utilities."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.utils.data import DataLoader, Dataset

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
from grasping_ai.inference.grasp_sampling import encode_grasp_conditioning

TrainingDataset = Dataset[tuple[torch.Tensor, torch.Tensor]] | list[tuple[torch.Tensor, torch.Tensor]]
BatchCollator = Callable[[list[tuple[torch.Tensor, torch.Tensor]]], tuple[torch.Tensor, torch.Tensor]]
SEED = int(FLATTENED_YAML_CONFIG.get("seed", 42))


def build_supervised_dataloader(
    dataset: TrainingDataset,
    batch_size: int,
    seed: int | None,
    *,
    num_workers: int = 0,
    collate_fn: BatchCollator | None = None,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build a reproducibly shuffled DataLoader for supervised pairs."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    generator = torch.Generator().manual_seed(SEED if seed is None else seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def build_conditioned_dataloader(
    dataset: TrainingDataset,
    batch_size: int,
    device: str,
    seed: int | None,
    encoder: torch.nn.Module,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build a shuffled DataLoader that batches encoded point-cloud samples."""
    device_obj = torch.device(device)

    def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        point_clouds, targets = torch.utils.data.default_collate(batch)
        conditioning, _, _ = encode_grasp_conditioning(encoder, point_clouds.to(device_obj))
        return conditioning, targets.to(device_obj)

    return build_supervised_dataloader(dataset, batch_size, seed, collate_fn=collate)
