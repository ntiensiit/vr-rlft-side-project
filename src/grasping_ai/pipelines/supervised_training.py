import random
from collections.abc import Iterator

import torch

from grasping_ai.inference.grasp_sampling import encode_grasp_conditioning


def iter_supervised_training_batches(
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    device: str,
    seed: int | None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield shuffled ``(point_clouds, targets)`` batches for supervised training."""
    num_samples = len(pairs)
    indices = list(range(num_samples))
    local_random = random.Random(seed if seed is not None else 42)
    local_random.shuffle(indices)

    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i : i + batch_size]
        point_clouds = torch.stack([pairs[idx][0] for idx in batch_indices]).to(device)
        targets = torch.stack([pairs[idx][1] for idx in batch_indices]).to(device)
        yield point_clouds, targets


class SupervisedTrainingDataloader:
    """Iterable dataloader that emits ``(pcs, targets)`` batches per epoch."""

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

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return iter_supervised_training_batches(
            self.pairs, self.batch_size, self.device, self.seed
        )


class ConditionedTrainingDataloader:
    """Iterable dataloader that pre-encodes point clouds into conditioning features."""

    def __init__(
        self,
        pairs: list[tuple[torch.Tensor, torch.Tensor]],
        batch_size: int,
        device: str,
        seed: int | None,
        encoder: torch.nn.Module,
    ) -> None:
        self.pairs = pairs
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.encoder = encoder

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for point_clouds, targets in iter_supervised_training_batches(
            self.pairs, self.batch_size, self.device, self.seed
        ):
            conditioning, _, _ = encode_grasp_conditioning(self.encoder, point_clouds)
            yield conditioning, targets
