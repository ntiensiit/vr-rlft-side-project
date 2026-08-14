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
    """Yield shuffled ``(point_clouds, targets)`` batches for supervised training.

    Args:
        pairs: Training samples as ``(point_cloud, grasp_vector)`` tuples.
        batch_size: Maximum number of samples per yielded batch.
        device: Torch device string passed to ``Tensor.to``.
        seed: Optional shuffle seed; defaults to ``42`` when omitted.

    Yields:
        Batched ``(point_clouds, targets)`` tensors on ``device``.
    """
    num_samples = len(pairs)
    indices = list(range(num_samples))
    local_random = random.Random(seed if seed is not None else 42)
    local_random.shuffle(indices)

    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i : i + batch_size]
        point_clouds = torch.stack([pairs[idx][0] for idx in batch_indices]).to(device)
        targets = torch.stack([pairs[idx][1] for idx in batch_indices]).to(device)
        yield point_clouds, targets


def iter_conditioned_training_batches(
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
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
