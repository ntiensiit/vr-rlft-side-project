from collections.abc import Callable
from pathlib import Path

import numpy as np

SampleTransform = Callable[
    [np.ndarray, np.ndarray | None, np.ndarray | None],
    tuple[np.ndarray, np.ndarray | None, np.ndarray | None],
]


def make_random_rotation_jitter(rng: np.random.Generator) -> SampleTransform:
    """Build a transform that applies a random SO(3) rotation to a sample.

    Args:
        rng: NumPy random generator used for sampling rotations.

    Returns:
        A callable transform operating on point cloud, grasp poses, and scores.
    """
    raise NotImplementedError


def make_translation_jitter(rng: np.random.Generator, scale: float) -> SampleTransform:
    """Build a transform that applies a small random translation to a sample.

    Args:
        rng: NumPy random generator.
        scale: Maximum magnitude of the translation offset.

    Returns:
        A callable transform operating on point cloud, grasp poses, and scores.
    """
    raise NotImplementedError


def compose_transforms(*transforms: SampleTransform) -> SampleTransform:
    """Compose multiple sample transforms into a single callable.

    Args:
        transforms: Ordered sample transforms to apply sequentially.

    Returns:
        A callable that applies each transform in order to a sample.
    """
    raise NotImplementedError


def save_grasp_dataset_index(dataset_root: Path, entries: list[dict[str, str]]) -> None:
    """Persist a dataset index file describing available records.

    Args:
        dataset_root: Root directory under which the index file is written.
        entries: List of metadata entries describing dataset records.
    """
    raise NotImplementedError
