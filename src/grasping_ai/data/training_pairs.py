from pathlib import Path
from typing import cast

import numpy as np
import torch

from grasping_ai.data.grasp_vector import se3_to_vec
from grasping_ai.data.pointcloud_dataset import (
    discover_dataset_files,
    iterate_grasp_dataset,
    load_grasp_sample,
)
from grasping_ai.models.equivariant_encoder import compute_se3_frame, world_transform_from_frame
from grasping_ai.robotics.transforms import invert_rigid_transform


def validate_grasp_dataset(dataset_root: Path) -> int:
    """Validate that a dataset root yields readable grasp samples.

    Args:
        dataset_root: Root directory containing ``.npy`` grasp dataset records.

    Returns:
        Number of valid grasp samples discovered under ``dataset_root``.

    Raises:
        TypeError: If ``dataset_root`` is not a ``pathlib.Path`` instance.
        ValueError: If no valid grasp samples are found.
    """
    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be a pathlib.Path instance")
    count = sum(1 for _ in iterate_grasp_dataset(dataset_root))
    if count == 0:
        raise ValueError(f"Dataset at {dataset_root} contains no valid grasp samples")
    return count


def build_supervised_training_pairs(
    dataset_root: Path,
    augment: bool = False,
    seed: int | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Load grasp dataset records as canonical-frame ``(point_cloud, grasp_9d)`` pairs.

    Expresses each grasp target in the encoder's canonical object frame so that
    supervised targets match the canonical-frame outputs of the generative heads;
    inference later maps canonical grasps back to the input frame.

    Args:
        dataset_root: Root directory of the grasp-pose dataset.
        augment: When ``True``, apply SO(3)/translation jitter to each sample.
        seed: Optional seed controlling augmentation randomness.

    Returns:
        List of ``(point_cloud, grasp_vector)`` tensor pairs.

    Raises:
        TypeError: If ``dataset_root`` is not a ``pathlib.Path`` instance, or a
            record field has an invalid type.
        ValueError: If the dataset is empty, a record has no grasp poses, or
            augmentation removes all grasp poses for a record.
    """
    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be a pathlib.Path instance")

    records = discover_dataset_files(dataset_root)
    if not records:
        raise ValueError("Dataset is empty")

    augment_rng = np.random.default_rng(seed) if augment else None

    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for record in records:
        sample = load_grasp_sample(record)
        pc = sample["point_cloud"]
        grasp_poses = sample["grasp_poses"]
        if not isinstance(pc, np.ndarray):
            raise TypeError(f"Record {record} point cloud must be a numpy array")
        if not isinstance(grasp_poses, np.ndarray):
            raise TypeError(f"Record {record} grasp poses must be a numpy array")
        if len(grasp_poses) == 0:
            raise ValueError(f"Record {record} has no target grasp poses")

        if augment_rng is not None:
            from grasping_ai.data.transforms import (
                compose_transforms,
                make_random_rotation_jitter,
                make_translation_jitter,
            )

            sample_transform = compose_transforms(
                make_random_rotation_jitter(augment_rng),
                make_translation_jitter(augment_rng, scale=0.01),
            )
            pc, grasp_poses, _ = sample_transform(pc, grasp_poses, None)
            if grasp_poses is None:
                raise ValueError(f"Augmentation removed grasp poses for {record}")

        pc_t = torch.from_numpy(pc).float()
        frame, centroid = compute_se3_frame(pc_t.unsqueeze(0))
        world = world_transform_from_frame(frame, centroid)[0]
        world_inv = torch.from_numpy(
            invert_rigid_transform(world.detach().cpu().numpy())
        ).float()
        for t_matrix in grasp_poses:
            t_tensor = torch.from_numpy(cast(np.ndarray, t_matrix)).float()
            canonical = world_inv @ t_tensor @ world
            t_vec = se3_to_vec(canonical.numpy())
            pairs.append((pc_t, torch.from_numpy(t_vec).float()))

    return pairs
