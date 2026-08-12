from pathlib import Path
from typing import cast

import numpy as np
import torch

from grasping_ai.data.grasp_vector import se3_to_vec
from grasping_ai.data.pointcloud_dataset import discover_dataset_files, load_grasp_sample
from grasping_ai.models.equivariant_encoder import compute_se3_frame, world_transform_from_frame
from grasping_ai.robotics.transforms import invert_rigid_transform


def build_supervised_training_pairs(
    dataset_root: Path,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Load grasp dataset records as canonical-frame ``(point_cloud, grasp_9d)`` pairs.

    Expresses each grasp target in the encoder's canonical object frame so that
    supervised targets match the canonical-frame outputs of the generative heads;
    inference later maps canonical grasps back to the input frame.

    Args:
        dataset_root: Root directory of the grasp-pose dataset.

    Returns:
        List of ``(point_cloud, grasp_vector)`` tensor pairs.
    """
    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be a pathlib.Path instance")

    records = discover_dataset_files(dataset_root)
    if not records:
        raise ValueError("Dataset is empty")

    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for record in records:
        sample = load_grasp_sample(record)
        pc = sample["point_cloud"]
        grasp_poses = sample["grasp_poses"]
        if grasp_poses is None or len(grasp_poses) == 0:
            raise ValueError(f"Record {record} has no target grasp poses")

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
