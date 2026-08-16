"""Build supervised training pairs from grasp samples."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
from grasping_ai.data.grasp_vector import se3_to_vec
from grasping_ai.data.pointcloud_dataset import (
    discover_dataset_files,
    iterate_grasp_dataset,
    load_grasp_sample,
)
from grasping_ai.data.transforms import (
    compose_transforms,
    make_random_rotation_jitter,
    make_translation_jitter,
)
from grasping_ai.models.equivariant_encoder import (
    compute_se3_frame,
    invert_rigid_transform_batch,
    world_transform_from_frame,
)
from grasping_ai.utils.path_validation import require_path

if TYPE_CHECKING:
    from pathlib import Path

AUGMENT = bool(FLATTENED_YAML_CONFIG.get("training.augment", False))
SEED = int(FLATTENED_YAML_CONFIG.get("seed", 42))
MIN_GRASP_SCORE = float(FLATTENED_YAML_CONFIG.get("supervised.min_grasp_score", 0.0))
SCORE_REPEAT_FACTOR = int(FLATTENED_YAML_CONFIG.get("supervised.score_repeat_factor", 0))
SCORE_REPEAT_POWER = float(FLATTENED_YAML_CONFIG.get("supervised.score_repeat_power", 1.0))
TRANSLATION_JITTER_SCALE = float(FLATTENED_YAML_CONFIG.get("training.translation_jitter_scale", 0.01))


def _resample_point_cloud(point_cloud: np.ndarray, num_points: int, seed: int) -> np.ndarray:
    """Return a deterministic fixed-size point cloud, sampling with replacement when needed."""
    if point_cloud.shape[0] == num_points:
        return point_cloud
    rng = np.random.default_rng(seed)
    indices = rng.choice(point_cloud.shape[0], size=num_points, replace=point_cloud.shape[0] < num_points)
    return point_cloud[indices]


class SupervisedGraspDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Map-style Dataset exposing canonical point-cloud/grasp training pairs.

    Dataset records remain stored as validated NPZ files. The dataset indexes
    grasp targets during construction, then loads and transforms only the
    selected record in ``__getitem__``.
    """

    def __init__(  # noqa: PLR0913
        self,
        dataset_root: Path,
        *,
        augment: bool = AUGMENT,
        seed: int | None = SEED,
        min_grasp_score: float = MIN_GRASP_SCORE,
        score_repeat_factor: int = SCORE_REPEAT_FACTOR,
        score_repeat_power: float = SCORE_REPEAT_POWER,
        num_points: int | None = None,
    ) -> None:
        """Index validated records and configure deterministic preprocessing."""
        require_path(dataset_root, "dataset_root")
        self.records = discover_dataset_files(dataset_root)
        self.augment = augment
        self.seed = 0 if seed is None else seed
        self.min_grasp_score = min_grasp_score
        self.score_repeat_factor = score_repeat_factor
        self.score_repeat_power = score_repeat_power
        first_sample = load_grasp_sample(self.records[0])
        self.num_points = num_points or int(first_sample["point_cloud"].shape[0])
        if self.num_points <= 0:
            msg = "num_points must be positive"
            raise ValueError(msg)
        self._items: list[tuple[int, int]] = []
        for record_index, record in enumerate(self.records):
            sample = load_grasp_sample(record)
            grasp_poses = sample["grasp_poses"]
            scores = sample.get("scores")
            if not isinstance(grasp_poses, np.ndarray):
                msg = f"Record {record} grasp poses must be a numpy array"
                raise TypeError(msg)
            if len(grasp_poses) == 0:
                msg = f"Record {record} has no target grasp poses"
                raise ValueError(msg)
            grasp_indices = _grasp_indices_above_score(
                grasp_poses,
                scores,
                self.min_grasp_score,
                record,
                f"={self.min_grasp_score}",
            )
            max_score = 0.0
            score_values = scores if isinstance(scores, np.ndarray) and scores.shape[0] == len(grasp_poses) else None
            if score_values is not None and self.score_repeat_factor > 0:
                max_score = float(np.max(score_values[grasp_indices]))
            for grasp_index in grasp_indices:
                repeats = _grasp_repeat_count(
                    score_values,
                    grasp_index,
                    max_score,
                    self.score_repeat_factor,
                    self.score_repeat_power,
                )
                self._items.extend((record_index, grasp_index) for _ in range(repeats))

    def __len__(self) -> int:
        """Return the number of indexed grasp examples."""
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and return one fixed-size point-cloud/grasp-vector pair."""
        if not isinstance(index, int):
            raise TypeError("index must be an integer")
        record_index, grasp_index = self._items[index]
        record = self.records[record_index]
        sample = load_grasp_sample(record)
        pc = sample["point_cloud"]
        grasp_poses = sample["grasp_poses"]
        scores = sample.get("scores")
        if not isinstance(grasp_poses, np.ndarray):
            msg = f"Record {record} grasp poses must be a numpy array"
            raise TypeError(msg)
        if self.augment:
            augment_rng = np.random.default_rng(self.seed + record_index)
            pc, grasp_poses, scores = _augment_sample(pc, grasp_poses, scores, augment_rng, record)
        pc = _resample_point_cloud(pc, self.num_points, self.seed + index)
        pc_t = torch.from_numpy(np.asarray(pc, dtype=np.float32)).float()
        frame, centroid = compute_se3_frame(pc_t.unsqueeze(0))
        world = world_transform_from_frame(frame, centroid)[0]
        world_inv = invert_rigid_transform_batch(world.unsqueeze(0))[0]
        target = _canonical_grasp_vector(world_inv, world, grasp_poses[grasp_index])
        return pc_t, torch.from_numpy(target).float()


def validate_grasp_dataset(dataset_root: Path) -> int:
    """Validate that a dataset root yields readable grasp samples.

    Args:
        dataset_root: Root directory containing ``.npz`` grasp dataset records.

    Returns:
        Number of valid grasp samples discovered under ``dataset_root``.

    Raises:
        TypeError: If ``dataset_root`` is not a ``pathlib.Path`` instance.
        ValueError: If no valid grasp samples are found.
    """
    require_path(dataset_root, "dataset_root")
    count = sum(1 for _ in iterate_grasp_dataset(dataset_root))
    if count == 0:
        msg = f"Dataset at {dataset_root} contains no valid grasp samples"
        raise ValueError(msg)
    return count


def _grasp_indices_above_score(
    grasp_poses: np.ndarray,
    scores: np.ndarray | None,
    min_grasp_score: float,
    record: Path,
    error_detail: str,
) -> list[int]:
    """Return grasp indices whose scores clear ``min_grasp_score``.

    Raises:
        ValueError: If scores are present and no grasp clears the threshold.
    """
    grasp_indices = list(range(len(grasp_poses)))
    if isinstance(scores, np.ndarray) and scores.shape[0] == len(grasp_poses):
        grasp_indices = [idx for idx in grasp_indices if float(scores[idx]) >= min_grasp_score]
        if not grasp_indices:
            msg = f"Record {record} has no grasp poses above min_grasp_score{error_detail}"
            raise ValueError(msg)
    return grasp_indices


def _augment_sample(
    pc: np.ndarray,
    grasp_poses: np.ndarray,
    scores: np.ndarray | None,
    augment_rng: np.random.Generator,
    record: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Apply rotation/translation jitter to one record.

    Raises:
        ValueError: If augmentation removes the grasp poses.
    """
    sample_transform = compose_transforms(
        make_random_rotation_jitter(augment_rng),
        make_translation_jitter(augment_rng, scale=TRANSLATION_JITTER_SCALE),
    )
    transformed_pc, transformed_grasp_poses, transformed_scores = sample_transform(
        pc,
        grasp_poses,
        scores if isinstance(scores, np.ndarray) else None,
    )
    if transformed_grasp_poses is None:
        msg = f"Augmentation removed grasp poses for {record}"
        raise ValueError(msg)
    return transformed_pc, transformed_grasp_poses, transformed_scores


def _canonical_grasp_vector(world_inv: torch.Tensor, world: torch.Tensor, grasp_pose: np.ndarray) -> np.ndarray:
    """Express one grasp pose in the canonical object frame as a 9D vector."""
    t_matrix = np.asarray(grasp_pose, dtype=np.float32)
    t_tensor = torch.from_numpy(t_matrix).float()
    canonical = world_inv @ t_tensor @ world
    return se3_to_vec(canonical.numpy())


def _grasp_repeat_count(
    score_values: np.ndarray | None,
    grasp_index: int,
    max_score: float,
    score_repeat_factor: int,
    score_repeat_power: float,
) -> int:
    """Compute the score-weighted duplication count for one grasp."""
    if score_values is None or score_repeat_factor <= 0 or max_score <= 0.0:
        return 1
    normalized = float(score_values[grasp_index]) / max_score
    return max(1, round(normalized**score_repeat_power * score_repeat_factor))
