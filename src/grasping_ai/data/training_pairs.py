"""Build supervised training pairs from grasp samples."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger

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
        make_translation_jitter(augment_rng, scale=0.01),
    )
    pc, grasp_poses, transformed_scores = sample_transform(
        pc,
        grasp_poses,
        scores if isinstance(scores, np.ndarray) else None,
    )
    if grasp_poses is None:
        msg = f"Augmentation removed grasp poses for {record}"
        raise ValueError(msg)
    return pc, grasp_poses, transformed_scores


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


def _pairs_from_record(
    record: Path,
    augment_rng: np.random.Generator | None,
    min_grasp_score: float,
    score_repeat_factor: int,
    score_repeat_power: float,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build canonical-frame training pairs for a single dataset record."""
    sample = load_grasp_sample(record)
    pc = sample["point_cloud"]
    grasp_poses = sample["grasp_poses"]
    scores = sample.get("scores")
    if not isinstance(pc, np.ndarray):
        msg = f"Record {record} point cloud must be a numpy array"
        raise TypeError(msg)
    if not isinstance(grasp_poses, np.ndarray):
        msg = f"Record {record} grasp poses must be a numpy array"
        raise TypeError(msg)
    if len(grasp_poses) == 0:
        msg = f"Record {record} has no target grasp poses"
        raise ValueError(msg)

    grasp_indices = _grasp_indices_above_score(
        grasp_poses,
        scores,
        min_grasp_score,
        record,
        f"={min_grasp_score}",
    )
    if augment_rng is not None:
        pc, grasp_poses, scores = _augment_sample(pc, grasp_poses, scores, augment_rng, record)
        grasp_indices = _grasp_indices_above_score(
            grasp_poses,
            scores,
            min_grasp_score,
            record,
            " after augment",
        )

    pc_t = torch.from_numpy(pc).float()
    frame, centroid = compute_se3_frame(pc_t.unsqueeze(0))
    world = world_transform_from_frame(frame, centroid)[0]
    world_inv = invert_rigid_transform_batch(world.unsqueeze(0))[0]

    score_values: np.ndarray | None = None
    if isinstance(scores, np.ndarray) and scores.shape[0] == len(grasp_poses):
        score_values = scores

    max_score = 0.0
    if score_values is not None and score_repeat_factor > 0:
        max_score = float(np.max(score_values[grasp_indices]))

    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for grasp_index in grasp_indices:
        t_vec = _canonical_grasp_vector(world_inv, world, grasp_poses[grasp_index])
        pair = (pc_t, torch.from_numpy(t_vec).float())
        repeats = _grasp_repeat_count(score_values, grasp_index, max_score, score_repeat_factor, score_repeat_power)
        pairs.extend([pair] * repeats)
    return pairs


# Public API: callers (pipelines/tests) pass every option by keyword.
def build_supervised_training_pairs(  # noqa: PLR0913
    dataset_root: Path,
    *,
    augment: bool = False,
    seed: int | None = None,
    min_grasp_score: float = 0.0,
    score_repeat_factor: int = 0,
    score_repeat_power: float = 1.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Load grasp dataset records as canonical-frame ``(point_cloud, grasp_9d)`` pairs.

    Expresses each grasp target in the encoder's canonical object frame so that
    supervised targets match the canonical-frame outputs of the generative heads;
    inference later maps canonical grasps back to the input frame.

    Args:
        dataset_root: Root directory of the grasp-pose dataset.
        augment: When ``True``, apply SO(3)/translation jitter to each sample.
        seed: Optional seed controlling augmentation randomness.
        min_grasp_score: Drop grasps whose score is below this threshold when
            scores are present in the dataset record.
        score_repeat_factor: When positive and scores are present, duplicate
            higher-scoring grasps up to this many weighted copies per grasp.
        score_repeat_power: Exponent applied to normalized scores when repeating
            grasps for weighted sampling.

    Returns:
        List of ``(point_cloud, grasp_vector)`` tensor pairs.

    Raises:
        TypeError: If ``dataset_root`` is not a ``pathlib.Path`` instance, or a
            record field has an invalid type.
        ValueError: If the dataset is empty, a record has no grasp poses, or
            augmentation removes all grasp poses for a record.
    """
    require_path(dataset_root, "dataset_root")

    records = discover_dataset_files(dataset_root)

    if not records:
        msg = "Dataset is empty"
        raise ValueError(msg)

    augment_rng = np.random.default_rng(seed) if augment else None

    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for record in records:
        pairs.extend(
            _pairs_from_record(record, augment_rng, min_grasp_score, score_repeat_factor, score_repeat_power),
        )

    logger.info(
        "Built {} supervised training pairs from {} records (augment={}, min_grasp_score={})",
        len(pairs),
        len(records),
        augment,
        min_grasp_score,
    )
    return pairs
