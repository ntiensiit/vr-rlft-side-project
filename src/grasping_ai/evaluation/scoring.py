"""Shared analytical grasp scoring helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
from grasping_ai.evaluation.collision import generate_analytical_contacts
from grasping_ai.evaluation.force_closure import compute_grasp_quality

if TYPE_CHECKING:
    import numpy as np

MIN_GRASP_CONTACTS = 2
FRICTION_COEFFICIENT = float(FLATTENED_YAML_CONFIG.get("metrics.friction_coefficient", 0.5))
CONTACT_CLEARANCE = float(FLATTENED_YAML_CONFIG.get("metrics.collision_clearance", 0.005))
MIN_QUALITY_SCORE = float(FLATTENED_YAML_CONFIG.get("evaluation.min_quality_score", 0.0))


def _antipodal_contact_quality(contacts: list[dict[str, np.ndarray]], friction_coefficient: float) -> float:
    """Score two-contact parallel-jaw grasps when full 6D closure is rank-deficient."""
    if len(contacts) < 2:
        return 0.0
    best = 0.0
    for first in contacts:
        for second in contacts:
            if first is second:
                continue
            first_position = np.asarray(first["position"], dtype=np.float64)
            second_position = np.asarray(second["position"], dtype=np.float64)
            direction = second_position - first_position
            direction_norm = np.linalg.norm(direction)
            if direction_norm <= 1e-8:
                continue
            direction /= direction_norm
            first_normal = np.asarray(first["normal"], dtype=np.float64)
            second_normal = np.asarray(second["normal"], dtype=np.float64)
            first_norm = np.linalg.norm(first_normal)
            second_norm = np.linalg.norm(second_normal)
            if min(first_norm, second_norm) <= 1e-8:
                continue
            first_alignment = -float(np.dot(first_normal / first_norm, direction))
            second_alignment = float(np.dot(second_normal / second_norm, direction))
            alignment = min(first_alignment, second_alignment)
            friction_margin = alignment + friction_coefficient * (1.0 - alignment)
            best = max(best, friction_margin if alignment > 0.0 else 0.0)
    return best


def _compute_contact_quality(contacts: list[dict[str, np.ndarray]], friction_coefficient: float) -> float:
    """Use wrench quality first, with an antipodal fallback for parallel jaws."""
    wrench_quality = compute_grasp_quality(contacts, friction_coefficient)
    if wrench_quality > 0.0:
        return wrench_quality
    return _antipodal_contact_quality(contacts, friction_coefficient)


# Pipelines call this with six positional arguments.
def score_grasp_poses_by_contacts(  # noqa: PLR0913,PLR0917
    grasp_poses: np.ndarray,
    object_point_cloud: np.ndarray,
    gripper_point_cloud: np.ndarray,
    friction_coefficient: float = FRICTION_COEFFICIENT,
    contact_clearance: float = CONTACT_CLEARANCE,
    min_quality_score: float = MIN_QUALITY_SCORE,
) -> list[tuple[np.ndarray, float]]:
    """Score grasp poses using analytical contacts and force-closure quality.

    Returns:
        ``(pose, score)`` pairs sorted by descending score. Poses with fewer
        than two contacts or scores below ``min_quality_score`` are omitted.
    """
    scored_grasps: list[tuple[np.ndarray, float]] = []
    for pose in grasp_poses:
        contacts = generate_analytical_contacts(
            object_point_cloud,
            gripper_point_cloud,
            pose,
            contact_clearance=contact_clearance,
        )
        if len(contacts) < MIN_GRASP_CONTACTS:
            continue
        score = _compute_contact_quality(contacts, friction_coefficient)
        if score >= min_quality_score:
            scored_grasps.append((pose, score))
    scored_grasps.sort(key=lambda item: item[1], reverse=True)
    return scored_grasps


def recompute_contact_scores(
    grasp_poses: np.ndarray,
    object_point_cloud: np.ndarray,
    gripper_point_cloud: np.ndarray,
    friction_coefficient: float = FRICTION_COEFFICIENT,
    contact_clearance: float = CONTACT_CLEARANCE,
) -> tuple[list[float], int]:
    """Recompute per-pose contact scores for auditing stored labels.

    Returns:
        A tuple of ``(scores, contact_scored_count)`` where ``contact_scored_count``
        counts poses with at least two analytical contacts.
    """
    recomputed_scores: list[float] = []
    contact_scored = 0
    for pose in grasp_poses:
        contacts = generate_analytical_contacts(
            object_point_cloud,
            gripper_point_cloud,
            pose,
            contact_clearance=contact_clearance,
        )
        if len(contacts) < MIN_GRASP_CONTACTS:
            continue
        contact_scored += 1
        recomputed_scores.append(_compute_contact_quality(contacts, friction_coefficient))
    return recomputed_scores, contact_scored
