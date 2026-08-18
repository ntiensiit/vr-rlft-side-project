"""Aggregate evaluation metrics for grasp pipelines."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG

WRENCH_DIM = int(FLATTENED_YAML_CONFIG.get("wrench.dim", 6))

StabilityJudge = Callable[[np.ndarray], bool]
LiftOutcomeJudge = Callable[[float, float], bool]


def build_stability_judge(max_linear_velocity: float, max_angular_velocity: float) -> StabilityJudge:
    """Build a callable that judges whether an executed grasp is stable.

    Args:
        max_linear_velocity: Maximum acceptable linear velocity of the object.
        max_angular_velocity: Maximum acceptable angular velocity of the object.

    Returns:
        A callable that maps an object velocity vector to ``True`` when the
        grasp is stable and ``False`` otherwise.
    """
    if max_linear_velocity < 0:
        msg = "max_linear_velocity must be non-negative"
        raise ValueError(msg)
    if max_angular_velocity < 0:
        msg = "max_angular_velocity must be non-negative"
        raise ValueError(msg)

    def judge(object_velocity: np.ndarray) -> bool:
        """Evaluate if the object velocity is within stability thresholds.

        Args:
            object_velocity: A 6D array containing linear and angular velocity.

        Returns:
            True if the grasp is considered stable, False otherwise.
        """
        if not isinstance(object_velocity, np.ndarray):
            msg = "object_velocity must be a numpy array"
            raise TypeError(msg)
        flat_vel = object_velocity.ravel()
        if flat_vel.shape[0] != WRENCH_DIM:
            msg = f"object_velocity must represent 6D velocity, got shape {object_velocity.shape}"
            raise ValueError(msg)

        lin_vel = flat_vel[:3]
        ang_vel = flat_vel[3:]

        lin_speed = float(np.linalg.norm(lin_vel))
        ang_speed = float(np.linalg.norm(ang_vel))

        return lin_speed <= max_linear_velocity and ang_speed <= max_angular_velocity

    return judge


def evaluate_stability(judge: StabilityJudge, object_velocity: np.ndarray) -> bool:
    """Evaluate whether an object velocity indicates a stable grasp.

    Args:
        judge: Callable returned by ``build_stability_judge``.
        object_velocity: Linear and angular velocity of the grasped object.

    Returns:
        ``True`` if the grasp is stable, otherwise ``False``.
    """
    return judge(object_velocity)


def build_lift_outcome_judge(lift_height_threshold: float) -> LiftOutcomeJudge:
    """Build a callable that judges whether a grasp succeeded as a lift.

    Args:
        lift_height_threshold: Minimum world-frame height required to count
            the lift as successful.

    Returns:
        A callable that maps ``(initial_height, final_height)`` to ``True``
        when the lift is successful and ``False`` otherwise.
    """
    if lift_height_threshold < 0:
        msg = "lift_height_threshold must be non-negative"
        raise ValueError(msg)

    def judge(initial_height: float, final_height: float) -> bool:
        """Evaluate if the lift attempt achieved the required height.

        Args:
            initial_height: The starting height of the object.
            final_height: The final height of the object after lifting.

        Returns:
            True if the object was lifted by at least the required threshold, False otherwise.
        """
        if not isinstance(initial_height, (int, float, np.floating, np.integer)):
            msg = "initial_height must be a number"
            raise TypeError(msg)
        if not isinstance(final_height, (int, float, np.floating, np.integer)):
            msg = "final_height must be a number"
            raise TypeError(msg)

        lift_dist = float(final_height - initial_height)
        return lift_dist >= lift_height_threshold

    return judge


def evaluate_lift_success(judge: LiftOutcomeJudge, initial_height: float, final_height: float) -> bool:
    """Evaluate whether a lift attempt succeeded.

    Args:
        judge: Callable returned by ``build_lift_outcome_judge``.
        initial_height: Object height before the lift attempt.
        final_height: Object height after the lift attempt.

    Returns:
        ``True`` if the lift succeeded, otherwise ``False``.
    """
    return judge(initial_height, final_height)


def aggregate_grasp_success_rate(per_object_success: dict[str, bool]) -> float:
    """Aggregate per-object grasp success flags into an overall success rate.

    Args:
        per_object_success: Mapping from object identifier to success flag.

    Returns:
        Fraction of objects for which the grasp attempt succeeded.
    """
    if not isinstance(per_object_success, dict):
        msg = "per_object_success must be a dictionary"
        raise TypeError(msg)
    if not per_object_success:
        return 0.0

    success_count = sum(1 for success in per_object_success.values() if success)
    return float(success_count / len(per_object_success))
