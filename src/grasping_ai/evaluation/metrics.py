from collections.abc import Callable

import numpy as np

StabilityJudge = Callable[[np.ndarray], bool]
LiftOutcomeJudge = Callable[[np.ndarray, np.ndarray], bool]


def build_stability_judge(
    max_linear_velocity: float, max_angular_velocity: float
) -> StabilityJudge:
    """Build a callable that judges whether an executed grasp is stable.

    Args:
        max_linear_velocity: Maximum acceptable linear velocity of the object.
        max_angular_velocity: Maximum acceptable angular velocity of the object.

    Returns:
        A callable that maps an object velocity vector to ``True`` when the
        grasp is stable and ``False`` otherwise.
    """
    raise NotImplementedError


def evaluate_stability(judge: StabilityJudge, object_velocity: np.ndarray) -> bool:
    """Evaluate whether an object velocity indicates a stable grasp.

    Args:
        judge: Callable returned by ``build_stability_judge``.
        object_velocity: Linear and angular velocity of the grasped object.

    Returns:
        ``True`` if the grasp is stable, otherwise ``False``.
    """
    raise NotImplementedError


def build_lift_outcome_judge(lift_height_threshold: float) -> LiftOutcomeJudge:
    """Build a callable that judges whether a grasp succeeded as a lift.

    Args:
        lift_height_threshold: Minimum world-frame height required to count
            the lift as successful.

    Returns:
        A callable that maps ``(initial_height, final_height)`` to ``True``
        when the lift is successful and ``False`` otherwise.
    """
    raise NotImplementedError


def evaluate_lift_success(
    judge: LiftOutcomeJudge, initial_height: float, final_height: float
) -> bool:
    """Evaluate whether a lift attempt succeeded.

    Args:
        judge: Callable returned by ``build_lift_outcome_judge``.
        initial_height: Object height before the lift attempt.
        final_height: Object height after the lift attempt.

    Returns:
        ``True`` if the lift succeeded, otherwise ``False``.
    """
    raise NotImplementedError


def aggregate_grasp_success_rate(per_object_success: dict[str, bool]) -> float:
    """Aggregate per-object grasp success flags into an overall success rate.

    Args:
        per_object_success: Mapping from object identifier to success flag.

    Returns:
        Fraction of objects for which the grasp attempt succeeded.
    """
    raise NotImplementedError
