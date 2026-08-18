"""Force-closure and antipodal grasp evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from scipy.optimize import linprog  # type: ignore[import-untyped]
from scipy.spatial import ConvexHull  # type: ignore[import-untyped]

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
from grasping_ai.utils.path_validation import require_path

ALIGNMENT_DOT_THRESHOLD = float(FLATTENED_YAML_CONFIG.get("tolerances.alignment_dot_threshold", 0.9))
HULL_HALFSPACE_EPS = float(FLATTENED_YAML_CONFIG.get("metrics.hull_halfspace_eps", 1e-9))
LP_FEASIBILITY_EPS = float(FLATTENED_YAML_CONFIG.get("metrics.lp_feasibility_eps", 1e-5))
MIN_WRENCH_COLUMNS = int(FLATTENED_YAML_CONFIG.get("wrench.min_columns", 6))
NORM_EPS = float(FLATTENED_YAML_CONFIG.get("tolerances.norm_eps", 1e-8))
WRENCH_DIM = int(FLATTENED_YAML_CONFIG.get("wrench.dim", 6))
WRENCH_LP_EQUALITY_ROWS = int(FLATTENED_YAML_CONFIG.get("wrench.lp_equality_rows", 7))

if TYPE_CHECKING:
    from pathlib import Path

ContactSet = list[dict[str, np.ndarray]]
ForceClosureJudge = Callable[[ContactSet], bool]


def _parse_contact_record(record: object) -> dict[str, np.ndarray]:
    """Validate one contact record from a serialized contact set.

    Args:
        record: Single contact entry loaded from disk.

    Returns:
        Mapping from string field names to ``numpy.ndarray`` values.

    Raises:
        TypeError: If ``record`` is not a dictionary or contains non-string
            keys or non-array values.
    """
    if not isinstance(record, dict):
        msg = "Each contact record must be a dictionary"
        raise TypeError(msg)
    parsed: dict[str, np.ndarray] = {}
    for key, value in record.items():
        if not isinstance(key, str):
            msg = "Contact record keys must be strings"
            raise TypeError(msg)
        if not isinstance(value, np.ndarray):
            msg = f"Contact record value for '{key}' must be a numpy array"
            raise TypeError(msg)
        parsed[key] = value
    return parsed


def parse_contact_set(loaded: object) -> ContactSet:
    """Validate a deserialized contact-set payload.

    Args:
        loaded: Raw object from ``numpy.load`` — a list of contact dicts, a
            single contact dict, or a numpy array convertible to a list.

    Returns:
        A list of validated contact records.

    Raises:
        TypeError: If ``loaded`` is not a list, dict, or numpy array, or if
            any contact record fails validation.
    """
    if isinstance(loaded, np.ndarray):
        loaded = loaded.tolist()
    if isinstance(loaded, dict):
        loaded = [loaded]
    if not isinstance(loaded, list):
        msg = "Contact set must deserialize to a list of contact records"
        raise TypeError(msg)
    return [_parse_contact_record(record) for record in loaded]


def load_contact_set(contact_path: Path) -> ContactSet:
    """Load a contact set from a serialized file.

    Args:
        contact_path: Path to a contact data file.

    Returns:
        A list of contact records, each describing contact points and normals.

    Raises:
        TypeError: If ``contact_path`` is not a ``pathlib.Path`` instance or
            the deserialized payload has invalid record types.
        FileNotFoundError: If the contact file does not exist.
        ValueError: If loading or parsing the file fails.
    """
    require_path(contact_path, "contact_path")
    if not contact_path.exists():
        msg = f"Contact file not found at: {contact_path}"
        raise FileNotFoundError(msg)

    try:
        data = np.load(contact_path, allow_pickle=True)
        loaded = data.item() if data.ndim == 0 else list(data)
        return parse_contact_set(loaded)
    except Exception as e:
        msg = f"Failed to load contact set: {e}"
        raise ValueError(msg) from e


def _solve_wrench_lp_margin(g_mat: np.ndarray, failure_log: str) -> float | None:
    """Solve the wrench-margin LP for a grasp wrench matrix.

    Maximize ``t`` subject to ``G @ alpha = 0``, ``sum(alpha) = 1``, and
    ``alpha_i >= t``. Variables: ``x = [alpha_0, ..., alpha_{m-1}, t]^T``.

    Args:
        g_mat: Grasp wrench matrix with shape ``(6, m)``.
        failure_log: Warning prefix logged when the LP raises.

    Returns:
        The optimal ``t`` value, or ``None`` when the LP fails or does not
        converge.
    """
    m = g_mat.shape[1]
    c = np.zeros(m + 1)
    c[-1] = -1.0  # We want to maximize t (minimize -t)

    # Equality constraints: G @ alpha = 0, sum(alpha) = 1
    a_eq = np.zeros((WRENCH_LP_EQUALITY_ROWS, m + 1))
    a_eq[:6, :m] = g_mat
    a_eq[6, :m] = 1.0
    b_eq = np.zeros(WRENCH_LP_EQUALITY_ROWS)
    b_eq[WRENCH_DIM] = 1.0

    # Inequality constraints: t - alpha_i <= 0 => -alpha_i + t <= 0
    a_ub = np.zeros((m, m + 1))
    a_ub[:, :m] = -np.eye(m)
    a_ub[:, -1] = 1.0
    b_ub = np.zeros(m)

    # Bounds: alpha_i >= 0, t can be negative
    bounds = [(0.0, None) for _ in range(m)] + [(None, None)]

    try:
        res = linprog(
            c,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
    # Any solver failure is reported as "no margin".
    except Exception as exc:  # noqa: BLE001
        logger.warning("{}: {}", failure_log, exc)
        return None
    if res.success:
        return float(res.x[-1])
    return None


def build_force_closure_judge(friction_coefficient: float, wrench_regularization: float) -> ForceClosureJudge:
    """Construct a callable force-closure judge for a contact set.

    Args:
        friction_coefficient: Coulomb friction coefficient for contact models.
        wrench_regularization: Regularization strength used to stabilize the
            underlying wrench-space analysis.

    Returns:
        A callable that maps a contact set to ``True`` when the grasp achieves
        force closure and ``False`` otherwise.
    """
    if friction_coefficient < 0:
        msg = "friction_coefficient must be non-negative"
        raise ValueError(msg)
    if wrench_regularization < 0:
        msg = "wrench_regularization must be non-negative"
        raise ValueError(msg)

    def judge(contact_set: ContactSet) -> bool:
        """Evaluate if the given contact set provides force closure.

        Args:
            contact_set: A list of contact records.

        Returns:
            True if the grasp provides force closure, False otherwise.
        """
        if not contact_set:
            return False

        # Compute grasp wrench matrix
        g_mat = compute_grasp_wrench_matrix(contact_set, friction_coefficient)
        # Add regularization columns to stabilize optimization/span check
        if wrench_regularization > 0:
            reg_eye = np.eye(6) * wrench_regularization
            g_mat = np.hstack([g_mat, reg_eye, -reg_eye])

        if g_mat.shape[1] < MIN_WRENCH_COLUMNS:
            return False

        # Check if rank is 6
        if np.linalg.matrix_rank(g_mat) < WRENCH_DIM:
            return False

        # Solve LP: Maximize t subject to G @ alpha = 0, sum(alpha) = 1, alpha_i >= t
        t_val = _solve_wrench_lp_margin(g_mat, "Force-closure LP failed")
        return t_val is not None and bool(t_val > LP_FEASIBILITY_EPS)

    return judge


def evaluate_force_closure(judge: ForceClosureJudge, contact_set: ContactSet) -> bool:
    """Evaluate whether a contact set provides force closure.

    Args:
        judge: Callable returned by ``build_force_closure_judge``.
        contact_set: Contact records describing the grasp.

    Returns:
        ``True`` if the grasp provides force closure, otherwise ``False``.
    """
    return judge(contact_set)


def compute_grasp_wrench_matrix(contact_set: ContactSet, friction_coefficient: float) -> np.ndarray:
    """Compute the grasp wrench matrix from a contact set.

    Args:
        contact_set: Contact records describing the grasp.
        friction_coefficient: Coulomb friction coefficient for the contacts.

    Returns:
        A ``(6, m)`` wrench matrix where ``m`` depends on the contact model.
    """
    if friction_coefficient < 0:
        msg = "friction_coefficient must be non-negative"
        raise ValueError(msg)

    wrenches = []
    for c in contact_set:
        pos = c.get("position")
        normal = c.get("normal")
        if pos is None or normal is None:
            continue

        pos = np.asarray(pos, dtype=np.float64)
        normal = np.asarray(normal, dtype=np.float64)

        # Normalize normal
        norm_val = np.linalg.norm(normal)
        if norm_val <= NORM_EPS:
            continue
        normal = normal / norm_val

        # Compute orthogonal tangents
        if np.abs(normal[0]) < ALIGNMENT_DOT_THRESHOLD:
            other = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            other = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        t1 = np.cross(normal, other)
        norm_t1 = np.linalg.norm(t1)
        if norm_t1 <= NORM_EPS:
            continue
        t1 = t1 / norm_t1

        t2 = np.cross(normal, t1)
        norm_t2 = np.linalg.norm(t2)
        if norm_t2 <= NORM_EPS:
            continue
        t2 = t2 / norm_t2

        # 4-sided pyramid approximation of friction cone
        forces = [
            normal + friction_coefficient * t1,
            normal - friction_coefficient * t1,
            normal + friction_coefficient * t2,
            normal - friction_coefficient * t2,
        ]

        for f in forces:
            # Build the 6D wrench from contact force and position.
            torque = np.cross(pos, f)
            wrench = np.concatenate([f, torque])
            wrenches.append(wrench)

    if not wrenches:
        return np.zeros((6, 0))

    return np.stack(wrenches, axis=1)


def compute_grasp_quality(contact_set: ContactSet, friction_coefficient: float) -> float:
    """Compute the standardized scalar grasp-quality metric for a contact set.

    Normalizes the grasp wrench matrix and measures the minimum distance from
    the origin to the boundary of the convex hull of normalized wrenches.
    Falls back to a linear programming-based margin if convex hull construction fails.

    Args:
        contact_set: Contact records describing the grasp.
        friction_coefficient: Friction coefficient used by force-closure analysis.

    Returns:
        A non-negative float representing the margin, or 0.0 if not force-closed.
    """
    if friction_coefficient < 0:
        msg = "friction_coefficient must be non-negative"
        raise ValueError(msg)

    if not contact_set:
        return 0.0

    g_mat = compute_grasp_wrench_matrix(contact_set, friction_coefficient)
    if g_mat.shape[1] < MIN_WRENCH_COLUMNS or np.linalg.matrix_rank(g_mat) < WRENCH_DIM:
        return 0.0

    # Normalize column vectors of g_mat by maximum finite column norm
    col_norms = np.linalg.norm(g_mat, axis=0)
    max_norm = np.max(col_norms)
    g_mat_normalized = g_mat / max_norm if max_norm > NORM_EPS else g_mat

    # Try convex hull in 6D
    if g_mat_normalized.shape[1] >= WRENCH_LP_EQUALITY_ROWS:
        try:
            hull = ConvexHull(g_mat_normalized.T)
            # check if origin is inside the hull
            if np.all(hull.equations[:, -1] <= HULL_HALFSPACE_EPS):
                # distance is -offset / norm(normal). Since normal has norm 1:
                return float(np.min(-hull.equations[:, -1]))
        # Degenerate hulls (QhullError) or bad input must fall back to the LP margin.
        except Exception as exc:  # noqa: BLE001
            logger.warning("Convex hull grasp-quality computation failed: {}", exc)

    # Fallback to LP formulation (similar to Ferrari-Canny sum-of-forces margin)
    t_val = _solve_wrench_lp_margin(g_mat_normalized, "Grasp-quality LP fallback failed")
    if t_val is None:
        return 0.0
    return max(0.0, t_val)
