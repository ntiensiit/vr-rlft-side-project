from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
from scipy.optimize import linprog  # type: ignore[import-untyped]
from scipy.spatial import ConvexHull  # type: ignore[import-untyped]

ContactSet = list[dict[str, np.ndarray]]
ForceClosureJudge = Callable[[ContactSet], bool]


def load_contact_set(contact_path: Path) -> ContactSet:
    """Load a contact set from a serialized file.

    Args:
        contact_path: Path to a contact data file.

    Returns:
        A list of contact records, each describing contact points and normals.
    """
    if not isinstance(contact_path, Path):
        raise TypeError("contact_path must be a pathlib.Path instance")
    if not contact_path.exists():
        raise FileNotFoundError(f"Contact file not found at: {contact_path}")

    try:
        data = np.load(contact_path, allow_pickle=True)
        if hasattr(data, "item"):
            loaded = data.item()
        else:
            loaded = list(data)
        return cast(ContactSet, loaded)
    except Exception as e:
        raise ValueError(f"Failed to load contact set: {e}") from e


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
        raise ValueError("friction_coefficient must be non-negative")
    if wrench_regularization < 0:
        raise ValueError("wrench_regularization must be non-negative")

    def judge(contact_set: ContactSet) -> bool:
        if not contact_set:
            return False

        # Compute grasp wrench matrix
        g_mat = compute_grasp_wrench_matrix(contact_set, friction_coefficient)
        # Add regularization columns to stabilize optimization/span check
        if wrench_regularization > 0:
            reg_eye = np.eye(6) * wrench_regularization
            g_mat = np.hstack([g_mat, reg_eye, -reg_eye])

        if g_mat.shape[1] < 6:
            return False

        # Check if rank is 6
        if np.linalg.matrix_rank(g_mat) < 6:
            return False

        # Solve LP: Maximize t subject to G @ alpha = 0, sum(alpha) = 1, alpha_i >= t
        # Variables: x = [alpha_0, ..., alpha_{m-1}, t]^T
        m = g_mat.shape[1]
        c = np.zeros(m + 1)
        c[-1] = -1.0  # We want to maximize t (minimize -t)

        # Equality constraints: G @ alpha = 0, sum(alpha) = 1
        a_eq = np.zeros((7, m + 1))
        a_eq[:6, :m] = g_mat
        a_eq[6, :m] = 1.0
        b_eq = np.zeros(7)
        b_eq[6] = 1.0

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
            if res.success:
                t_val = res.x[-1]
                return bool(t_val > 1e-5)
            return False
        except Exception:
            return False

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
        raise ValueError("friction_coefficient must be non-negative")

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
        if norm_val <= 1e-8:
            continue
        normal = normal / norm_val

        # Compute orthogonal tangents
        if np.abs(normal[0]) < 0.9:
            other = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            other = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        t1 = np.cross(normal, other)
        norm_t1 = np.linalg.norm(t1)
        if norm_t1 <= 1e-8:
            continue
        t1 = t1 / norm_t1

        t2 = np.cross(normal, t1)
        norm_t2 = np.linalg.norm(t2)
        if norm_t2 <= 1e-8:
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
            # Wrench = [force, torque]
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
        raise ValueError("friction_coefficient must be non-negative")

    if not contact_set:
        return 0.0

    g_mat = compute_grasp_wrench_matrix(contact_set, friction_coefficient)
    if g_mat.shape[1] < 6 or np.linalg.matrix_rank(g_mat) < 6:
        return 0.0

    # Normalize column vectors of g_mat by maximum finite column norm
    col_norms = np.linalg.norm(g_mat, axis=0)
    max_norm = np.max(col_norms)
    if max_norm > 1e-8:
        g_mat_normalized = g_mat / max_norm
    else:
        g_mat_normalized = g_mat

    # Try convex hull in 6D
    if g_mat_normalized.shape[1] >= 7:
        try:
            hull = ConvexHull(g_mat_normalized.T)
            # check if origin is inside the hull
            if np.all(hull.equations[:, -1] <= 1e-9):
                # distance is -offset / norm(normal). Since normal has norm 1:
                return float(np.min(-hull.equations[:, -1]))
        except Exception:
            pass

    # Fallback to LP formulation (similar to Ferrari-Canny sum-of-forces margin)
    m = g_mat_normalized.shape[1]
    c = np.zeros(m + 1)
    c[-1] = -1.0  # Maximize t (minimize -t)

    a_eq = np.zeros((7, m + 1))
    a_eq[:6, :m] = g_mat_normalized
    a_eq[6, :m] = 1.0
    b_eq = np.zeros(7)
    b_eq[6] = 1.0

    a_ub = np.zeros((m, m + 1))
    a_ub[:, :m] = -np.eye(m)
    a_ub[:, -1] = 1.0
    b_ub = np.zeros(m)

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
        if res.success:
            t_val = res.x[-1]
            return max(0.0, float(t_val))
    except Exception:
        pass

    return 0.0
