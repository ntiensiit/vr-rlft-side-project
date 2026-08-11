from collections.abc import Callable
from pathlib import Path

import numpy as np

ContactSet = list[dict[str, np.ndarray]]
ForceClosureJudge = Callable[[ContactSet], bool]


def load_contact_set(contact_path: Path) -> ContactSet:
    """Load a contact set from a serialized file.

    Args:
        contact_path: Path to a contact data file.

    Returns:
        A list of contact records, each describing contact points and normals.
    """
    raise NotImplementedError


def build_force_closure_judge(
    friction_coefficient: float, wrench_regularization: float
) -> ForceClosureJudge:
    """Construct a callable force-closure judge for a contact set.

    Args:
        friction_coefficient: Coulomb friction coefficient for contact models.
        wrench_regularization: Regularization strength used to stabilize the
            underlying wrench-space analysis.

    Returns:
        A callable that maps a contact set to ``True`` when the grasp achieves
        force closure and ``False`` otherwise.
    """
    raise NotImplementedError


def evaluate_force_closure(judge: ForceClosureJudge, contact_set: ContactSet) -> bool:
    """Evaluate whether a contact set provides force closure.

    Args:
        judge: Callable returned by ``build_force_closure_judge``.
        contact_set: Contact records describing the grasp.

    Returns:
        ``True`` if the grasp provides force closure, otherwise ``False``.
    """
    raise NotImplementedError


def compute_grasp_wrench_matrix(contact_set: ContactSet, friction_coefficient: float) -> np.ndarray:
    """Compute the grasp wrench matrix from a contact set.

    Args:
        contact_set: Contact records describing the grasp.
        friction_coefficient: Coulomb friction coefficient for the contacts.

    Returns:
        A ``(6, m)`` wrench matrix where ``m`` depends on the contact model.
    """
    raise NotImplementedError
