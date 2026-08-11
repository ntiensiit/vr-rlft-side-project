import numpy as np

from grasping_ai.evaluation.force_closure import compute_grasp_quality


def test_grasp_quality_empty():
    assert compute_grasp_quality([], friction_coefficient=0.5) == 0.0

def test_grasp_quality_rank_deficient():
    # Only 1 contact, wrench matrix will be rank deficient (< 6)
    contacts = [
        {"position": np.array([0.0, 0.0, 0.0]), "normal": np.array([0.0, 0.0, 1.0])}
    ]
    assert compute_grasp_quality(contacts, friction_coefficient=0.5) == 0.0

def test_grasp_quality_valid_force_closure():
    # Setup 3 orthogonal contact points creating a valid force-closure grasp (e.g. 6 contacts/wrenches)
    # Let's define a block grasp or multiple contacts opposing each other
    contacts = [
        {"position": np.array([-0.05, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        {"position": np.array([0.05, 0.0, 0.0]), "normal": np.array([-1.0, 0.0, 0.0])},
        {"position": np.array([0.0, -0.05, 0.0]), "normal": np.array([0.0, 1.0, 0.0])},
        {"position": np.array([0.0, 0.05, 0.0]), "normal": np.array([0.0, -1.0, 0.0])},
        {"position": np.array([0.0, 0.0, -0.05]), "normal": np.array([0.0, 0.0, 1.0])},
        {"position": np.array([0.0, 0.0, 0.05]), "normal": np.array([0.0, 0.0, -1.0])},
    ]
    quality = compute_grasp_quality(contacts, friction_coefficient=0.5)
    assert quality > 0.0
    assert np.isfinite(quality)
