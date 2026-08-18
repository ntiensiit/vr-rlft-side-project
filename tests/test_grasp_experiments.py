"""Regression tests for validated grasp experiment inputs and contact metrics."""
# ruff: noqa: S101, SLF001

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grasping_ai.pipelines import grasp_experiments


def _sample(*, validation: np.ndarray | None) -> dict[str, object]:
    sample: dict[str, object] = {
        "object_id": "003_cracker_box",
        "grasp_poses": np.stack((np.eye(4), np.eye(4))),
        "grasp_pose_format": "object",
    }
    if validation is not None:
        sample["sim_validated"] = validation
    return sample


def test_load_candidate_requires_simulation_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pose baselines must not accept archives with no physical-validation field."""
    monkeypatch.setattr(
        grasp_experiments,
        "load_grasp_sample",
        lambda _path: _sample(validation=None),
    )

    with pytest.raises(ValueError, match="missing sim_validated"):
        grasp_experiments._load_candidate(Path("candidate.npz"), "003_cracker_box", 0)


def test_load_candidate_rejects_invalid_validation_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each candidate must have exactly one validation result."""
    monkeypatch.setattr(
        grasp_experiments,
        "load_grasp_sample",
        lambda _path: _sample(validation=np.array([True])),
    )

    with pytest.raises(ValueError, match="does not match grasp candidate count"):
        grasp_experiments._load_candidate(Path("candidate.npz"), "003_cracker_box", 0)


def test_sustained_contact_requires_consecutive_physics_steps() -> None:
    """A one-frame touch is not reported as a sustained grasp."""
    assert not grasp_experiments._has_sustained_bilateral_contact(1)
    assert grasp_experiments._has_sustained_bilateral_contact(2)
