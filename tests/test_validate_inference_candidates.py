"""Tests for post-inference physical grasp validation."""
# ruff: noqa: S101, SLF001

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from grasping_ai.data.pointcloud_dataset import load_grasp_sample
from grasping_ai.pipelines import validate_inference_candidates as validator

if TYPE_CHECKING:
    from pathlib import Path


def _outcome(*, success: bool) -> dict[str, object]:
    return {
        "success": success,
        "ik_converged": True,
        "lift_ik_converged": True,
        "initial_height": 0.28,
        "final_height": 0.38 if success else 0.28,
        "contact_count": 2.0,
        "bilateral_contact": True,
        "stable": success,
        "contact_sustained": success,
        "initial_robot_object_collision_free": True,
        "table_collision_free": success,
        "fk_position_error": 0.0,
    }


def test_validates_exact_object_frame_candidates_and_preserves_rejections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-inference validation records one physics outcome for every raw pose."""
    candidates = np.stack((np.eye(4), np.eye(4)))
    candidates[1, 0, 3] = 0.02
    candidate_path = tmp_path / "candidates.npy"
    observation_path = tmp_path / "object.npy"
    output_path = tmp_path / "validated.npz"
    np.save(candidate_path, candidates)
    np.save(observation_path, np.zeros((8, 3), dtype=np.float32))

    object_to_world = np.eye(4)
    object_to_world[:3, 3] = [0.5, 0.0, 0.28]
    calls: list[np.ndarray] = []
    outcomes = iter((_outcome(success=True), _outcome(success=False)))
    monkeypatch.setattr(validator, "_placed_object_pose", lambda **_kwargs: object_to_world)
    monkeypatch.setattr(
        validator,
        "simulate_grasp",
        lambda grasp_pose, *_args, **_kwargs: (calls.append(grasp_pose.copy()) or next(outcomes)),
    )

    sample = validator.validate_inference_candidates(
        candidate_path=candidate_path,
        observation_path=observation_path,
        output_path=output_path,
        object_id="003_cracker_box",
        ycb_root=tmp_path,
        robot_xml=tmp_path / "robot.xml",
        table_xml=tmp_path / "table.xml",
        object_position=np.array([0.5, 0.0, 0.28]),
        num_simulation_steps=20,
        gripper_close_command=np.array([0.0]),
        gripper_width=0.08,
        lift_height_threshold=0.05,
        lift_distance=0.1,
        max_linear_velocity=0.05,
        max_angular_velocity=0.1,
        min_contacts=2.0,
    )

    assert np.allclose(calls[0], object_to_world @ candidates[0])
    assert np.allclose(calls[1], object_to_world @ candidates[1])
    assert np.array_equal(sample["sim_validated"], np.array([True, False]))
    assert np.array_equal(sample["candidate_indices"], np.array([0, 1], dtype=np.int32))
    assert sample["validation_failure_reasons"][1] == "robot intersects table"

    loaded = load_grasp_sample(output_path)
    assert np.array_equal(loaded["sim_validated"], np.array([True, False]))
    assert loaded["object_id"] == "003_cracker_box"
    assert loaded["grasp_pose_format"] == "object"


def test_rejects_nonrigid_raw_candidates(tmp_path: Path) -> None:
    """Invalid raw files are rejected before any MuJoCo scene is created."""
    path = tmp_path / "bad.npy"
    bad = np.eye(4).reshape(1, 4, 4)
    bad[0, 0, 0] = 2.0
    np.save(path, bad)

    with pytest.raises(ValueError, match="orthonormal"):
        validator._load_object_frame_candidates(path)
