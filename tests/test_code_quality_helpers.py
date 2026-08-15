from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from grasping_ai.evaluation.force_closure import (
    build_force_closure_judge,
    compute_grasp_quality,
    compute_grasp_wrench_matrix,
    load_contact_set,
    parse_contact_set,
)
from grasping_ai.models.rl_policy import (
    _sequential_linear_layers,
    build_policy_network,
    build_sb3_net_arch,
    copy_sb3_policy_weights,
)
from grasping_ai.pipelines.generate_grasps import load_generated_grasps
from grasping_ai.robotics.kinematics import robot_model_mj_model, robot_model_nq
from grasping_ai.training.checkpoint_io import read_checkpoint_model_state_dict


def test_parse_contact_set_accepts_list_dict_and_array() -> None:
    """Validate list, single-dict, and numpy-array contact payloads."""
    record = {"position": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0])}
    assert parse_contact_set([record]) == [record]
    assert parse_contact_set(record) == [record]
    assert parse_contact_set(np.array([record], dtype=object)) == [record]


def test_parse_contact_set_rejects_invalid_records() -> None:
    """Reject malformed contact records and payload types."""
    with pytest.raises(TypeError, match="must be a dictionary"):
        parse_contact_set(["not-a-dict"])
    with pytest.raises(TypeError, match="keys must be strings"):
        parse_contact_set([{1: np.zeros(3)}])
    with pytest.raises(TypeError, match="must be a numpy array"):
        parse_contact_set([{"position": [0.0, 0.0, 0.0]}])
    with pytest.raises(TypeError, match="list of contact records"):
        parse_contact_set(42)


def test_load_contact_set_rejects_invalid_file_payload(tmp_path: Path) -> None:
    """Surface parse failures when a contact file contains invalid data."""
    path = tmp_path / "bad_contacts.npy"
    np.save(path, np.array("not-contacts", dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="Failed to load contact set"):
        load_contact_set(path)


def test_load_generated_grasps_rejects_invalid_dict_and_array(tmp_path: Path) -> None:
    """Reject grasp dictionaries with invalid keys/values and non-array payloads."""
    bad_key_path = tmp_path / "bad_key.npy"
    np.save(bad_key_path, np.array({1: np.eye(4)[None]}, dtype=object), allow_pickle=True)
    with pytest.raises(TypeError, match="keys must be strings"):
        load_generated_grasps(bad_key_path)

    bad_value_path = tmp_path / "bad_value.npy"
    np.save(bad_value_path, np.array({"obj": [[1.0]]}, dtype=object), allow_pickle=True)
    with pytest.raises(TypeError, match="must be a numpy array"):
        load_generated_grasps(bad_value_path)

    bad_array_path = tmp_path / "bad_array.npy"
    np.save(bad_array_path, np.array("not-an-array", dtype=object), allow_pickle=True)
    with pytest.raises(TypeError, match="must contain a numpy array"):
        load_generated_grasps(bad_array_path)


def test_read_checkpoint_model_state_dict_filters_payload() -> None:
    """Return tensor state dict entries and None for missing or empty payloads."""
    assert read_checkpoint_model_state_dict({}) is None
    assert read_checkpoint_model_state_dict({"model_state_dict": "bad"}) is None
    assert read_checkpoint_model_state_dict({"model_state_dict": {}}) is None

    weight = torch.ones(2, 2)
    state = read_checkpoint_model_state_dict(
        {"model_state_dict": {"0.weight": weight, 1: weight, "bad": "x"}}
    )
    assert state == {"0.weight": weight}


def test_robot_model_accessors_validate_payload() -> None:
    """Validate typed accessors for robot model dictionaries."""
    assert robot_model_nq({"nq": 7}) == 7
    sentinel = object()
    assert robot_model_mj_model({"model": sentinel}) is sentinel

    with pytest.raises(TypeError, match="must be int"):
        robot_model_nq({"nq": 7.0})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must contain 'model'"):
        robot_model_mj_model({})


def test_build_sb3_net_arch_rejects_invalid_dims() -> None:
    """Reject non-positive SB3 architecture dimensions."""
    with pytest.raises(ValueError, match="hidden_dim"):
        build_sb3_net_arch(0, 2)
    with pytest.raises(ValueError, match="num_layers"):
        build_sb3_net_arch(64, 0)


def test_sequential_linear_layers_requires_sequential() -> None:
    """Reject non-Sequential modules when extracting linear layers."""
    with pytest.raises(TypeError, match=r"torch\.nn\.Sequential"):
        _sequential_linear_layers(torch.nn.Linear(2, 2))


def test_copy_sb3_policy_weights_validation_errors() -> None:
    """Reject invalid SB3/legacy policy modules during weight export."""
    legacy = build_policy_network(4, 2, 16, 2)
    with pytest.raises(TypeError, match="sb3_policy"):
        copy_sb3_policy_weights("bad", legacy)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="legacy_policy"):
        copy_sb3_policy_weights(torch.nn.Module(), "bad")  # type: ignore[arg-type]

    broken = torch.nn.Module()
    with pytest.raises(ValueError, match="mlp_extractor"):
        copy_sb3_policy_weights(broken, legacy)


def test_compute_grasp_wrench_matrix_skips_degenerate_contacts() -> None:
    """Skip contacts with zero normals or missing fields."""
    contacts = [
        {"position": np.zeros(3), "normal": np.zeros(3)},
        {"position": np.zeros(3)},
        {
            "position": np.array([0.0, 0.0, 0.0]),
            "normal": np.array([0.0, 0.0, 1.0]),
        },
    ]
    wrench = compute_grasp_wrench_matrix(contacts, friction_coefficient=0.5)
    assert wrench.shape[0] == 6
    assert wrench.shape[1] == 4


def test_force_closure_judge_handles_linprog_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return False when the force-closure LP solver raises."""
    def _raise(*args: object, **kwargs: object) -> None:
        raise RuntimeError("lp failed")  # noqa: TRY003

    monkeypatch.setattr("grasping_ai.evaluation.force_closure.linprog", _raise)
    judge = build_force_closure_judge(0.5, 1e-6)
    contacts = [
        {
            "position": np.array([0.0, 0.0, 0.0]),
            "normal": np.array([0.0, 0.0, 1.0]),
        },
        {
            "position": np.array([0.05, 0.0, 0.0]),
            "normal": np.array([0.0, 0.0, -1.0]),
        },
    ]
    assert judge(contacts) is False


def test_compute_grasp_quality_handles_solver_and_hull_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return zero quality when hull or LP fallback solvers fail."""
    contacts = [
        {
            "position": np.array([0.0, 0.0, 0.0]),
            "normal": np.array([1.0, 0.0, 0.0]),
        },
        {
            "position": np.array([0.0, 0.1, 0.0]),
            "normal": np.array([0.0, 1.0, 0.0]),
        },
        {
            "position": np.array([0.0, 0.0, 0.1]),
            "normal": np.array([0.0, 0.0, 1.0]),
        },
        {
            "position": np.array([0.1, 0.0, 0.0]),
            "normal": np.array([-1.0, 0.0, 0.0]),
        },
        {
            "position": np.array([0.0, -0.1, 0.0]),
            "normal": np.array([0.0, -1.0, 0.0]),
        },
        {
            "position": np.array([0.0, 0.0, -0.1]),
            "normal": np.array([0.0, 0.0, -1.0]),
        },
        {
            "position": np.array([0.05, 0.05, 0.05]),
            "normal": np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
        },
    ]

    monkeypatch.setattr(
        "grasping_ai.evaluation.force_closure.ConvexHull",
        MagicMock(side_effect=RuntimeError("hull failed")),
    )
    monkeypatch.setattr(
        "grasping_ai.evaluation.force_closure.linprog",
        MagicMock(side_effect=RuntimeError("lp failed")),
    )
    assert compute_grasp_quality(contacts, friction_coefficient=0.5) == 0.0

    tiny_contacts = [
        {"position": np.zeros(3), "normal": np.array([1e-12, 0.0, 0.0])},
    ] * 7
    assert compute_grasp_quality(tiny_contacts, friction_coefficient=0.5) == 0.0
