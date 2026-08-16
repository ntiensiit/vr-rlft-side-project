"""Phase 10 evaluation pipeline tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import scipy.optimize
import scipy.spatial

if TYPE_CHECKING:
    from collections.abc import Iterator

from grasping_ai.evaluation.collision import (
    build_collision_checker,
    check_collision,
    filter_collision_free_grasps,
    generate_analytical_contacts,
)
from grasping_ai.evaluation.force_closure import (
    build_force_closure_judge,
    compute_grasp_quality,
    compute_grasp_wrench_matrix,
    evaluate_force_closure,
    load_contact_set,
)
from grasping_ai.evaluation.metrics import (
    aggregate_grasp_success_rate,
    build_lift_outcome_judge,
    build_stability_judge,
    evaluate_lift_success,
    evaluate_stability,
)
from grasping_ai.pipelines.evaluate import (
    aggregate_evaluation_results,
    evaluate_generated_grasps,
    read_jsonl_records,
    write_evaluation_report,
    write_jsonl_records,
)
from grasping_ai.pipelines.generate_grasps import load_generated_grasps

EXPECTED_PAIR_COUNT = 2
EXPECTED_HALF = 0.5
EXPECTED_HIGH_METRIC = 0.95
FAKE_MAX_MIN_LEN = 8


@pytest.fixture
def temp_files() -> Iterator[dict[str, Path]]:
    """Fixture providing paths to temporary point clouds, grasp arrays, and dictionary files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Object and gripper point clouds
        obj_pc = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        grip_pc = np.array([[0.0, 0.0, 0.002]], dtype=np.float32)

        obj_path = tmp_path / "object_pc.npy"
        grip_path = tmp_path / "gripper_pc.npy"

        np.save(obj_path, obj_pc)
        np.save(grip_path, grip_pc)

        # Plain grasps array
        grasps_arr = np.stack([np.eye(4), np.eye(4)], axis=0)
        grasps_arr_path = tmp_path / "grasps_arr.npy"
        np.save(grasps_arr_path, grasps_arr)

        # Dictionary grasps payload
        grasps_dict = {"mustard_bottle": grasps_arr}
        grasps_dict_path = tmp_path / "grasps_dict.npy"
        np.save(grasps_dict_path, grasps_dict, allow_pickle=True)

        report_path = tmp_path / "report.jsonl"

        yield {
            "obj_path": obj_path,
            "grip_path": grip_path,
            "grasps_arr_path": grasps_arr_path,
            "grasps_dict_path": grasps_dict_path,
            "report_path": report_path,
        }


def test_analytical_contacts_valid() -> None:
    """Verify contact positions and normals for gripper points within clearance."""
    # Setup simple geometry
    # Object is a single point at origin
    object_pc = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    # Gripper has two points
    gripper_pc = np.array(
        [
            [0.0, 0.0, 0.002],
            [0.0, 0.0, 0.010],
        ],
        dtype=np.float32,
    )

    # Identity pose
    grasp_pose = np.eye(4, dtype=np.float32)

    # Clearance is 0.005. Only the first gripper point is within clearance (dist=0.002)
    contacts = generate_analytical_contacts(object_pc, gripper_pc, grasp_pose, contact_clearance=0.005)

    if not (len(contacts) == 1):
        raise AssertionError
    if not (np.allclose(contacts[0]["position"], np.array([0.0, 0.0, 0.0]))):
        raise AssertionError
    # The contact normal points from the gripper sample toward the object origin.
    if not (np.allclose(contacts[0]["normal"], np.array([0.0, 0.0, -1.0]))):
        raise AssertionError


def test_analytical_contacts_rejection() -> None:
    """Verify that generate_analytical_contacts yields zero contacts when geometry points are too far apart."""
    object_pc = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    gripper_pc = np.array([[0.0, 0.0, 0.1]], dtype=np.float32)
    grasp_pose = np.eye(4, dtype=np.float32)

    # Too far, should yield 0 contacts
    contacts = generate_analytical_contacts(object_pc, gripper_pc, grasp_pose, contact_clearance=0.005)
    if not (len(contacts) == 0):
        raise AssertionError


def test_analytical_contacts_invalid_shapes() -> None:
    """Verify that generate_analytical_contacts raises ValueError for dimension or shape mismatches."""
    object_pc = np.array([[0.0, 0.0]], dtype=np.float32)
    gripper_pc = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    grasp_pose = np.eye(4, dtype=np.float32)

    with pytest.raises(ValueError, match="object_point_cloud"):
        generate_analytical_contacts(object_pc, gripper_pc, grasp_pose, 0.005)

    object_pc = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    grasp_pose = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match="grasp_pose"):
        generate_analytical_contacts(object_pc, gripper_pc, grasp_pose, 0.005)


def test_grasp_quality_empty() -> None:
    """Verify that compute_grasp_quality returns zero for empty contact lists."""
    if not (compute_grasp_quality([], friction_coefficient=0.5) == 0.0):
        raise AssertionError


def test_grasp_quality_rank_deficient() -> None:
    """Verify that compute_grasp_quality returns zero for rank-deficient wrench matrices (e.g. single contact point)."""
    # Only 1 contact, wrench matrix will be rank deficient (< 6)
    contacts = [{"position": np.array([0.0, 0.0, 0.0]), "normal": np.array([0.0, 0.0, 1.0])}]
    if not (compute_grasp_quality(contacts, friction_coefficient=0.5) == 0.0):
        raise AssertionError


def test_grasp_quality_valid_force_closure() -> None:
    """Verify that compute_grasp_quality calculates positive quality for valid force-closure contacts."""
    # Setup 3 orthogonal contact points creating a valid force-closure grasp (e.g. 6 contacts/wrenches)
    # Let's define a block grasp or multiple contacts opposing each other
    contacts = [
        {"position": np.array([0.05, 0.0, 0.0]), "normal": np.array([-1.0, 0.0, 0.0])},
        {"position": np.array([-0.05, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        {"position": np.array([0.0, 0.05, 0.0]), "normal": np.array([0.0, -1.0, 0.0])},
        {"position": np.array([0.0, -0.05, 0.0]), "normal": np.array([0.0, 1.0, 0.0])},
    ]

    q = compute_grasp_quality(contacts, friction_coefficient=0.5)
    if not (q > 0.0):
        raise AssertionError


def test_evaluate_generated_grasps_analytical() -> None:
    """Verify that evaluate_generated_grasps correctly runs analytical evaluations on batch grasp inputs."""
    # Setup minimal point clouds
    object_pc = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    gripper_pc = np.array(
        [
            [0.0, 0.0, 0.002],
            [0.0, 0.0, -0.002],
        ],
        dtype=np.float32,
    )

    # Identitiy pose -> contacts will be at (0,0,0.002) and (0,0,-0.002) with opposing normals
    grasp_poses = np.eye(4, dtype=np.float32)[None, ...]

    evals = evaluate_generated_grasps(
        grasp_poses,
        object_pc,
        gripper_pc,
        friction_coefficient=0.5,
        clearance=0.002,
    )

    if not (len(evals) == 1):
        raise AssertionError
    if evals[0]["collision_free"] is not True:
        raise AssertionError
    if evals[0]["force_closure"] is not True:
        raise AssertionError
    if evals[0]["grasp_success"] is not True:
        raise AssertionError


def test_aggregate_evaluation_results() -> None:
    """Verify that aggregate_evaluation_results computes correct statistics over multiple objects and runs."""
    per_obj = {
        "bottle": [
            {
                "collision_free": True,
                "force_closure": True,
                "grasp_success": True,
                "grasp_quality": 0.5,
            },
            {
                "collision_free": False,
                "force_closure": False,
                "grasp_success": False,
                "grasp_quality": 0.0,
            },
        ],
        "can": [
            {
                "collision_free": True,
                "force_closure": False,
                "grasp_success": False,
                "grasp_quality": 0.0,
            },
        ],
    }

    report = aggregate_evaluation_results(per_obj)
    if not (report["success_rate"] == pytest.approx(1.0 / 3.0)):
        raise AssertionError
    if not (report["force_closure_rate"] == pytest.approx(1.0 / 3.0)):
        raise AssertionError
    if not (report["collision_free_rate"] == pytest.approx(2.0 / 3.0)):
        raise AssertionError
    if not (report["mean_grasp_quality"] == pytest.approx(0.5 / 3.0)):
        raise AssertionError


def test_evaluate_script_plain_array(temp_files: dict[str, Path]) -> None:
    """Verify that evaluate_generated_grasps can load and evaluate plain numpy array grasp formats."""
    grasps = load_generated_grasps(temp_files["grasps_arr_path"], object_key="default")
    object_point_cloud = np.load(temp_files["obj_path"])
    gripper_point_cloud = np.load(temp_files["grip_path"])

    evals = evaluate_generated_grasps(
        grasps,
        object_point_cloud,
        gripper_point_cloud,
        clearance=0.002,
    )
    if not (len(evals) == EXPECTED_PAIR_COUNT):
        raise AssertionError

    # Write report
    report = aggregate_evaluation_results({"default": evals})
    write_evaluation_report(temp_files["report_path"], report)
    if not (temp_files["report_path"].is_file()):
        raise AssertionError


def test_evaluate_script_dictionary(temp_files: dict[str, Path]) -> None:
    """Verify that evaluate_generated_grasps can load and evaluate pickled dictionary grasp formats."""
    grasps = load_generated_grasps(temp_files["grasps_dict_path"], object_key="mustard_bottle")
    object_point_cloud = np.load(temp_files["obj_path"])
    gripper_point_cloud = np.load(temp_files["grip_path"])

    evals = evaluate_generated_grasps(
        grasps,
        object_point_cloud,
        gripper_point_cloud,
        clearance=0.002,
    )
    if not (len(evals) == EXPECTED_PAIR_COUNT):
        raise AssertionError

    results = {"mustard_bottle": evals}
    report = aggregate_evaluation_results(results)
    if "mean_grasp_quality" not in report:
        raise AssertionError


def test_force_closure_load_contact_set_validations(tmp_path: Path) -> None:
    """Verify validations and type/value exceptions for loading contact set file payloads."""
    with pytest.raises(TypeError, match="contact_path must be"):
        load_contact_set("not_a_path")  # type: ignore[arg-type]

    non_existent = tmp_path / "missing.npy"
    with pytest.raises(FileNotFoundError, match="Contact file not found"):
        load_contact_set(non_existent)

    contacts = [{"position": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0])}]
    payload = np.empty((), dtype=object)
    payload[()] = contacts
    valid_file = tmp_path / "contacts.npy"
    np.save(valid_file, payload, allow_pickle=True)
    loaded = load_contact_set(valid_file)
    if not (len(loaded) == 1):
        raise AssertionError

    corrupted_file = tmp_path / "corrupted.npy"
    corrupted_file.write_bytes(b"invalid data")
    with pytest.raises(ValueError, match="Failed to load contact set"):
        load_contact_set(corrupted_file)


def test_force_closure_judge_and_quality_validations() -> None:
    """Verify validation boundaries and force closure computations under various contact normals."""
    with pytest.raises(ValueError, match="friction_coefficient must be non-negative"):
        build_force_closure_judge(-0.1, 0.01)

    with pytest.raises(ValueError, match="wrench_regularization must be non-negative"):
        build_force_closure_judge(0.5, -0.01)

    judge = build_force_closure_judge(0.5, 0.01)
    if judge([]):
        raise AssertionError
    if evaluate_force_closure(judge, []):
        raise AssertionError

    contacts_x_normal = [
        {"position": np.array([-0.05, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        {"position": np.array([0.05, 0.0, 0.0]), "normal": np.array([-1.0, 0.0, 0.0])},
        {"position": np.array([0.0, -0.05, 0.0]), "normal": np.array([0.0, 1.0, 0.0])},
        {"position": np.array([0.0, 0.05, 0.0]), "normal": np.array([0.0, -1.0, 0.0])},
        {"position": np.array([0.0, 0.0, -0.05]), "normal": np.array([0.0, 0.0, 1.0])},
        {"position": np.array([0.0, 0.0, 0.05]), "normal": np.array([0.0, 0.0, -1.0])},
    ]

    with pytest.raises(ValueError, match="friction_coefficient must be non-negative"):
        compute_grasp_quality(contacts_x_normal, -0.5)

    q = compute_grasp_quality(contacts_x_normal, 0.5)
    if not (q > 0.0):
        raise AssertionError


def test_metrics_validations() -> None:
    """Verify stability and lift outcome judge validations for dimensions, non-negative bounds, and types."""
    with pytest.raises(ValueError, match="max_linear_velocity must be non-negative"):
        build_stability_judge(-1.0, 1.0)
    with pytest.raises(ValueError, match="max_angular_velocity must be non-negative"):
        build_stability_judge(1.0, -1.0)

    stab_judge = build_stability_judge(1.0, 1.0)
    with pytest.raises(TypeError, match="object_velocity must be a numpy array"):
        evaluate_stability(stab_judge, [0.0] * 6)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="6D velocity"):
        evaluate_stability(stab_judge, np.zeros(3))

    if not (evaluate_stability(stab_judge, np.zeros(6))):
        raise AssertionError
    if evaluate_stability(stab_judge, np.ones(6) * 10.0):
        raise AssertionError

    with pytest.raises(ValueError, match="lift_height_threshold must be non-negative"):
        build_lift_outcome_judge(-0.1)

    lift_judge = build_lift_outcome_judge(0.05)
    with pytest.raises(TypeError, match="initial_height must be a number"):
        evaluate_lift_success(lift_judge, "invalid", 0.1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="final_height must be a number"):
        evaluate_lift_success(lift_judge, 0.0, "invalid")  # type: ignore[arg-type]

    if not (evaluate_lift_success(lift_judge, 0.0, 0.06)):
        raise AssertionError
    if evaluate_lift_success(lift_judge, 0.0, 0.02):
        raise AssertionError

    with pytest.raises(TypeError, match="per_object_success must be a dictionary"):
        aggregate_grasp_success_rate("not_a_dict")  # type: ignore[arg-type]
    if not (aggregate_grasp_success_rate({}) == 0.0):
        raise AssertionError
    if not (aggregate_grasp_success_rate({"obj1": True, "obj2": False}) == EXPECTED_HALF):
        raise AssertionError


def test_collision_and_evaluate_pipeline_validations() -> None:
    """Verify that collision checking and grasp evaluation functions validate shapes, boundaries, and clear ranges."""
    with pytest.raises(ValueError, match="clearance must be non-negative"):
        build_collision_checker(np.zeros((10, 3)), np.zeros((5, 3)), -0.05)

    obj_pc = np.zeros((10, 3))
    grip_pc = np.full((5, 3), 10.0)
    checker = build_collision_checker(obj_pc, grip_pc, 0.005)
    with pytest.raises(ValueError, match="grasp_pose must have shape"):
        check_collision(checker, np.zeros((3, 3)))

    with pytest.raises(ValueError, match="grasp_poses must have shape"):
        filter_collision_free_grasps(checker, np.zeros((10, 2)))

    with pytest.raises(ValueError, match="grasp_poses must have shape"):
        filter_collision_free_grasps(checker, np.zeros((2, 3, 4)))

    with pytest.raises(ValueError, match="grasp_poses must have shape"):
        evaluate_generated_grasps(
            np.zeros((2, 3)),
            np.zeros((10, 3)),
            np.zeros((5, 3)),
        )

    with pytest.raises(ValueError, match="grasp_poses must have shape"):
        evaluate_generated_grasps(
            np.zeros((1, 3, 4)),
            np.zeros((10, 3)),
            np.zeros((5, 3)),
        )

    empty_filtered = filter_collision_free_grasps(checker, np.zeros((0, 4, 4)))
    if not (empty_filtered.shape == (0, 4, 4)):
        raise AssertionError

    res_2d = filter_collision_free_grasps(checker, np.eye(4))
    if not (res_2d.shape == (1, 4, 4)):
        raise AssertionError

    empty_evals = evaluate_generated_grasps(
        grasp_poses=np.zeros((0, 4, 4)),
        object_point_cloud=np.zeros((10, 3)),
        gripper_point_cloud=np.zeros((5, 3)),
    )
    if not (len(empty_evals) == 0):
        raise AssertionError

    if not (aggregate_evaluation_results({})["success_rate"] == 0.0):
        raise AssertionError


def test_force_closure_additional_branches() -> None:
    """Verify additional force closure logic branches including incomplete contact structures and zero contact lists."""
    incomplete_contacts = [{"position": np.zeros(3)}, {"normal": np.zeros(3)}]
    w_matrix = compute_grasp_wrench_matrix(incomplete_contacts, 0.5)
    if not (w_matrix.shape == (6, 0)):
        raise AssertionError

    zero_contacts = [{"position": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0])} for _ in range(7)]
    q_zero = compute_grasp_quality(zero_contacts, 0.5)
    if not (q_zero == 0.0):
        raise AssertionError


def test_collision_additional_branches() -> None:
    """Verify validation checks on input geometries, non-finite arrays, and invalid contact clearance boundaries."""
    with pytest.raises(ValueError, match="object_point_cloud"):
        build_collision_checker(np.zeros(3), np.zeros((5, 3)), 0.05)

    with pytest.raises(ValueError, match="gripper_point_cloud"):
        build_collision_checker(np.zeros((10, 3)), np.zeros(3), 0.05)

    with pytest.raises(ValueError, match="object_point_cloud"):
        generate_analytical_contacts("not_array", np.zeros((5, 3)), np.eye(4), 0.05)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="gripper_point_cloud"):
        generate_analytical_contacts(np.zeros((10, 3)), "not_array", np.eye(4), 0.05)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="grasp_pose"):
        generate_analytical_contacts(np.zeros((10, 3)), np.zeros((5, 3)), np.zeros(3), 0.05)

    with pytest.raises(ValueError, match="finite"):
        generate_analytical_contacts(np.full((10, 3), np.nan), np.zeros((5, 3)), np.eye(4), 0.05)

    with pytest.raises(ValueError, match="finite"):
        generate_analytical_contacts(np.zeros((10, 3)), np.full((5, 3), np.inf), np.eye(4), 0.05)

    with pytest.raises(ValueError, match="finite"):
        generate_analytical_contacts(np.zeros((10, 3)), np.zeros((5, 3)), np.full((4, 4), np.nan), 0.05)

    with pytest.raises(ValueError, match="contact_clearance"):
        generate_analytical_contacts(np.zeros((10, 3)), np.zeros((5, 3)), np.eye(4), -0.01)

    if not (generate_analytical_contacts(np.zeros((0, 3)), np.zeros((5, 3)), np.eye(4), 0.05) == []):
        raise AssertionError

    contacts_overlap = generate_analytical_contacts(np.zeros((1, 3)), np.zeros((1, 3)), np.eye(4), 0.05)
    if not (len(contacts_overlap) == 1):
        raise AssertionError
    if not (np.allclose(contacts_overlap[0]["normal"], [0.0, 0.0, 1.0])):
        raise AssertionError

    colliding_checker = build_collision_checker(np.zeros((10, 3)), np.zeros((5, 3)), 0.05)
    if not (filter_collision_free_grasps(colliding_checker, np.eye(4)).shape == (0, 4, 4)):
        raise AssertionError
    if not (filter_collision_free_grasps(colliding_checker, np.stack([np.eye(4), np.eye(4)])).shape == (0, 4, 4)):
        raise AssertionError


def test_evaluate_pipeline_additional_branches(tmp_path: Path) -> None:
    """Verify custom contact providers and file path arguments inside the evaluation pipeline."""
    contacts = [{"position": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0])}]
    payload = np.empty((), dtype=object)
    payload[()] = contacts
    contact_file = tmp_path / "contacts.npy"
    np.save(contact_file, payload, allow_pickle=True)

    evals_from_path = evaluate_generated_grasps(
        grasp_poses=np.eye(4),
        object_point_cloud=np.zeros((10, 3)),
        gripper_point_cloud=np.zeros((5, 3)),
        contact_path=contact_file,
        filter_collisions=True,
    )
    if not (len(evals_from_path) == 0):
        raise AssertionError

    custom_evals = evaluate_generated_grasps(
        grasp_poses=np.eye(4),
        object_point_cloud=np.full((10, 3), 10.0),
        gripper_point_cloud=np.zeros((5, 3)),
        contact_set_provider=lambda _pose: [],
    )
    if not (len(custom_evals) == 1):
        raise AssertionError
    if custom_evals[0]["force_closure"]:
        raise AssertionError

    with pytest.raises(TypeError, match="per_object_results must be a dictionary"):
        aggregate_evaluation_results("not_a_dict")  # type: ignore[arg-type]


def test_force_closure_additional_rank_and_friction() -> None:
    """Verify force closure evaluations for negative friction coefficients and zero-normal vectors."""
    with pytest.raises(ValueError, match="friction_coefficient must be non-negative"):
        compute_grasp_wrench_matrix([], -0.5)

    judge = build_force_closure_judge(0.5, wrench_regularization=0.0)
    single_contact = [{"position": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0])}]
    if judge(single_contact):
        raise AssertionError

    zero_normal_contact = [{"position": np.zeros(3), "normal": np.zeros(3)}]
    w_zero = compute_grasp_wrench_matrix(zero_normal_contact, 0.5)
    if not (w_zero.shape == (6, 0)):
        raise AssertionError


def test_force_closure_degenerate_contacts_hull_fallback() -> None:
    """Verify that degenerate contact points spanning a subspace yield zero grasp quality values."""
    degenerate_contacts = [
        {
            "position": np.array([i * 0.01, 0.0, 0.0]),
            "normal": np.array([0.0, 0.0, 1.0]),
        }
        for i in range(7)
    ]
    q_deg = compute_grasp_quality(degenerate_contacts, friction_coefficient=0.5)
    if not (q_deg == 0.0):
        raise AssertionError


def test_aggregate_evaluation_results_partial_records() -> None:
    """Verify aggregation results on partial evaluation records and missing keys."""
    per_obj = {
        "empty_obj": [],
        "partial_obj": [
            {"collision_free": True},
            {"force_closure": False, "grasp_quality": 0.0},
        ],
    }
    agg = aggregate_evaluation_results(per_obj)
    if "success_rate" not in agg:
        raise AssertionError
    if not (agg["success_rate"] == 0.0):
        raise AssertionError


def test_force_closure_enclosing_origin_6d_hull() -> None:
    """Verify that six opposing contact points enclosing the coordinate origin yield positive grasp quality."""
    contacts = [
        {"position": np.array([0.05, 0.0, 0.0]), "normal": np.array([-1.0, 0.0, 0.0])},
        {"position": np.array([-0.05, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        {"position": np.array([0.0, 0.05, 0.0]), "normal": np.array([0.0, -1.0, 0.0])},
        {"position": np.array([0.0, -0.05, 0.0]), "normal": np.array([0.0, 1.0, 0.0])},
        {"position": np.array([0.0, 0.0, 0.05]), "normal": np.array([0.0, 0.0, -1.0])},
        {"position": np.array([0.0, 0.0, -0.05]), "normal": np.array([0.0, 0.0, 1.0])},
    ]
    judge = build_force_closure_judge(0.5, wrench_regularization=0.0)
    if judge(contacts) is not True:
        raise AssertionError
    quality = compute_grasp_quality(contacts, 0.5)
    if not (quality > 0.0):
        raise AssertionError


def test_load_contact_set_1d_array(tmp_path: Path) -> None:
    """Verify that load_contact_set properly unpacks 1D arrays or scalar object payloads from disk."""
    payload = np.empty((), dtype=object)
    payload[()] = {"position": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0])}
    path = tmp_path / "1d_contacts.npy"
    np.save(path, payload, allow_pickle=True)
    loaded = load_contact_set(path)
    if not (len(loaded) == 1):
        raise AssertionError
    if "position" not in loaded[0]:
        raise AssertionError


def test_evaluate_generated_grasps_validations_and_contact_path(tmp_path: Path) -> None:
    """Verify validation checks and file loading in the main generate_grasps evaluation pipeline."""
    obj_pc = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    grip_pc = np.array([[0.0, 0.0, 0.002]], dtype=np.float32)

    with pytest.raises(ValueError, match="grasp_poses must have shape"):
        evaluate_generated_grasps(np.zeros((2, 3)), obj_pc, grip_pc)

    with pytest.raises(ValueError, match="friction_coefficient must be non-negative"):
        evaluate_generated_grasps(np.eye(4), obj_pc, grip_pc, friction_coefficient=-0.1)

    with pytest.raises(ValueError, match="lift_height_threshold must be non-negative"):
        evaluate_generated_grasps(np.eye(4), obj_pc, grip_pc, lift_height_threshold=-0.1)

    with pytest.raises(ValueError, match="clearance must be non-negative"):
        evaluate_generated_grasps(np.eye(4), obj_pc, grip_pc, clearance=-0.1)

    with pytest.raises(ValueError, match="wrench_regularization must be non-negative"):
        evaluate_generated_grasps(np.eye(4), obj_pc, grip_pc, wrench_regularization=-0.1)

    contact_path = tmp_path / "contacts.npy"
    contacts_list = [{"position": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0])}]
    payload = np.empty((), dtype=object)
    payload[()] = contacts_list
    np.save(contact_path, payload, allow_pickle=True)

    results = evaluate_generated_grasps(np.eye(4), obj_pc, grip_pc, contact_path=contact_path)
    if not (len(results) == 1):
        raise AssertionError


def test_evaluate_generated_grasps_all_colliding_filtered() -> None:
    """Verify that all colliding grasps are filtered out when collision checking is enabled."""
    obj_pc = np.zeros((10, 3), dtype=np.float32)
    grip_pc = np.zeros((10, 3), dtype=np.float32)
    grasps = np.eye(4)[None]

    results = evaluate_generated_grasps(grasps, obj_pc, grip_pc, clearance=0.05, filter_collisions=True)
    if not (results == []):
        raise AssertionError


def test_write_evaluation_report_tb_and_exceptions(tmp_path: Path) -> None:
    """Verify that write_evaluation_report writes JSONL outputs, logs to TensorBoard, and validates paths."""
    with pytest.raises(TypeError, match=r"report_path must be a pathlib\.Path"):
        write_evaluation_report("not_a_path", {})  # type: ignore[arg-type]

    report_file = tmp_path / "report.jsonl"
    tb_dir = tmp_path / "tb_events"
    write_evaluation_report(report_file, {"metric": 0.95}, experiment_log_dir=tb_dir)
    if not (report_file.is_file()):
        raise AssertionError
    if not (tb_dir.exists()):
        raise AssertionError
    records = read_jsonl_records(report_file)
    if not (records[0]["record_type"] == "summary"):
        raise AssertionError
    if not (records[0]["metric"] == EXPECTED_HIGH_METRIC):
        raise AssertionError

    dir_path = tmp_path / "is_a_dir"
    dir_path.mkdir()
    with pytest.raises(ValueError, match="Failed to write JSONL records"):
        write_evaluation_report(dir_path, {})


def test_jsonl_io_validation_and_error_paths(tmp_path: Path) -> None:
    """Validate JSONL read/write helpers and their error handling.

    Args:
        tmp_path: Temporary directory for JSONL fixture files.

    Returns:
        None. Asserts type, parse, and mapping validation failures are raised.
    """
    with pytest.raises(TypeError, match=r"output_path must be a pathlib\.Path"):
        write_jsonl_records("bad", [])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"input_path must be a pathlib\.Path"):
        read_jsonl_records("bad")  # type: ignore[arg-type]

    records_path = tmp_path / "records.jsonl"
    write_jsonl_records(records_path, [{"record_type": "summary", "ok": True}])
    if read_jsonl_records(records_path)[0]["ok"] is not True:
        raise AssertionError

    with_blank = tmp_path / "blank.jsonl"
    with_blank.write_text('\n\n{"record_type": "summary"}\n\n', encoding="utf-8")
    if not (read_jsonl_records(with_blank)[0]["record_type"] == "summary"):
        raise AssertionError

    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to read JSONL records"):
        read_jsonl_records(bad_json)

    non_mapping = tmp_path / "non_mapping.jsonl"
    non_mapping.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(TypeError, match="must be a mapping"):
        read_jsonl_records(non_mapping)

    missing = tmp_path / "missing.jsonl"
    with pytest.raises(ValueError, match="Failed to read JSONL records"):
        read_jsonl_records(missing)

    write_evaluation_report(
        tmp_path / "multi.jsonl",
        {"success_rate": 0.5},
        per_object_results={"obj_a": {"success_rate": 0.0}},
    )
    multi_records = read_jsonl_records(tmp_path / "multi.jsonl")
    if not (multi_records[0]["record_type"] == "object"):
        raise AssertionError
    if not (multi_records[1]["record_type"] == "summary"):
        raise AssertionError


def _check_rank_deficient_quality() -> None:
    c1 = [{"position": np.array([0.01, 0.0, 0.0]), "normal": np.array([-1.0, 0.0, 0.0])}]
    q1 = compute_grasp_quality(c1, friction_coefficient=0.5)
    if not (q1 == 0.0):
        raise AssertionError

    c_zero_rank = [
        {"position": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        {"position": np.array([0.0, 0.0, 0.0]), "normal": np.array([-1.0, 0.0, 0.0])},
    ]
    judge = build_force_closure_judge(friction_coefficient=0.5, wrench_regularization=0.0)
    is_fc = judge(c_zero_rank)
    if is_fc:
        raise AssertionError


def _check_multi_item_contact_files(tmp_path: Path) -> None:
    c_list = [
        {"position": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0])},
        {"position": np.ones(3), "normal": np.array([0.0, 0.0, -1.0])},
    ]
    payload = np.empty((), dtype=object)
    payload[()] = c_list
    path = tmp_path / "multi_item.npy"
    np.save(path, payload, allow_pickle=True)
    loaded = load_contact_set(path)
    if not (len(loaded) == EXPECTED_PAIR_COUNT):
        raise AssertionError

    # 1. Test line 32: load list directly (not data.item)
    payload_list = np.array(c_list, dtype=object)
    path_list = tmp_path / "list_contact.npy"
    np.save(path_list, payload_list, allow_pickle=True)
    loaded_list = load_contact_set(path_list)
    if not (len(loaded_list) == EXPECTED_PAIR_COUNT):
        raise AssertionError


def _check_hull_and_lp_failure_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    # 2. Test line 225: max_norm <= 1e-8
    orig_max = np.max

    def fake_max(
        a: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        if isinstance(a, np.ndarray) and a.ndim == 1 and len(a) >= FAKE_MAX_MIN_LEN:
            return 0.0
        return orig_max(a, *args, **kwargs)

    monkeypatch.setattr(np, "max", fake_max)

    class ConvexHullError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("Convex hull failed")

    def raise_hull(*_args: object, **_kwargs: object) -> None:
        raise ConvexHullError

    monkeypatch.setattr(scipy.spatial, "ConvexHull", raise_hull)

    c_valid = [
        {"position": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        {"position": np.array([0.05, 0.0, 0.0]), "normal": np.array([-1.0, 0.0, 0.0])},
    ]
    q_lp = compute_grasp_quality(c_valid, friction_coefficient=0.5)
    if not (q_lp >= 0.0):
        raise AssertionError

    # 4. Test LP exception (lines 269-270)
    class LinprogError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("linprog failed")

    def raise_linprog(*_args: object, **_kwargs: object) -> None:
        raise LinprogError

    monkeypatch.setattr(scipy.optimize, "linprog", raise_linprog)
    q_err = compute_grasp_quality(c_valid, friction_coefficient=0.5)
    if not (q_err == 0.0):
        raise AssertionError


def test_force_closure_additional_coverage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify force closure judge evaluations for rank-deficient systems and loaded multi-item contact files."""
    _check_rank_deficient_quality()
    _check_multi_item_contact_files(tmp_path)
    _check_hull_and_lp_failure_branches(monkeypatch)


def test_force_closure_full_branch_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify optimizer branch failures and exceptions inside the force closure linear programming solver."""
    c_3pts = [
        {"position": np.array([1.0, 0.0, 0.0]), "normal": np.array([0.0, 1.0, 0.0])},
        {"position": np.array([0.0, 1.0, 0.0]), "normal": np.array([0.0, 0.0, 1.0])},
        {"position": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
    ]
    q = compute_grasp_quality(c_3pts, friction_coefficient=0.5)
    if not (q == 0.0):
        raise AssertionError

    judge = build_force_closure_judge(0.5, wrench_regularization=0.0)

    class FailedRes:
        success = False

    monkeypatch.setattr(scipy.optimize, "linprog", lambda *_args, **_kwargs: FailedRes())
    if judge(c_3pts):
        raise AssertionError

    def raise_err(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError

    monkeypatch.setattr(scipy.optimize, "linprog", raise_err)
    if judge(c_3pts):
        raise AssertionError


def test_evaluate_additional_branch_coverage(tmp_path: Path) -> None:
    """Verify evaluation and reporting branches when all configurations trigger collisions."""
    eval_mod = sys.modules["grasping_ai.pipelines.evaluate"]
    evaluate_generated_grasps = eval_mod.evaluate_generated_grasps
    aggregate_evaluation_results = eval_mod.aggregate_evaluation_results
    write_evaluation_report = eval_mod.write_evaluation_report

    obj_pc = np.zeros((10, 3))
    grip_pc = np.zeros((10, 3))
    poses = np.eye(4)[None, ...]
    res = evaluate_generated_grasps(poses, obj_pc, grip_pc, filter_collisions=True, clearance=100.0)
    if not (res == []):
        raise AssertionError

    outcomes = {"obj1": [{"grasp_success": False, "collision_free": False, "force_closure": False}]}
    summary = aggregate_evaluation_results(outcomes)
    if not (summary["mean_grasp_quality"] == 0.0):
        raise AssertionError

    rep = tmp_path / "report_str.json"
    tb = tmp_path / "tb"
    write_evaluation_report(rep, {"str_key": "val", "num_key": 1.0}, experiment_log_dir=tb)
    if not (rep.is_file()):
        raise AssertionError
