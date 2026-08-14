import tempfile
from pathlib import Path

import numpy as np
import pytest

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
)
from grasping_ai.pipelines.generate_grasps import load_generated_grasps


@pytest.fixture
def temp_files():
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


def test_analytical_contacts_valid():
    # Setup simple geometry
    # Object is a single point at origin
    object_pc = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    # Gripper has two points
    gripper_pc = np.array([
        [0.0, 0.0, 0.002],
        [0.0, 0.0, 0.010],
    ], dtype=np.float32)

    # Identity pose
    grasp_pose = np.eye(4, dtype=np.float32)

    # Clearance is 0.005. Only the first gripper point is within clearance (dist=0.002)
    contacts = generate_analytical_contacts(
        object_pc, gripper_pc, grasp_pose, contact_clearance=0.005
    )

    assert len(contacts) == 1
    assert np.allclose(contacts[0]["position"], np.array([0.0, 0.0, 0.0]))
    # Normal points from gripper point (0,0,0.002) to object point (0,0,0)
    # i.e. vector is (0,0,-0.002), normalized is (0,0,-1.0)
    assert np.allclose(contacts[0]["normal"], np.array([0.0, 0.0, -1.0]))


def test_analytical_contacts_rejection():
    object_pc = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    gripper_pc = np.array([[0.0, 0.0, 0.1]], dtype=np.float32)
    grasp_pose = np.eye(4, dtype=np.float32)

    # Too far, should yield 0 contacts
    contacts = generate_analytical_contacts(
        object_pc, gripper_pc, grasp_pose, contact_clearance=0.005
    )
    assert len(contacts) == 0


def test_analytical_contacts_invalid_shapes():
    object_pc = np.array([[0.0, 0.0]], dtype=np.float32)
    gripper_pc = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    grasp_pose = np.eye(4, dtype=np.float32)

    with pytest.raises(ValueError, match="object_point_cloud"):
        generate_analytical_contacts(object_pc, gripper_pc, grasp_pose, 0.005)

    object_pc = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    grasp_pose = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match="grasp_pose"):
        generate_analytical_contacts(object_pc, gripper_pc, grasp_pose, 0.005)


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


def test_evaluate_generated_grasps_analytical():
    # Setup minimal point clouds
    object_pc = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    gripper_pc = np.array([[0.0, 0.0, 0.002]], dtype=np.float32)

    # 2 identical grasps
    grasps = np.stack([np.eye(4), np.eye(4)], axis=0)

    evals = evaluate_generated_grasps(
        grasp_poses=grasps,
        object_point_cloud=object_pc,
        gripper_point_cloud=gripper_pc,
        contact_set_provider=None, # triggers analytical contact generation
        friction_coefficient=0.5,
        lift_height_threshold=0.05,
        wrench_regularization=0.0,
    )

    assert len(evals) == 2
    for res in evals:
        assert "collision_free" in res
        assert "force_closure" in res
        assert "grasp_success" in res
        assert "grasp_quality" in res
        # Wait, since the rank will be < 6 (only 1 contact point), force_closure should be False
        assert not res["force_closure"]
        assert res["grasp_quality"] == 0.0


def test_aggregate_evaluation_results():
    per_obj = {
        "bottle": [
            {"collision_free": True, "force_closure": True, "grasp_success": True, "grasp_quality": 0.5},
            {"collision_free": True, "force_closure": False, "grasp_success": False, "grasp_quality": 0.0},
        ]
    }

    aggregated = aggregate_evaluation_results(per_obj)
    assert aggregated["success_rate"] == 0.5
    assert aggregated["collision_free_rate"] == 1.0
    assert aggregated["force_closure_rate"] == 0.5
    assert aggregated["mean_grasp_quality"] == 0.25
    assert aggregated["min_grasp_quality"] == 0.0
    assert aggregated["max_grasp_quality"] == 0.5


def test_evaluate_script_plain_array(temp_files):
    grasps = load_generated_grasps(temp_files["grasps_arr_path"], object_key="default")
    object_point_cloud = np.load(temp_files["obj_path"])
    gripper_point_cloud = np.load(temp_files["grip_path"])
    per_grasp = evaluate_generated_grasps(
        grasp_poses=grasps,
        object_point_cloud=object_point_cloud,
        gripper_point_cloud=gripper_point_cloud,
        friction_coefficient=0.5,
        lift_height_threshold=0.05,
    )
    write_evaluation_report(
        temp_files["report_path"],
        aggregate_evaluation_results({"default": per_grasp}),
    )

    assert temp_files["report_path"].is_file()
    report = next(
        record
        for record in read_jsonl_records(temp_files["report_path"])
        if record.get("record_type") == "summary"
    )

    assert "success_rate" in report
    assert "mean_grasp_quality" in report


def test_evaluate_script_dictionary(temp_files):
    grasps = load_generated_grasps(
        temp_files["grasps_dict_path"], object_key="mustard_bottle"
    )
    object_point_cloud = np.load(temp_files["obj_path"])
    gripper_point_cloud = np.load(temp_files["grip_path"])
    per_grasp = evaluate_generated_grasps(
        grasp_poses=grasps,
        object_point_cloud=object_point_cloud,
        gripper_point_cloud=gripper_point_cloud,
        friction_coefficient=0.5,
        lift_height_threshold=0.05,
    )
    write_evaluation_report(
        temp_files["report_path"],
        aggregate_evaluation_results({"mustard_bottle": per_grasp}),
    )

    assert temp_files["report_path"].is_file()
    report = next(
        record
        for record in read_jsonl_records(temp_files["report_path"])
        if record.get("record_type") == "summary"
    )

    assert "success_rate" in report
    assert "mean_grasp_quality" in report


def test_force_closure_load_contact_set_validations(tmp_path: Path) -> None:
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
    assert len(loaded) == 1

    corrupted_file = tmp_path / "corrupted.npy"
    corrupted_file.write_bytes(b"invalid data")
    with pytest.raises(ValueError, match="Failed to load contact set"):
        load_contact_set(corrupted_file)


def test_force_closure_judge_and_quality_validations() -> None:
    with pytest.raises(ValueError, match="friction_coefficient must be non-negative"):
        build_force_closure_judge(-0.1, 0.01)

    with pytest.raises(ValueError, match="wrench_regularization must be non-negative"):
        build_force_closure_judge(0.5, -0.01)

    judge = build_force_closure_judge(0.5, 0.01)
    assert not judge([])
    assert not evaluate_force_closure(judge, [])

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
    assert q > 0.0


def test_metrics_validations() -> None:
    with pytest.raises(ValueError, match="max_linear_velocity must be non-negative"):
        build_stability_judge(-1.0, 1.0)
    with pytest.raises(ValueError, match="max_angular_velocity must be non-negative"):
        build_stability_judge(1.0, -1.0)

    stab_judge = build_stability_judge(1.0, 1.0)
    with pytest.raises(TypeError, match="object_velocity must be a numpy array"):
        evaluate_stability(stab_judge, [0.0] * 6)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="6D velocity"):
        evaluate_stability(stab_judge, np.zeros(3))

    assert evaluate_stability(stab_judge, np.zeros(6))
    assert not evaluate_stability(stab_judge, np.ones(6) * 10.0)

    with pytest.raises(ValueError, match="lift_height_threshold must be non-negative"):
        build_lift_outcome_judge(-0.1)

    lift_judge = build_lift_outcome_judge(0.05)
    with pytest.raises(TypeError, match="initial_height must be a number"):
        evaluate_lift_success(lift_judge, "invalid", 0.1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="final_height must be a number"):
        evaluate_lift_success(lift_judge, 0.0, "invalid")  # type: ignore[arg-type]

    assert evaluate_lift_success(lift_judge, 0.0, 0.06)
    assert not evaluate_lift_success(lift_judge, 0.0, 0.02)

    with pytest.raises(TypeError, match="per_object_success must be a dictionary"):
        aggregate_grasp_success_rate("not_a_dict")  # type: ignore[arg-type]
    assert aggregate_grasp_success_rate({}) == 0.0
    assert aggregate_grasp_success_rate({"obj1": True, "obj2": False}) == 0.5


def test_collision_and_evaluate_pipeline_validations() -> None:
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

    empty_filtered = filter_collision_free_grasps(checker, np.zeros((0, 4, 4)))
    assert empty_filtered.shape == (0, 4, 4)

    res_2d = filter_collision_free_grasps(checker, np.eye(4))
    assert res_2d.shape == (1, 4, 4)

    empty_evals = evaluate_generated_grasps(
        grasp_poses=np.zeros((0, 4, 4)),
        object_point_cloud=np.zeros((10, 3)),
        gripper_point_cloud=np.zeros((5, 3)),
    )
    assert len(empty_evals) == 0

    assert aggregate_evaluation_results({})["success_rate"] == 0.0


def test_force_closure_additional_branches() -> None:
    incomplete_contacts = [{"position": np.zeros(3)}, {"normal": np.zeros(3)}]
    w_matrix = compute_grasp_wrench_matrix(incomplete_contacts, 0.5)
    assert w_matrix.shape == (6, 0)

    zero_contacts = [{"position": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0])} for _ in range(7)]
    q_zero = compute_grasp_quality(zero_contacts, 0.5)
    assert q_zero == 0.0


def test_collision_additional_branches() -> None:
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

    assert generate_analytical_contacts(np.zeros((0, 3)), np.zeros((5, 3)), np.eye(4), 0.05) == []

    contacts_overlap = generate_analytical_contacts(np.zeros((1, 3)), np.zeros((1, 3)), np.eye(4), 0.05)
    assert len(contacts_overlap) == 1
    assert np.allclose(contacts_overlap[0]["normal"], [0.0, 0.0, 1.0])

    colliding_checker = build_collision_checker(np.zeros((10, 3)), np.zeros((5, 3)), 0.05)
    assert filter_collision_free_grasps(colliding_checker, np.eye(4)).shape == (0, 4, 4)
    assert filter_collision_free_grasps(colliding_checker, np.stack([np.eye(4), np.eye(4)])).shape == (0, 4, 4)


def test_evaluate_pipeline_additional_branches(tmp_path: Path) -> None:
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
    assert len(evals_from_path) == 0

    custom_evals = evaluate_generated_grasps(
        grasp_poses=np.eye(4),
        object_point_cloud=np.full((10, 3), 10.0),
        gripper_point_cloud=np.zeros((5, 3)),
        contact_set_provider=lambda pose: [],
    )
    assert len(custom_evals) == 1
    assert not custom_evals[0]["force_closure"]

    with pytest.raises(TypeError, match="per_object_results must be a dictionary"):
        aggregate_evaluation_results("not_a_dict")  # type: ignore[arg-type]


def test_force_closure_additional_rank_and_friction() -> None:
    with pytest.raises(ValueError, match="friction_coefficient must be non-negative"):
        compute_grasp_wrench_matrix([], -0.5)

    judge = build_force_closure_judge(0.5, wrench_regularization=0.0)
    single_contact = [{"position": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0])}]
    assert not judge(single_contact)

    zero_normal_contact = [{"position": np.zeros(3), "normal": np.zeros(3)}]
    w_zero = compute_grasp_wrench_matrix(zero_normal_contact, 0.5)
    assert w_zero.shape == (6, 0)


def test_force_closure_degenerate_contacts_hull_fallback() -> None:
    degenerate_contacts = [
        {"position": np.array([i * 0.01, 0.0, 0.0]), "normal": np.array([0.0, 0.0, 1.0])}
        for i in range(7)
    ]
    q_deg = compute_grasp_quality(degenerate_contacts, friction_coefficient=0.5)
    assert q_deg == 0.0


def test_aggregate_evaluation_results_partial_records() -> None:
    per_obj = {
        "empty_obj": [],
        "partial_obj": [
            {"collision_free": True},
            {"force_closure": False, "grasp_quality": 0.0},
        ],
    }
    agg = aggregate_evaluation_results(per_obj)
    assert "success_rate" in agg
    assert agg["success_rate"] == 0.0


def test_force_closure_enclosing_origin_6d_hull() -> None:
    contacts = [
        {"position": np.array([0.05, 0.0, 0.0]), "normal": np.array([-1.0, 0.0, 0.0])},
        {"position": np.array([-0.05, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        {"position": np.array([0.0, 0.05, 0.0]), "normal": np.array([0.0, -1.0, 0.0])},
        {"position": np.array([0.0, -0.05, 0.0]), "normal": np.array([0.0, 1.0, 0.0])},
        {"position": np.array([0.0, 0.0, 0.05]), "normal": np.array([0.0, 0.0, -1.0])},
        {"position": np.array([0.0, 0.0, -0.05]), "normal": np.array([0.0, 0.0, 1.0])},
    ]
    judge = build_force_closure_judge(0.5, wrench_regularization=0.0)
    assert judge(contacts) is True
    quality = compute_grasp_quality(contacts, 0.5)
    assert quality > 0.0


def test_load_contact_set_1d_array(tmp_path: Path) -> None:
    payload = np.empty((), dtype=object)
    payload[()] = {"position": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0])}
    path = tmp_path / "1d_contacts.npy"
    np.save(path, payload, allow_pickle=True)
    loaded = load_contact_set(path)
    assert isinstance(loaded, dict)
    assert "position" in loaded


def test_evaluate_generated_grasps_validations_and_contact_path(tmp_path: Path) -> None:
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

    results = evaluate_generated_grasps(
        np.eye(4), obj_pc, grip_pc, contact_path=contact_path
    )
    assert len(results) == 1


def test_evaluate_generated_grasps_all_colliding_filtered() -> None:
    obj_pc = np.zeros((10, 3), dtype=np.float32)
    grip_pc = np.zeros((10, 3), dtype=np.float32)
    grasps = np.eye(4)[None]

    results = evaluate_generated_grasps(
        grasps, obj_pc, grip_pc, clearance=0.05, filter_collisions=True
    )
    assert results == []


def test_write_evaluation_report_tb_and_exceptions(tmp_path: Path) -> None:
    from grasping_ai.pipelines.evaluate import read_jsonl_records, write_evaluation_report

    with pytest.raises(TypeError, match=r"report_path must be a pathlib\.Path"):
        write_evaluation_report("not_a_path", {})  # type: ignore[arg-type]

    report_file = tmp_path / "report.jsonl"
    tb_dir = tmp_path / "tb_events"
    write_evaluation_report(report_file, {"metric": 0.95}, experiment_log_dir=tb_dir)
    assert report_file.is_file()
    assert tb_dir.exists()
    records = read_jsonl_records(report_file)
    assert records[0]["record_type"] == "summary"
    assert records[0]["metric"] == 0.95

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
    from grasping_ai.pipelines.evaluate import read_jsonl_records, write_jsonl_records

    with pytest.raises(TypeError, match=r"output_path must be a pathlib\.Path"):
        write_jsonl_records("bad", [])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"input_path must be a pathlib\.Path"):
        read_jsonl_records("bad")  # type: ignore[arg-type]

    records_path = tmp_path / "records.jsonl"
    write_jsonl_records(records_path, [{"record_type": "summary", "ok": True}])
    assert read_jsonl_records(records_path)[0]["ok"] is True

    with_blank = tmp_path / "blank.jsonl"
    with_blank.write_text("\n\n{\"record_type\": \"summary\"}\n\n", encoding="utf-8")
    assert read_jsonl_records(with_blank)[0]["record_type"] == "summary"

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
    assert multi_records[0]["record_type"] == "object"
    assert multi_records[1]["record_type"] == "summary"


def test_force_closure_additional_coverage(monkeypatch, tmp_path: Path) -> None:
    from grasping_ai.evaluation.force_closure import (
        build_force_closure_judge,
        compute_grasp_quality,
        load_contact_set,
    )

    c1 = [{"position": np.array([0.01, 0.0, 0.0]), "normal": np.array([-1.0, 0.0, 0.0])}]
    q1 = compute_grasp_quality(c1, friction_coefficient=0.5)
    assert q1 == 0.0

    c_zero_rank = [
        {"position": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        {"position": np.array([0.0, 0.0, 0.0]), "normal": np.array([-1.0, 0.0, 0.0])},
    ]
    judge = build_force_closure_judge(friction_coefficient=0.5, wrench_regularization=0.0)
    is_fc = judge(c_zero_rank)
    assert not is_fc

    c_list = [
        {"position": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0])},
        {"position": np.ones(3), "normal": np.array([0.0, 0.0, -1.0])},
    ]
    payload = np.empty((), dtype=object)
    payload[()] = c_list
    path = tmp_path / "multi_item.npy"
    np.save(path, payload, allow_pickle=True)
    loaded = load_contact_set(path)
    assert len(loaded) == 2


def test_force_closure_full_branch_coverage(monkeypatch) -> None:
    import scipy.optimize

    from grasping_ai.evaluation.force_closure import (
        build_force_closure_judge,
        compute_grasp_quality,
    )

    c_3pts = [
        {"position": np.array([1.0, 0.0, 0.0]), "normal": np.array([0.0, 1.0, 0.0])},
        {"position": np.array([0.0, 1.0, 0.0]), "normal": np.array([0.0, 0.0, 1.0])},
        {"position": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
    ]
    q = compute_grasp_quality(c_3pts, friction_coefficient=0.5)
    assert q == 0.0

    judge = build_force_closure_judge(0.5, wrench_regularization=0.0)

    class FailedRes:
        success = False

    monkeypatch.setattr(scipy.optimize, "linprog", lambda *args, **kwargs: FailedRes())
    assert not judge(c_3pts)

    def raise_err(*args, **kwargs):
        raise RuntimeError

    monkeypatch.setattr(scipy.optimize, "linprog", raise_err)
    assert not judge(c_3pts)


def test_evaluate_additional_branch_coverage(tmp_path: Path) -> None:
    import sys

    eval_mod = sys.modules["grasping_ai.pipelines.evaluate"]
    evaluate_generated_grasps = eval_mod.evaluate_generated_grasps
    aggregate_evaluation_results = eval_mod.aggregate_evaluation_results
    write_evaluation_report = eval_mod.write_evaluation_report

    obj_pc = np.zeros((10, 3))
    grip_pc = np.zeros((10, 3))
    poses = np.eye(4)[None, ...]
    res = evaluate_generated_grasps(
        poses, obj_pc, grip_pc, filter_collisions=True, clearance=100.0
    )
    assert res == []

    outcomes = {
        "obj1": [{"grasp_success": False, "collision_free": False, "force_closure": False}]
    }
    summary = aggregate_evaluation_results(outcomes)
    assert summary["mean_grasp_quality"] == 0.0

    rep = tmp_path / "report_str.json"
    tb = tmp_path / "tb"
    write_evaluation_report(rep, {"str_key": "val", "num_key": 1.0}, experiment_log_dir=tb)
    assert rep.is_file()
