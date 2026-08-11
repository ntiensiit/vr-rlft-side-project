import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scripts.evaluate import evaluate_main

from grasping_ai.evaluation.collision import generate_analytical_contacts
from grasping_ai.evaluation.force_closure import compute_grasp_quality
from grasping_ai.pipelines.evaluate import (
    aggregate_evaluation_results,
    evaluate_generated_grasps,
)


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

        report_path = tmp_path / "report.json"

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
        assert "lift_success" in res
        assert "grasp_quality" in res
        # Wait, since the rank will be < 6 (only 1 contact point), force_closure should be False
        assert not res["force_closure"]
        assert res["grasp_quality"] == 0.0


def test_aggregate_evaluation_results():
    per_obj = {
        "bottle": [
            {"collision_free": True, "force_closure": True, "lift_success": True, "grasp_quality": 0.5},
            {"collision_free": True, "force_closure": False, "lift_success": False, "grasp_quality": 0.0},
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
    evaluate_main(
        grasps_path=temp_files["grasps_arr_path"],
        object_id="default",
        object_point_cloud_path=temp_files["obj_path"],
        gripper_point_cloud_path=temp_files["grip_path"],
        report_path=temp_files["report_path"],
        friction_coefficient=0.5,
        lift_height_threshold=0.05,
    )

    assert temp_files["report_path"].is_file()
    with open(temp_files["report_path"]) as f:
        report = json.load(f)

    assert "success_rate" in report
    assert "mean_grasp_quality" in report


def test_evaluate_script_dictionary(temp_files):
    evaluate_main(
        grasps_path=temp_files["grasps_dict_path"],
        object_id="mustard_bottle",
        object_point_cloud_path=temp_files["obj_path"],
        gripper_point_cloud_path=temp_files["grip_path"],
        report_path=temp_files["report_path"],
        friction_coefficient=0.5,
        lift_height_threshold=0.05,
    )

    assert temp_files["report_path"].is_file()
    with open(temp_files["report_path"]) as f:
        report = json.load(f)

    assert "success_rate" in report
    assert "mean_grasp_quality" in report
