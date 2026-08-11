import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scripts.evaluate import evaluate_main


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
