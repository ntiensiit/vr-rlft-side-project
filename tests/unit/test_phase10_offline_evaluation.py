import numpy as np

from grasping_ai.pipelines.evaluate import (
    aggregate_evaluation_results,
    evaluate_generated_grasps,
)


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
