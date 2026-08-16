"""Phase 6 orchestration pipeline tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import grasping_ai
from grasping_ai.evaluation.collision import (
    build_collision_checker,
    check_collision,
    filter_collision_free_grasps,
)
from grasping_ai.evaluation.force_closure import (
    build_force_closure_judge,
    compute_grasp_wrench_matrix,
    evaluate_force_closure,
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
from grasping_ai.pipelines.simulate_grasp import simulate_grasp
from grasping_ai.robotics.gripper import panda_hand_to_contact_transform
from grasping_ai.robotics.kinematics import (
    build_forward_kinematics,
    load_robot_model,
)
from grasping_ai.robotics.transforms import transform_grasp_pose

WRENCH_MATRIX_ROWS = 6
WRENCH_MATRIX_COLS = 8
EXPECTED_GRASP_COUNT = 2


@pytest.fixture
def minimal_ycb_root(tmp_path: Path) -> Path:
    """Fixture providing a raw YCB root directory with a minimal mustard bottle MuJoCo XML."""
    obj_dir = tmp_path / "ycb" / "006_mustard_bottle"
    obj_dir.mkdir(parents=True)
    xml_content = """
    <mujoco model="mustard_bottle">
        <worldbody>
            <body name="mustard_bottle" pos="0 0 0">
                <geom name="mustard_bottle_geom" type="box" size="0.05 0.05 0.05"/>
            </body>
        </worldbody>
    </mujoco>
    """
    (obj_dir / "mustard_bottle.xml").write_text(xml_content, encoding="utf-8")
    return tmp_path / "ycb"


@pytest.fixture
def minimal_ycb_root_differing_body_name(tmp_path: Path) -> Path:
    """Fixture providing a YCB root where the object body name differs from the object folder/file name."""
    obj_dir = tmp_path / "ycb" / "006_mustard_bottle"
    obj_dir.mkdir(parents=True)
    xml_content = """
    <mujoco model="mustard_bottle">
        <worldbody>
            <body name="original_body_name" pos="0 0 0.5">
                <geom name="original_body_geom" type="box" size="0.05 0.05 0.05"/>
            </body>
        </worldbody>
    </mujoco>
    """
    (obj_dir / "mustard_bottle.xml").write_text(xml_content, encoding="utf-8")
    return tmp_path / "ycb"


@pytest.fixture
def minimal_ycb_root_at_height(tmp_path: Path) -> Path:
    """Fixture providing a YCB root where the object is initialized at a specific height offset."""
    obj_dir = tmp_path / "ycb" / "006_mustard_bottle"
    obj_dir.mkdir(parents=True)
    xml_content = """
    <mujoco model="mustard_bottle">
        <worldbody>
            <body name="mustard_bottle" pos="0 0 0.5">
                <geom name="mustard_bottle_geom" type="box" size="0.05 0.05 0.05"/>
            </body>
        </worldbody>
    </mujoco>
    """
    (obj_dir / "mustard_bottle.xml").write_text(xml_content, encoding="utf-8")
    return tmp_path / "ycb"


@pytest.fixture
def minimal_ycb_root_freejoint(tmp_path: Path) -> Path:
    """Fixture providing a YCB root where the object body contains a freejoint for dynamic simulation."""
    obj_dir = tmp_path / "ycb" / "006_mustard_bottle"
    obj_dir.mkdir(parents=True)
    xml_content = """
    <mujoco model="mustard_bottle">
        <worldbody>
            <body name="mustard_bottle" pos="0 0 0.05">
                <freejoint/>
                <geom name="mustard_bottle_geom" type="box" size="0.05 0.05 0.05"/>
            </body>
        </worldbody>
    </mujoco>
    """
    (obj_dir / "mustard_bottle.xml").write_text(xml_content, encoding="utf-8")
    return tmp_path / "ycb"


def test_phase1_package_import_remains_stable() -> None:
    """Verify that grasping_ai is importable."""
    if not (grasping_ai.__name__ == "grasping_ai"):
        raise AssertionError


def test_evaluation_config_files_exist() -> None:
    """Verify evaluation config default alias, common base, and method variants exist."""
    if not (Path("configs") / "evaluation" / "default.yaml").is_file():
        raise AssertionError
    if not (Path("configs") / "evaluation" / "diffusion.yaml").is_file():
        raise AssertionError
    if not (Path("configs") / "evaluation" / "flow.yaml").is_file():
        raise AssertionError
    if not (Path("configs") / "evaluation" / "rl.yaml").is_file():
        raise AssertionError


def test_collision_checker_shape_checks() -> None:
    """Verify collision checker validates inputs."""
    rng = np.random.default_rng()
    obj_pc = rng.standard_normal((10, 3))
    grip_pc = rng.standard_normal((5, 3))

    with pytest.raises(ValueError, match="object_point_cloud"):
        build_collision_checker(rng.standard_normal((10, 2)), grip_pc, 0.01)

    with pytest.raises(ValueError, match="gripper_point_cloud"):
        build_collision_checker(obj_pc, rng.standard_normal((5, 4)), 0.01)

    with pytest.raises(ValueError, match="clearance"):
        build_collision_checker(obj_pc, grip_pc, -0.01)


def test_collision_metric_returns_valid_output() -> None:
    """Verify check_collision and filter_collision_free_grasps work."""
    obj_pc = np.array([[0.0, 0.0, 0.0]])
    grip_pc = np.array([[0.0, 0.0, 0.0]])

    checker = build_collision_checker(obj_pc, grip_pc, clearance=0.1)

    # Pose at (1, 1, 1) -> distance is sqrt(3) > 0.1 -> collision free
    pose_free = np.eye(4)
    pose_free[:3, 3] = [1.0, 1.0, 1.0]
    if check_collision(checker, pose_free) is not True:
        raise AssertionError

    # Pose at (0, 0, 0) -> distance is 0 < 0.1 -> collision
    pose_coll = np.eye(4)
    if check_collision(checker, pose_coll) is not False:
        raise AssertionError

    # Filter
    poses = np.stack([pose_free, pose_coll], axis=0)
    filtered = filter_collision_free_grasps(checker, poses)
    if not (filtered.shape[0] == 1):
        raise AssertionError
    if not (np.allclose(filtered[0][:3, 3], [1.0, 1.0, 1.0])):
        raise AssertionError


def test_force_closure_metric_returns_valid_output() -> None:
    """Verify force closure LP solver works."""
    judge = build_force_closure_judge(friction_coefficient=0.5, wrench_regularization=1.0)

    # Empty contacts -> not force closed
    if evaluate_force_closure(judge, []) is not False:
        raise AssertionError

    # Contacts that span 6D space (force closure)
    contacts = [
        {"position": np.array([0.05, 0.0, 0.0]), "normal": np.array([-1.0, 0.0, 0.0])},
        {"position": np.array([-0.05, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
    ]

    w_mat = compute_grasp_wrench_matrix(contacts, 0.5)
    if not (w_mat.shape[0] == WRENCH_MATRIX_ROWS):
        raise AssertionError
    if not (w_mat.shape[1] == WRENCH_MATRIX_COLS):
        raise AssertionError  # 2 contacts * 4 pyramid basis vectors

    fc = evaluate_force_closure(judge, contacts)
    if not (isinstance(fc, bool)):
        raise TypeError


def test_stability_judge_basic() -> None:
    """Verify stability judge evaluates linear/angular velocities."""
    judge = build_stability_judge(max_linear_velocity=0.1, max_angular_velocity=0.05)

    vel_stable = np.array([0.02, 0.0, 0.01, 0.0, 0.02, 0.0])
    if evaluate_stability(judge, vel_stable) is not True:
        raise AssertionError

    vel_unstable = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    if evaluate_stability(judge, vel_unstable) is not False:
        raise AssertionError


def test_lift_outcome_judge_basic() -> None:
    """Verify lift outcome judge evaluates height difference."""
    judge = build_lift_outcome_judge(lift_height_threshold=0.05)
    if evaluate_lift_success(judge, 0.1, 0.16) is not True:
        raise AssertionError
    if evaluate_lift_success(judge, 0.1, 0.12) is not False:
        raise AssertionError


def test_aggregate_grasp_success_rate() -> None:
    """Verify success rate computation."""
    per_obj = {"obj1": True, "obj2": False, "obj3": True}
    if not (aggregate_grasp_success_rate(per_obj) == pytest.approx(2.0 / 3.0)):
        raise AssertionError


def test_run_simulation_rejects_missing_robot_xml(minimal_ycb_root: Path) -> None:
    """Verify simulation raises error on missing robot XML."""
    grasp = np.eye(4)
    with pytest.raises(FileNotFoundError):
        simulate_grasp(
            grasp,
            "mustard_bottle",
            minimal_ycb_root,
            Path("missing_robot.xml"),
            None,
            10,
            np.zeros(1),
        )


def test_run_simulation_creates_outcome_report(panda_robot_xml: Path, minimal_ycb_root: Path) -> None:
    """Verify simulate_grasp runs and returns valid outcome dict."""
    grasp = np.eye(4)
    grasp[:3, 3] = [0.0, 0.0, 0.3]
    outcome = simulate_grasp(
        grasp,
        "mustard_bottle",
        minimal_ycb_root,
        panda_robot_xml,
        None,
        5,
        np.zeros(1),
    )
    if "success" not in outcome:
        raise AssertionError
    if "initial_height" not in outcome:
        raise AssertionError
    if "final_height" not in outcome:
        raise AssertionError
    if "contact_count" not in outcome:
        raise AssertionError


def test_run_simulation_validates_success_contract_params(panda_robot_xml: Path, minimal_ycb_root: Path) -> None:
    """Verify simulate_grasp validates lift and stability thresholds."""
    grasp = np.eye(4)
    grasp[:3, 3] = [0.0, 0.0, 0.3]
    with pytest.raises(ValueError, match="lift_height_threshold"):
        simulate_grasp(
            grasp,
            "mustard_bottle",
            minimal_ycb_root,
            panda_robot_xml,
            None,
            5,
            np.zeros(1),
            lift_height_threshold=-0.01,
        )
    with pytest.raises(ValueError, match="max_linear_velocity"):
        simulate_grasp(
            grasp,
            "mustard_bottle",
            minimal_ycb_root,
            panda_robot_xml,
            None,
            5,
            np.zeros(1),
            max_linear_velocity=-0.1,
        )
    with pytest.raises(ValueError, match="max_angular_velocity"):
        simulate_grasp(
            grasp,
            "mustard_bottle",
            minimal_ycb_root,
            panda_robot_xml,
            None,
            5,
            np.zeros(1),
            max_angular_velocity=-0.1,
        )


def test_evaluate_creates_report_from_synthetic_inputs() -> None:
    """Verify evaluate_generated_grasps runs evaluation on batch."""
    grasps = np.stack([np.eye(4), np.eye(4)], axis=0)
    rng = np.random.default_rng()
    obj_pc = rng.standard_normal((10, 3))
    grip_pc = rng.standard_normal((5, 3))

    def mock_contact_provider(_pose: np.ndarray) -> list[dict[str, np.ndarray]]:
        return [
            {
                "position": np.array([0.05, 0.0, 0.0]),
                "normal": np.array([-1.0, 0.0, 0.0]),
            },
        ]

    per_grasp = evaluate_generated_grasps(
        grasps,
        obj_pc,
        grip_pc,
        mock_contact_provider,
        friction_coefficient=0.5,
        lift_height_threshold=0.05,
    )
    if not (len(per_grasp) == EXPECTED_GRASP_COUNT):
        raise AssertionError
    if "collision_free" not in per_grasp[0]:
        raise AssertionError
    if "force_closure" not in per_grasp[0]:
        raise AssertionError

    aggregated = aggregate_evaluation_results({"mustard_bottle": per_grasp})
    if "success_rate" not in aggregated:
        raise AssertionError
    if "collision_free_rate" not in aggregated:
        raise AssertionError

    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = Path(tmp_dir) / "report.jsonl"
        write_evaluation_report(report_path, aggregated)
        if not (report_path.exists()):
            raise AssertionError
        records = read_jsonl_records(report_path)
        summary = next(record for record in records if record.get("record_type") == "summary")
        if "success_rate" not in summary:
            raise AssertionError


def test_phase6_pipelines_do_not_leak_global_state(panda_robot_xml: Path, minimal_ycb_root: Path) -> None:
    """Verify that multiple simulation runs are independent."""
    grasp1 = np.eye(4)
    grasp2 = np.eye(4)
    grasp2[0, 3] = 1.0
    outcome1 = simulate_grasp(
        grasp1,
        "mustard_bottle",
        minimal_ycb_root,
        panda_robot_xml,
        None,
        5,
        np.zeros(1),
    )
    outcome2 = simulate_grasp(
        grasp2,
        "mustard_bottle",
        minimal_ycb_root,
        panda_robot_xml,
        None,
        5,
        np.zeros(1),
    )
    if not (np.allclose(outcome1["grasp_pose"], grasp1)):
        raise AssertionError
    if not (np.allclose(outcome2["grasp_pose"], grasp2)):
        raise AssertionError


def test_simulate_grasp_renames_object_body_to_object_identifier(
    panda_robot_xml: Path,
    minimal_ycb_root_differing_body_name: Path,
) -> None:
    """Verify object body is renamed to the identifier so lookups succeed.

    Args:
        panda_robot_xml: Path to ``deploy/robot.xml``.
        minimal_ycb_root_differing_body_name: YCB root whose object body name
            differs from the object identifier.
    """
    r_model = load_robot_model(str(panda_robot_xml))
    q_home = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04])
    hand_pose = build_forward_kinematics(r_model)(q_home)
    contact_grasp = transform_grasp_pose(hand_pose, panda_hand_to_contact_transform())
    outcome = simulate_grasp(
        contact_grasp,
        "mustard_bottle",
        minimal_ycb_root_differing_body_name,
        panda_robot_xml,
        None,
        5,
        np.zeros(1),
    )
    if not (
        set(outcome.keys())
        == {
            "success",
            "initial_height",
            "final_height",
            "contact_count",
            "object_velocity",
            "grasp_pose",
            "fk_position_error",
        }
    ):
        raise AssertionError
    if not (outcome["initial_height"] == pytest.approx(0.5)):
        raise AssertionError
    if not (outcome["final_height"] == pytest.approx(0.5)):
        raise AssertionError
    if not (np.allclose(outcome["grasp_pose"], contact_grasp)):
        raise AssertionError


def test_simulate_grasp_ik_failure_returns_unsuccessful_outcome(
    panda_robot_xml: Path,
    minimal_ycb_root_at_height: Path,
) -> None:
    """Verify IK failure returns an unsuccessful outcome without simulating."""
    grasp = np.eye(4)
    grasp[:3, 3] = [0.0, 0.0, 1.0]
    outcome = simulate_grasp(
        grasp,
        "mustard_bottle",
        minimal_ycb_root_at_height,
        panda_robot_xml,
        None,
        5,
        np.zeros(1),
    )
    if outcome["success"] is not False:
        raise AssertionError
    if not (outcome["initial_height"] == 0.0):
        raise AssertionError
    if not (outcome["final_height"] == 0.0):
        raise AssertionError
    if not (outcome["contact_count"] == 0.0):
        raise AssertionError
    if not (np.array_equal(outcome["object_velocity"], np.zeros(6))):
        raise AssertionError
    if not (np.allclose(outcome["grasp_pose"], grasp)):
        raise AssertionError


def test_simulate_grasp_ik_failure_aligns_freejoint_object(
    panda_robot_xml: Path,
    minimal_ycb_root_freejoint: Path,
) -> None:
    """Verify unreachable grasps still run physics when the object can teleport."""
    grasp = np.eye(4)
    grasp[:3, 3] = [0.0, 0.0, 1.0]
    outcome = simulate_grasp(
        grasp,
        "mustard_bottle",
        minimal_ycb_root_freejoint,
        panda_robot_xml,
        None,
        5,
        np.zeros(1),
    )
    if outcome["success"] is not False:
        raise AssertionError
    if not (outcome["fk_position_error"] == float("inf")):
        raise AssertionError
    if not (outcome["initial_height"] != 0.0):
        raise AssertionError


def test_simulate_grasp_missing_object_body_reports_error(panda_robot_xml: Path, tmp_path: Path) -> None:
    """Verify a missing object body is reported as an error, not silently zeroed."""
    obj_dir = tmp_path / "ycb" / "006_mustard_bottle"
    obj_dir.mkdir(parents=True)
    xml_content = "<mujoco model='mustard_bottle'><worldbody/></mujoco>"
    (obj_dir / "mustard_bottle.xml").write_text(xml_content, encoding="utf-8")
    ycb_root = tmp_path / "ycb"
    grasp = np.eye(4)
    with pytest.raises(ValueError, match="No body element found"):
        simulate_grasp(
            grasp,
            "mustard_bottle",
            ycb_root,
            panda_robot_xml,
            None,
            5,
            np.zeros(1),
        )
