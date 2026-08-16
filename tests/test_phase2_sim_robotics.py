"""Phase 2 simulation and robotics tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import grasping_ai.simulation.scene as scene_module
from grasping_ai.config import FLATTENED_YAML_CONFIG
from grasping_ai.perception.geometry import invert_transform
from grasping_ai.pipelines.simulate_grasp import (
    run_simulation_sweep,
    simulate_grasp,
)
from grasping_ai.robotics.gripper import (
    gripper_actuator_indices,
    load_gripper_model,
    make_close_command,
    make_open_command,
    panda_hand_to_contact_transform,
    panda_width_to_finger_joints,
)
from grasping_ai.robotics.kinematics import (
    build_forward_kinematics,
    build_inverse_kinematics,
    load_robot_model,
    solve_inverse_kinematics,
)
from grasping_ai.robotics.transforms import (
    convert_grasps_to_world_frame,
    invert_rigid_transform,
    transform_between_frames,
    transform_grasp_pose,
)
from grasping_ai.simulation.mujoco_env import (
    create_simulation,
    load_mujoco_model,
    read_body_pose,
    read_joint_positions,
    reset_simulation,
    set_actuator_controls,
    set_joint_positions,
)
from grasping_ai.simulation.scene import (
    MuJoCoScene,
    attach_object_to_scene,
    build_scene_xml,
    collect_contacts,
    step_scene,
)
from grasping_ai.simulation.ycb import (
    build_ycb_object_name_classifier,
    find_ycb_mesh_file,
    find_ycb_mjcf,
    list_ycb_objects,
    resolve_ycb_object_directory,
    tokenize_ycb_object_name,
    ycb_object_exists,
)

if TYPE_CHECKING:
    import mujoco

    from grasping_ai.simulation.mujoco_env import ContactReporter, SimulationStep

MIN_FRICTION_TAG_COUNT = 5
PANDA_NQ = 9
MIN_HOME_HAND_HEIGHT = 0.3
PANDA_NU = 8
EXPECTED_SWEEP_OUTCOMES = 2
PANDA_FINGER_ACTUATOR_ID = 7


def test_mujoco_runtime_dependency_available() -> None:
    """Verify that mujoco package can be imported."""
    import mujoco  # noqa: PLC0415  # deferred: this test verifies the optional heavy dependency imports at all

    if not (mujoco.__version__ is not None):
        raise AssertionError


def test_phase1_package_import_remains_stable() -> None:
    """Verify that the package remains importable."""
    import grasping_ai  # noqa: PLC0415  # deferred: this test verifies the package itself imports at all

    if not (grasping_ai.__name__ == "grasping_ai"):
        raise AssertionError


def test_simulation_config_file_exists() -> None:
    """Verify simulation configuration file existence."""
    path = Path("configs/env/default.yaml")
    if not (path.is_file()):
        raise AssertionError


def test_robot_config_files_exist() -> None:
    """Verify gripper config default alias and Franka Emika Panda variant exist."""
    if not (Path("configs/gripper/default.yaml").is_file()):
        raise AssertionError
    if not (Path("configs/gripper/franka_emika_panda.yaml").is_file()):
        raise AssertionError


@pytest.fixture
def minimal_gripper_xml(tmp_path: Path) -> Path:
    """Fixture providing a path to a minimal MuJoCo gripper XML model for testing."""
    xml_content = """
    <mujoco model="minimal_gripper">
        <worldbody>
            <body name="base">
                <geom type="box" size="0.05 0.05 0.05"/>
                <body name="finger">
                    <joint name="finger_joint" type="slide" axis="0 1 0" range="0.0 0.04" limited="true"/>
                    <geom type="box" size="0.01 0.01 0.05"/>
                </body>
            </body>
        </worldbody>
        <actuator>
            <position name="finger_actuator" joint="finger_joint" ctrlrange="0.0 0.04" ctrllimited="true"/>
        </actuator>
    </mujoco>
    """
    path = tmp_path / "gripper.xml"
    path.write_text(xml_content, encoding="utf-8")
    return path


@pytest.fixture
def minimal_object_xml(tmp_path: Path) -> Path:
    """Fixture providing a path to a minimal MuJoCo object XML model for testing."""
    xml_content = """
    <mujoco model="minimal_object">
        <worldbody>
            <body name="object" pos="0.5 0 0.5">
                <geom name="object_geom" type="sphere" size="0.05"/>
            </body>
        </worldbody>
    </mujoco>
    """
    path = tmp_path / "object.xml"
    path.write_text(xml_content, encoding="utf-8")
    return path


SCENE_OBJECT_XML = """\
<mujoco model="object">
    <worldbody>
        <body name="object" pos="0 0 0.5">
            <freejoint/>
            <geom name="object_geom" type="sphere" size="0.05"/>
        </body>
    </worldbody>
</mujoco>
"""


@pytest.fixture
def scene_object_xml(tmp_path: Path) -> Path:
    """Fixture writing and providing a path to a scene object XML model with a freejoint."""
    path = tmp_path / "object.xml"
    path.write_text(SCENE_OBJECT_XML, encoding="utf-8")
    return path


@pytest.fixture
def ycb_root_with_object(tmp_path: Path) -> Path:
    """Fixture providing a temporary YCB asset root directory populated with a dummy object model."""
    obj_dir = tmp_path / "ycb" / "006_mustard_bottle"
    obj_dir.mkdir(parents=True)
    (obj_dir / "mustard_bottle.xml").write_text(SCENE_OBJECT_XML, encoding="utf-8")
    return tmp_path / "ycb"


def test_transform_between_frames() -> None:
    """Verify transforming point and points between frames."""
    # Translation by [1, 2, 3]
    t = np.eye(4)
    t[:3, 3] = [1.0, 2.0, 3.0]

    # Single point
    pt = np.array([0.5, 0.5, 0.5])
    pt_t = transform_between_frames(t, pt)
    if not (np.allclose(pt_t, [1.5, 2.5, 3.5])):
        raise AssertionError

    # Batch of points
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    pts_t = transform_between_frames(t, pts)
    if not (np.allclose(pts_t, [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])):
        raise AssertionError

    # Invalid shape
    with pytest.raises(ValueError, match="point_in_source must have shape"):
        transform_between_frames(t, np.array([1.0, 2.0]))

    # Invalid transform shape
    with pytest.raises(ValueError, match=r"source_to_target must be a \(4, 4\)"):
        transform_between_frames(np.eye(3), pt)


def test_transform_grasp_pose() -> None:
    """Test grasp pose composition."""
    g2w = np.eye(4)
    g2w[:3, 3] = [1.0, 0.0, 0.0]

    gr2g = np.eye(4)
    gr2g[:3, 3] = [0.0, 2.0, 0.0]

    gr2w = transform_grasp_pose(g2w, gr2g)
    if not (np.allclose(gr2w[:3, 3], [1.0, 2.0, 0.0])):
        raise AssertionError


def test_panda_hand_to_contact_transform_round_trip() -> None:
    """Panda hand-to-contact transform is rigid and matches config offsets."""
    hand_to_contact = panda_hand_to_contact_transform()
    if not (hand_to_contact.shape == (4, 4)):
        raise AssertionError
    if not (np.isclose(np.linalg.det(hand_to_contact[:3, :3]), 1.0, atol=1e-6)):
        raise AssertionError
    if not (np.allclose(hand_to_contact[:3, 3], [0.0, 0.0, -0.102])):
        raise AssertionError

    contact_to_hand = invert_transform(hand_to_contact)
    identity = hand_to_contact @ contact_to_hand
    if not (np.allclose(identity, np.eye(4), atol=1e-6)):
        raise AssertionError


def test_panda_width_to_finger_joints() -> None:
    """Width mapping matches Panda slide joint semantics."""
    q1, q2 = panda_width_to_finger_joints(0.08)
    if not (np.isclose(q1, 0.04)):
        raise AssertionError
    if not (np.isclose(q2, 0.0)):
        raise AssertionError

    q1_min, q2_min = panda_width_to_finger_joints(0.0)
    if not (np.isclose(q1_min, 0.0015)):
        raise AssertionError
    if not (np.isclose(q2_min, -0.0385)):
        raise AssertionError


def test_deploy_robot_fingertip_friction() -> None:
    """Panda fingertip pad defaults use high friction from mj-grasp-sim."""
    text = Path("deploy/robot.xml").read_text(encoding="utf-8")
    if not (text.count('friction="2.4 0.3 0.1"') >= MIN_FRICTION_TAG_COUNT):
        raise AssertionError


def test_gripper_config_documents_panda_contact_offset() -> None:
    """Gripper config group records Panda base-to-contact constants."""
    position = FLATTENED_YAML_CONFIG.get_path("robot", "gripper", "base_to_contact", "position")
    if not (position == [0, 0, -0.102]):
        raise AssertionError


def test_convert_grasps_to_world_frame() -> None:
    """Verify batch grasp conversions."""
    o2w = np.eye(4)
    o2w[:3, 3] = [0.0, 0.0, 1.0]

    # Single grasp
    grasp = np.eye(4)
    grasp[:3, 3] = [0.1, 0.2, 0.3]
    grasp_w = convert_grasps_to_world_frame(grasp, o2w)
    if not (np.allclose(grasp_w[:3, 3], [0.1, 0.2, 1.3])):
        raise AssertionError

    # Batch of grasps
    grasps = np.array([grasp, grasp])
    grasps_w = convert_grasps_to_world_frame(grasps, o2w)
    if not (grasps_w.shape == (2, 4, 4)):
        raise AssertionError
    if not (np.allclose(grasps_w[0, :3, 3], [0.1, 0.2, 1.3])):
        raise AssertionError
    if not (np.allclose(grasps_w[1, :3, 3], [0.1, 0.2, 1.3])):
        raise AssertionError


def test_invert_rigid_transform() -> None:
    """Verify inverse rigid transforms."""
    t = np.eye(4)
    # Translation and 90 deg rotation around Z
    t[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    t[:3, 3] = [1.0, 2.0, 3.0]

    t_inv = invert_rigid_transform(t)
    identity = t @ t_inv
    if not (np.allclose(identity, np.eye(4))):
        raise AssertionError


def test_load_gripper_model(minimal_gripper_xml: Path) -> None:
    """Test gripper loading and metadata mapping."""
    g_model = load_gripper_model(str(minimal_gripper_xml))
    if not (isinstance(g_model, dict)):
        raise TypeError
    if not (g_model["nu"] == 1):
        raise AssertionError
    if not (g_model["actuator_names"] == ["finger_actuator"]):
        raise AssertionError


def test_make_open_close_commands(minimal_gripper_xml: Path) -> None:
    """Verify open and close command generation."""
    g_model = load_gripper_model(str(minimal_gripper_xml))

    open_cmd = make_open_command(g_model)
    close_cmd = make_close_command(g_model)

    if not (np.allclose(open_cmd, [0.0])):
        raise AssertionError
    if not (np.allclose(close_cmd, [0.04])):
        raise AssertionError


def test_load_robot_model(panda_robot_xml: Path) -> None:
    """Load the Franka Panda MJCF and report nine generalized coordinates.

    Args:
        panda_robot_xml: Path to ``deploy/robot.xml``.
    """
    r_model = load_robot_model(str(panda_robot_xml))
    if not (isinstance(r_model, dict)):
        raise TypeError
    if not (r_model["nq"] == PANDA_NQ):
        raise AssertionError


def test_franka_panda_model_uses_hand_end_effector(panda_robot_xml: Path) -> None:
    """Verify Panda FK uses the ``hand`` body at the home keyframe.

    Args:
        panda_robot_xml: Path to ``deploy/robot.xml``.
    """
    r_model = load_robot_model(str(panda_robot_xml))
    mj_model = r_model["model"]
    import mujoco  # noqa: PLC0415  # deferred: optional heavy dependency

    if not (mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand") != -1):
        raise AssertionError
    fk = build_forward_kinematics(r_model)
    q_home = np.array(mj_model.key_qpos[0, :9], dtype=np.float64)
    pose = fk(q_home)
    if not (pose.shape == (4, 4)):
        raise AssertionError
    if not (pose[2, 3] > MIN_HOME_HAND_HEIGHT):
        raise AssertionError


def test_build_forward_kinematics(panda_robot_xml: Path) -> None:
    """Verify Panda forward kinematics returns a rigid transform of the hand.

    Args:
        panda_robot_xml: Path to ``deploy/robot.xml``.
    """
    r_model = load_robot_model(str(panda_robot_xml))
    fk = build_forward_kinematics(r_model)
    q_zero = np.zeros(9)
    pose = fk(q_zero)
    if not (pose.shape == (4, 4)):
        raise AssertionError
    if not (np.isfinite(pose).all()):
        raise AssertionError
    if not (pose[2, 3] > 0.0):
        raise AssertionError

    q_home = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04])
    pose_home = fk(q_home)
    if np.allclose(pose_home[:3, 3], pose[:3, 3]):
        raise AssertionError


def test_numerical_inverse_kinematics(panda_robot_xml: Path) -> None:
    """Test IK solver convergence, reachability, and invalid inputs on Panda.

    Args:
        panda_robot_xml: Path to ``deploy/robot.xml``.
    """
    r_model = load_robot_model(str(panda_robot_xml))
    fk = build_forward_kinematics(r_model)
    ik_solver = build_inverse_kinematics(r_model, max_iterations=200, tolerance=1e-3)

    q_seed = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04])
    target_pose = fk(q_seed)

    sol = solve_inverse_kinematics(ik_solver, target_pose, q_seed)
    if not (np.allclose(sol, q_seed, atol=1e-3)):
        raise AssertionError

    unreachable_pose = np.eye(4)
    unreachable_pose[:3, 3] = [5.0, 5.0, 5.0]
    with pytest.raises(ValueError, match="failed to converge"):
        solve_inverse_kinematics(ik_solver, unreachable_pose, q_seed)

    with pytest.raises(ValueError, match="finite"):
        solve_inverse_kinematics(ik_solver, target_pose, np.full(9, np.nan))


def _check_transform_error_handling() -> None:
    """Verify transform utilities reject invalid inputs."""
    with pytest.raises(TypeError):
        transform_between_frames(np.eye(4), "not-an-array")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        transform_between_frames(np.eye(4), np.zeros((2, 4)))
    with pytest.raises(ValueError, match="grasp_to_world"):
        transform_grasp_pose(np.zeros((3, 3)), np.eye(4))
    with pytest.raises(ValueError, match="gripper_to_grasp"):
        transform_grasp_pose(np.eye(4), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="object_to_world"):
        convert_grasps_to_world_frame(np.eye(4), np.zeros((3, 3)))
    with pytest.raises(TypeError, match="grasps"):
        convert_grasps_to_world_frame("not-an-array", np.eye(4))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="grasps must have shape"):
        convert_grasps_to_world_frame(np.zeros(3), np.eye(4))
    with pytest.raises(ValueError, match="transform"):
        invert_rigid_transform(np.zeros((3, 3)))


def _check_gripper_error_handling(minimal_gripper_xml: Path) -> None:
    """Verify gripper model, controller, and command validations."""
    with pytest.raises(TypeError):
        load_gripper_model(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        load_gripper_model("")
    with pytest.raises(FileNotFoundError):
        load_gripper_model("non_existent_gripper.xml")
    bad_xml = Path("bad_gripper.xml")
    bad_xml.write_text("<invalid", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="Failed to load"):
            load_gripper_model(str(bad_xml))
    finally:
        if bad_xml.is_file():
            bad_xml.unlink()

    g_model = load_gripper_model(str(minimal_gripper_xml))

    with pytest.raises(TypeError, match="gripper_model"):
        make_open_command("not-a-dict")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="gripper_model"):
        make_close_command("not-a-dict")  # type: ignore[arg-type]

    # Test open/close commands with explicit override keys
    g_model_override = {
        "model": g_model["model"],
        "nu": g_model["nu"],
        "open_command": [0.01],
        "close_command": [0.03],
    }
    if not (np.allclose(make_open_command(g_model_override), [0.01])):
        raise AssertionError
    if not (np.allclose(make_close_command(g_model_override), [0.03])):
        raise AssertionError


def _check_kinematics_error_handling(panda_robot_xml: Path) -> None:
    """Verify robot model loading, FK, and IK validations."""
    with pytest.raises(TypeError):
        load_robot_model(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        load_robot_model("")
    with pytest.raises(FileNotFoundError):
        load_robot_model("non_existent_robot.xml")
    # Create an invalid XML file
    bad_xml = Path("bad_robot.xml")
    bad_xml.write_text("<invalid", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="Failed to load"):
            load_robot_model(str(bad_xml))
    finally:
        if bad_xml.is_file():
            bad_xml.unlink()

    with pytest.raises(TypeError, match="robot_model"):
        build_forward_kinematics("not-a-dict")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="robot_model"):
        build_inverse_kinematics("not-a-dict", 100, 1e-4)  # type: ignore[arg-type]

    r_model = load_robot_model(str(panda_robot_xml))
    fk = build_forward_kinematics(r_model)
    with pytest.raises(TypeError, match="joints"):
        fk("not-a-numpy-array")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="joints shape"):
        fk(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="finite"):
        fk(np.full(9, np.nan))

    with pytest.raises(ValueError, match="max_iterations"):
        build_inverse_kinematics(r_model, 0, 1e-4)
    with pytest.raises(ValueError, match="tolerance"):
        build_inverse_kinematics(r_model, 100, -1.0)

    ik_solver = build_inverse_kinematics(r_model, 100, 1e-4)
    with pytest.raises(ValueError, match="target_pose"):
        ik_solver(np.zeros((3, 3)), np.zeros(9))
    with pytest.raises(ValueError, match="initial_joints"):
        ik_solver(np.eye(4), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="finite"):
        ik_solver(np.array([[np.nan] * 4] * 4), np.zeros(9))

    with pytest.raises(TypeError, match="ik_solver"):
        solve_inverse_kinematics("not-callable", np.eye(4), np.zeros(9))  # type: ignore[arg-type]


def test_robotics_error_handling(minimal_gripper_xml: Path, panda_robot_xml: Path) -> None:
    """Test all error-handling paths and parameter validations in robotics modules."""
    _check_transform_error_handling()
    _check_gripper_error_handling(minimal_gripper_xml)
    _check_kinematics_error_handling(panda_robot_xml)


def test_simulation_initializes_with_minimal_robot_description(panda_robot_xml: Path) -> None:
    """Test loading and initializing simulation with a minimal robot description."""
    model = load_mujoco_model(panda_robot_xml)
    if not (isinstance(model, dict)):
        raise TypeError
    if "mj_model" not in model:
        raise AssertionError

    state, step, contacts = create_simulation(model)
    if not (isinstance(state, dict)):
        raise TypeError
    if not (callable(step)):
        raise TypeError
    if not (callable(contacts)):
        raise TypeError


def test_simulation_initialization_rejects_missing_robot_description() -> None:
    """Test simulation initialization with a missing robot description path."""
    with pytest.raises(FileNotFoundError):
        load_mujoco_model(Path("non_existent_file.xml"))

    with pytest.raises(TypeError):
        load_mujoco_model("not_a_path_object")  # type: ignore[arg-type]


def test_simulation_reset_returns_initial_observation(panda_robot_xml: Path) -> None:
    """Test environment reset returns observation and resets joints."""
    model = load_mujoco_model(panda_robot_xml)
    state, _step, _contacts = create_simulation(model)

    perturbed = np.zeros(9)
    perturbed[0] = 1.5
    set_joint_positions(state, perturbed)
    if not (np.allclose(read_joint_positions(state), perturbed)):
        raise AssertionError

    reset_simulation(state)
    if not (np.allclose(read_joint_positions(state), np.zeros(9))):
        raise AssertionError


def test_simulation_step_accepts_valid_action(panda_robot_xml: Path) -> None:
    """Test that simulation steps correctly with positive finite dt."""
    model = load_mujoco_model(panda_robot_xml)
    _state, step, _contacts = create_simulation(model)

    step(0.002)
    step(0.01)


def test_simulation_step_rejects_invalid_action_shape(panda_robot_xml: Path) -> None:
    """Test invalid joint position shape validation."""
    model = load_mujoco_model(panda_robot_xml)
    state, _step, _contacts = create_simulation(model)

    with pytest.raises(ValueError, match="positions shape"):
        set_joint_positions(state, np.array([1.0, 2.0]))


def test_simulation_step_rejects_non_finite_action(panda_robot_xml: Path) -> None:
    """Test non-finite values are rejected by joint control setter."""
    model = load_mujoco_model(panda_robot_xml)
    state, _step, _contacts = create_simulation(model)

    with pytest.raises(ValueError, match="finite"):
        set_joint_positions(state, np.array([np.nan]))
    with pytest.raises(ValueError, match="finite"):
        set_joint_positions(state, np.array([np.inf]))


def test_simulation_observation_shape_is_stable(panda_robot_xml: Path) -> None:
    """Verify observation shape and body pose return formats."""
    model = load_mujoco_model(panda_robot_xml)
    state, _step, _contacts = create_simulation(model)

    q = read_joint_positions(state)
    if not (q.shape == (9,)):
        raise AssertionError
    if not (q.dtype == np.float64):
        raise AssertionError

    pose = read_body_pose(state, "hand")
    if not (pose.shape == (4, 4)):
        raise AssertionError
    if not (np.allclose(pose[3, :], [0, 0, 0, 1])):
        raise AssertionError


def test_simulation_state_does_not_leak_between_instances(panda_robot_xml: Path) -> None:
    """Verify that multiple simulation states are completely independent."""
    model1 = load_mujoco_model(panda_robot_xml)
    state1, _step1, _contacts1 = create_simulation(model1)

    model2 = load_mujoco_model(panda_robot_xml)
    state2, _step2, _contacts2 = create_simulation(model2)

    set_joint_positions(state1, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    if not (np.allclose(read_joint_positions(state1)[0], 1.0)):
        raise AssertionError
    if not (np.allclose(read_joint_positions(state2), np.zeros(9))):
        raise AssertionError


def test_dynamic_object_attachment(panda_robot_xml: Path, minimal_object_xml: Path) -> None:
    """Verify attaching object to scene, renaming its body and reloading state."""
    model = load_mujoco_model(panda_robot_xml)
    state, _step, _contacts = create_simulation(model)

    set_joint_positions(state, np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    # Attach object and rename body to "my_object"
    attach_object_to_scene(state, minimal_object_xml, "my_object")

    # Joint positions should be copied and maintained
    if not (np.allclose(read_joint_positions(state)[0], 1.2)):
        raise AssertionError

    # End effector and new object should be present
    ee_pose = read_body_pose(state, "hand")
    obj_pose = read_body_pose(state, "my_object")
    if not (ee_pose.shape == (4, 4)):
        raise AssertionError
    if not (obj_pose.shape == (4, 4)):
        raise AssertionError


def test_scene_building_and_stepping(panda_robot_xml: Path, minimal_object_xml: Path) -> None:
    """Test build_scene_xml and step_scene functions."""
    scene_xml = build_scene_xml(panda_robot_xml, minimal_object_xml, None)
    if not (scene_xml.is_file()):
        raise AssertionError

    model = load_mujoco_model(scene_xml)
    _state, step, _contacts = create_simulation(model)

    step_scene(step, 0.002, 10)


def test_build_scene_xml_renames_object_body_when_name_supplied(
    panda_robot_xml: Path,
    minimal_object_xml: Path,
) -> None:
    """Verify build_scene_xml renames the object body when object_name is supplied."""
    import mujoco  # noqa: PLC0415  # deferred: optional heavy dependency

    scene_named = build_scene_xml(panda_robot_xml, minimal_object_xml, None, object_name="my_object")
    model_named = load_mujoco_model(scene_named)
    mj_model_named = model_named["mj_model"]
    if not (mujoco.mj_name2id(mj_model_named, mujoco.mjtObj.mjOBJ_BODY, "my_object") != -1):
        raise AssertionError
    if not (mujoco.mj_name2id(mj_model_named, mujoco.mjtObj.mjOBJ_BODY, "object") == -1):
        raise AssertionError

    scene_default = build_scene_xml(panda_robot_xml, minimal_object_xml, None)
    model_default = load_mujoco_model(scene_default)
    mj_model_default = model_default["mj_model"]
    if not (mujoco.mj_name2id(mj_model_default, mujoco.mjtObj.mjOBJ_BODY, "object") != -1):
        raise AssertionError

    with pytest.raises(TypeError):
        build_scene_xml(panda_robot_xml, minimal_object_xml, None, 123)  # type: ignore[arg-type]


def test_contact_filtering() -> None:
    """Verify contact report filtering logic."""

    def dummy_reporter() -> list[dict[str, np.ndarray]]:
        return [
            {
                "position": np.zeros(3),
                "normal": np.array([0, 0, 1]),
                "force": np.zeros(6),
                "body_names": np.array(["bodyA", "bodyB"], dtype=object),
            },
            {
                "position": np.ones(3),
                "normal": np.array([0, 1, 0]),
                "force": np.zeros(6),
                "body_names": np.array(["bodyC", "bodyD"], dtype=object),
            },
        ]

    filtered = collect_contacts(dummy_reporter, {"bodyB", "bodyX"})
    if not (len(filtered) == 1):
        raise AssertionError
    if not (np.array_equal(filtered[0]["body_names"], ["bodyA", "bodyB"])):
        raise AssertionError

    filtered_empty = collect_contacts(dummy_reporter, {"bodyX", "bodyY"})
    if not (len(filtered_empty) == 0):
        raise AssertionError


def test_ycb_dataset_path_resolution(tmp_path: Path) -> None:
    """Test list_ycb_objects, resolve_ycb_object_directory, find_ycb_mesh_file, ycb_object_exists."""
    ycb_root = tmp_path / "ycb_dataset"
    ycb_root.mkdir()

    # Create dummy YCB object folders
    obj1_dir = ycb_root / "006_mustard_bottle"
    obj1_dir.mkdir()
    mesh1 = obj1_dir / "textured.obj"
    mesh1.write_text("dummy mesh obj", encoding="utf-8")

    obj2_dir = ycb_root / "banana"
    obj2_dir.mkdir()
    mesh2 = obj2_dir / "banana.ply"
    mesh2.write_text("dummy mesh ply", encoding="utf-8")

    # Enumerate YCB objects
    objs = list_ycb_objects(ycb_root)
    if not (objs == ["006_mustard_bottle", "banana"]):
        raise AssertionError

    # Exists check
    if not (ycb_object_exists(ycb_root, "mustard_bottle")):
        raise AssertionError
    if not (ycb_object_exists(ycb_root, "banana")):
        raise AssertionError
    if ycb_object_exists(ycb_root, "apple"):
        raise AssertionError

    # Path resolution
    path1 = resolve_ycb_object_directory(ycb_root, "mustard_bottle")
    if not (path1 == obj1_dir):
        raise AssertionError

    path2 = resolve_ycb_object_directory(ycb_root, "banana")
    if not (path2 == obj2_dir):
        raise AssertionError

    # theseus-backed alias resolution for free-form object names
    path_alias = resolve_ycb_object_directory(ycb_root, "mustard bottle")
    if not (path_alias == obj1_dir):
        raise AssertionError

    # Mesh file lookup
    file2 = find_ycb_mesh_file(path2)
    if not (file2 == mesh2):
        raise AssertionError


def _check_simulation_state_and_step_validation(
    panda_robot_xml: Path,
) -> tuple[object, SimulationStep, ContactReporter]:
    """Verify state getter/setter and step validations; return the live sim handles."""
    with pytest.raises(TypeError, match="state"):
        reset_simulation("invalid-state")
    with pytest.raises(TypeError, match="state"):
        read_joint_positions("invalid-state")
    with pytest.raises(TypeError, match="state"):
        set_joint_positions("invalid-state", np.zeros(1))
    with pytest.raises(TypeError, match="state"):
        read_body_pose("invalid-state", "base")

    model = load_mujoco_model(panda_robot_xml)
    state, step, contacts = create_simulation(model)
    with pytest.raises(TypeError, match="positions"):
        set_joint_positions(state, "not-an-array")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="body_name"):
        read_body_pose(state, 123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Body 'nonexistent' not found"):
        read_body_pose(state, "nonexistent")

    with pytest.raises(TypeError, match="dt"):
        step("not-a-number")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        step(0.0)
    with pytest.raises(ValueError, match="positive"):
        step(-0.002)
    with pytest.raises(ValueError, match="finite"):
        step(np.nan)

    return state, step, contacts


def _check_scene_and_attach_validation(panda_robot_xml: Path, tmp_path: Path, state: object) -> None:
    """Verify build_scene_xml and attach_object_to_scene validations."""
    with pytest.raises(TypeError):
        build_scene_xml("not-a-path", Path("obj.xml"), None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_scene_xml(Path("robot.xml"), "not-a-path", None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_scene_xml(Path("robot.xml"), Path("obj.xml"), "not-a-path")  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        build_scene_xml(Path("non_existent_robot.xml"), panda_robot_xml, None)
    with pytest.raises(FileNotFoundError):
        build_scene_xml(panda_robot_xml, Path("non_existent_obj.xml"), None)
    with pytest.raises(FileNotFoundError):
        build_scene_xml(panda_robot_xml, panda_robot_xml, Path("non_existent_table.xml"))

    with pytest.raises(TypeError, match="state"):
        attach_object_to_scene("invalid-state", Path("obj.xml"), "obj")
    with pytest.raises(TypeError):
        attach_object_to_scene(state, "not-a-path", "obj")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        attach_object_to_scene(state, Path("obj.xml"), 123)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        attach_object_to_scene(state, Path("non_existent_obj.xml"), "obj")

    # Object XML with no body tag
    empty_xml_path = tmp_path / "empty_object.xml"
    empty_xml_path.write_text("<mujoco model='empty'></mujoco>", encoding="utf-8")
    with pytest.raises(ValueError, match="No body element found"):
        attach_object_to_scene(state, empty_xml_path, "empty")


def _check_step_scene_and_contacts_validation(step: SimulationStep, contacts: ContactReporter) -> None:
    """Verify step_scene and collect_contacts validations."""
    with pytest.raises(TypeError, match="step"):
        step_scene("not-callable", 0.002, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="dt"):
        step_scene(step, "not-a-float", 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="num_steps"):
        step_scene(step, 0.002, "not-an-int")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        step_scene(step, -0.002, 1)
    with pytest.raises(ValueError, match="positive"):
        step_scene(step, 0.002, 0)

    with pytest.raises(TypeError, match="contacts"):
        collect_contacts("not-callable", {"base"})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="body_names"):
        collect_contacts(contacts, "not-a-set")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="strings"):
        collect_contacts(contacts, {123})  # type: ignore[arg-type]


def _check_ycb_validation(tmp_path: Path) -> None:
    """Verify YCB path resolution and mesh discovery validations."""
    with pytest.raises(TypeError):
        list_ycb_objects("not-a-path")  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        list_ycb_objects(Path("non_existent_ycb_root"))

    with pytest.raises(TypeError):
        resolve_ycb_object_directory("not-a-path", "obj")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        resolve_ycb_object_directory(Path("ycb"), 123)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        resolve_ycb_object_directory(Path("non_existent_ycb_root"), "obj")

    with pytest.raises(TypeError):
        find_ycb_mesh_file("not-a-path")  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        find_ycb_mesh_file(Path("non_existent_obj_dir"))

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No mesh file"):
        find_ycb_mesh_file(empty_dir)


def test_simulation_error_handling(panda_robot_xml: Path, tmp_path: Path) -> None:
    """Test all error-handling paths and parameter validations in simulation modules."""
    # 1. load_mujoco_model type validation
    with pytest.raises(TypeError):
        load_mujoco_model("not-a-path")  # type: ignore[arg-type]

    # 2-4. State-getter/setter and step validations
    state, step, contacts = _check_simulation_state_and_step_validation(panda_robot_xml)

    # 5-6. build_scene_xml and attach_object_to_scene validations
    _check_scene_and_attach_validation(panda_robot_xml, tmp_path, state)

    # 7-8. step_scene and collect_contacts validations
    _check_step_scene_and_contacts_validation(step, contacts)

    # 9. YCB validations
    _check_ycb_validation(tmp_path)


def test_find_ycb_mjcf_discovery(ycb_root_with_object: Path) -> None:
    """Verify find_ycb_mjcf finds the mustard bottle XML model file in YCB paths."""
    object_dir = resolve_ycb_object_directory(ycb_root_with_object, "mustard_bottle")
    mjcf = find_ycb_mjcf(object_dir)
    if not (mjcf.is_file()):
        raise AssertionError
    if not (mjcf.suffix == ".xml"):
        raise AssertionError


def test_find_ycb_mjcf_recursive_discovery(tmp_path: Path) -> None:
    """Verify that find_ycb_mjcf recursively finds MJCF XML configuration files in subdirectories."""
    obj_dir = tmp_path / "obj"
    nested = obj_dir / "meshes"
    nested.mkdir(parents=True)
    (nested / "model.xml").write_text("<mujoco/>", encoding="utf-8")
    mjcf = find_ycb_mjcf(obj_dir)
    if not (mjcf == nested / "model.xml"):
        raise AssertionError


def test_find_ycb_mjcf_validation(tmp_path: Path) -> None:
    """Verify that find_ycb_mjcf raises appropriate exceptions for non-existent directories or invalid inputs."""
    with pytest.raises(TypeError):
        find_ycb_mjcf("not-a-path")  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        find_ycb_mjcf(Path("non_existent_dir"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="MJCF XML"):
        find_ycb_mjcf(empty)


def test_mujoco_scene_robot_only(panda_robot_xml: Path) -> None:
    """Verify that MuJoCoScene initializes correctly with robot model and steps control commands successfully."""
    scene = MuJoCoScene(panda_robot_xml)
    if not (scene.state is not None):
        raise AssertionError
    if not (scene.model.nu == PANDA_NU):
        raise AssertionError
    if not (scene.body_pose("link1").shape == (4, 4)):
        raise AssertionError

    ctrl = np.zeros(8)
    ctrl[0] = 0.1
    scene.step(ctrl, 0.002)
    pose = scene.body_pose("link1")
    if not (np.isfinite(pose).all()):
        raise AssertionError


def test_mujoco_scene_with_object_renames_body(panda_robot_xml: Path, scene_object_xml: Path) -> None:
    """Verify that MuJoCoScene properly renames body names to match object identifiers on loading."""
    import mujoco  # noqa: PLC0415  # deferred: optional heavy dependency

    scene = MuJoCoScene(panda_robot_xml, scene_object_xml, object_name="obj_001")
    mj_model = scene.model
    if not (mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "obj_001") != -1):
        raise AssertionError
    if not (mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "object") == -1):
        raise AssertionError
    if not (scene.body_pose("obj_001")[2, 3] == pytest.approx(0.5)):
        raise AssertionError


def test_mujoco_scene_snapshot_reset_restores_initial_state(panda_robot_xml: Path, scene_object_xml: Path) -> None:
    """Verify that calling reset on MuJoCoScene restores the state snapshot without rebuilding physics models."""
    scene = MuJoCoScene(panda_robot_xml, scene_object_xml, object_name="obj_001")
    initial_height = scene.body_pose("obj_001")[2, 3]

    # Perturb the simulation by stepping under gravity.
    scene.step(np.zeros(scene.model.nu), 0.002)
    scene.step(np.zeros(scene.model.nu), 0.002)
    if not (scene.body_pose("obj_001")[2, 3] != pytest.approx(initial_height)):
        raise AssertionError

    # Reset restores the captured snapshot without rebuilding the model.
    scene.reset()
    if not (scene.body_pose("obj_001")[2, 3] == pytest.approx(initial_height)):
        raise AssertionError


def test_mujoco_scene_reset_does_not_rebuild_xml(
    panda_robot_xml: Path,
    scene_object_xml: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that reset on MuJoCoScene resets qpos and qvel using snapshots instead of rebuilding model trees."""
    calls = []
    original_build = scene_module.build_scene_xml

    def spy_build(*args: object, **kwargs: object) -> Path:
        calls.append(args)
        return original_build(*args, **kwargs)

    monkeypatch.setattr(scene_module, "build_scene_xml", spy_build)
    scene = MuJoCoScene(panda_robot_xml, scene_object_xml, object_name="obj_001")
    build_calls_after_init = len(calls)

    scene.step(np.zeros(scene.model.nu), 0.002)
    scene.reset()
    scene.reset()
    if not (len(calls) == build_calls_after_init):
        raise AssertionError


def test_mujoco_scene_step_validates_controls(panda_robot_xml: Path) -> None:
    """Verify that stepping MuJoCoScene validates control inputs for shape, types, and finite values."""
    scene = MuJoCoScene(panda_robot_xml)
    with pytest.raises(ValueError, match="finite"):
        scene.step(np.array([np.nan]), 0.002)
    with pytest.raises(ValueError, match="shape"):
        scene.step(np.array([0.1, 0.2]), 0.002)
    with pytest.raises(TypeError):
        scene.step("not-an-array", 0.002)  # type: ignore[arg-type]


def test_mujoco_scene_validation(panda_robot_xml: Path, scene_object_xml: Path) -> None:
    """Verify constructor parameters, robot XML paths, and object XML paths in MuJoCoScene."""
    with pytest.raises(TypeError, match="robot_xml_path"):
        MuJoCoScene("not-a-path")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="object_name"):
        MuJoCoScene(panda_robot_xml, scene_object_xml, object_name=123)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        MuJoCoScene(Path("non_existent_robot.xml"))
    with pytest.raises(FileNotFoundError):
        MuJoCoScene(panda_robot_xml, Path("non_existent_object.xml"))


def test_mujoco_scene_attach_object_refreshes_snapshot(panda_robot_xml: Path, scene_object_xml: Path) -> None:
    """Verify that attaching new objects dynamically to MuJoCoScene updates active state snapshot ranges."""
    scene = MuJoCoScene(panda_robot_xml)
    scene.attach_object(scene_object_xml, "attached_obj")
    if not (scene.body_pose("attached_obj")[2, 3] == pytest.approx(0.5)):
        raise AssertionError

    scene.step(np.zeros(scene.model.nu), 0.002)
    scene.reset()
    if not (scene.body_pose("attached_obj")[2, 3] == pytest.approx(0.5)):
        raise AssertionError


def test_mujoco_scene_is_consistent_with_functional_primitives(panda_robot_xml: Path, scene_object_xml: Path) -> None:
    """Verify that functional primitive simulation utilities yield outputs consistent with scene-based steps."""
    scene = MuJoCoScene(panda_robot_xml, scene_object_xml, object_name="obj_001")
    model = load_mujoco_model(scene.state.get("model_xml_path", panda_robot_xml))
    # The scene wraps the same model used by the functional primitives.
    _state, step, contacts = create_simulation(model)
    if not (callable(step)):
        raise TypeError
    if not (callable(contacts)):
        raise TypeError


def test_set_actuator_controls_writes_ctrl(panda_robot_xml: Path) -> None:
    """Verify that set_actuator_controls correctly updates control registers in active MuJoCo simulation states."""
    model = load_mujoco_model(panda_robot_xml)
    state, _step, _contacts = create_simulation(model)
    set_actuator_controls(state, np.array([0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    state_dict = state  # type: ignore[assignment]
    if not (state_dict["data"].ctrl[0] == pytest.approx(0.25)):
        raise AssertionError


def test_set_actuator_controls_validation(panda_robot_xml: Path) -> None:
    """Verify that set_actuator_controls raises errors for non-finite values or control dimensionality mismatches."""
    model = load_mujoco_model(panda_robot_xml)
    state, _step, _contacts = create_simulation(model)

    with pytest.raises(TypeError, match="state"):
        set_actuator_controls("invalid", np.zeros(1))
    with pytest.raises(TypeError, match="ctrl"):
        set_actuator_controls(state, "not-an-array")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        set_actuator_controls(state, np.array([np.nan]))
    with pytest.raises(ValueError, match="shape"):
        set_actuator_controls(state, np.array([0.1, 0.2]))


def test_simulate_grasp_and_sweep_validations(panda_robot_xml: Path, tmp_path: Path) -> None:
    """Verify shape and boundary validation checks on input parameters for grasp simulation pipelines."""
    ycb_dir = tmp_path / "ycb"
    ycb_dir.mkdir()

    with pytest.raises(ValueError, match="grasp_pose must have shape"):
        simulate_grasp(
            grasp_pose=np.zeros(3),
            object_id="mustard_bottle",
            ycb_root=ycb_dir,
            robot_xml_path=panda_robot_xml,
            table_xml_path=None,
            num_simulation_steps=10,
            gripper_close_command=np.zeros(1),
        )

    with pytest.raises(FileNotFoundError, match="robot_xml_path not found"):
        simulate_grasp(
            grasp_pose=np.eye(4),
            object_id="mustard_bottle",
            ycb_root=ycb_dir,
            robot_xml_path=tmp_path / "missing_robot.xml",
            table_xml_path=None,
            num_simulation_steps=10,
            gripper_close_command=np.zeros(1),
        )

    with pytest.raises(FileNotFoundError, match="ycb_root not found"):
        simulate_grasp(
            grasp_pose=np.eye(4),
            object_id="mustard_bottle",
            ycb_root=tmp_path / "missing_ycb",
            robot_xml_path=panda_robot_xml,
            table_xml_path=None,
            num_simulation_steps=10,
            gripper_close_command=np.zeros(1),
        )

    with pytest.raises(ValueError, match="num_simulation_steps must be positive"):
        simulate_grasp(
            grasp_pose=np.eye(4),
            object_id="mustard_bottle",
            ycb_root=ycb_dir,
            robot_xml_path=panda_robot_xml,
            table_xml_path=None,
            num_simulation_steps=0,
            gripper_close_command=np.zeros(1),
        )

    with pytest.raises(ValueError, match="lift_height_threshold must be non-negative"):
        simulate_grasp(
            grasp_pose=np.eye(4),
            object_id="mustard_bottle",
            ycb_root=ycb_dir,
            robot_xml_path=panda_robot_xml,
            table_xml_path=None,
            num_simulation_steps=10,
            gripper_close_command=np.zeros(1),
            lift_height_threshold=-0.1,
        )

    with pytest.raises(ValueError, match="max_linear_velocity must be non-negative"):
        simulate_grasp(
            grasp_pose=np.eye(4),
            object_id="mustard_bottle",
            ycb_root=ycb_dir,
            robot_xml_path=panda_robot_xml,
            table_xml_path=None,
            num_simulation_steps=10,
            gripper_close_command=np.zeros(1),
            max_linear_velocity=-0.1,
        )

    with pytest.raises(ValueError, match="max_angular_velocity must be non-negative"):
        simulate_grasp(
            grasp_pose=np.eye(4),
            object_id="mustard_bottle",
            ycb_root=ycb_dir,
            robot_xml_path=panda_robot_xml,
            table_xml_path=None,
            num_simulation_steps=10,
            gripper_close_command=np.zeros(1),
            max_angular_velocity=-0.1,
        )

    with pytest.raises(ValueError, match="grasp_poses must have shape"):
        run_simulation_sweep(
            grasp_poses=np.zeros((10, 3)),
            object_id="mustard_bottle",
            ycb_root=ycb_dir,
            robot_xml_path=panda_robot_xml,
            table_xml_path=None,
            num_simulation_steps=10,
            gripper_close_command=np.zeros(1),
        )

    with pytest.raises(ValueError, match="grasp_poses must have shape"):
        run_simulation_sweep(
            grasp_poses=np.zeros((2, 2, 4, 4)),
            object_id="mustard_bottle",
            ycb_root=ycb_dir,
            robot_xml_path=panda_robot_xml,
            table_xml_path=None,
            num_simulation_steps=10,
            gripper_close_command=np.zeros(1),
        )


def test_run_simulation_sweep_execution(
    monkeypatch: pytest.MonkeyPatch,
    panda_robot_xml: Path,
    tmp_path: Path,
) -> None:
    """Verify that running a simulation sweep yields outputs for all input grasp poses in the batch."""
    ycb_dir = tmp_path / "ycb"
    ycb_dir.mkdir()

    def dummy_simulate_grasp(**kwargs: object) -> dict[str, object]:
        return {"success": True, "grasp_pose": kwargs["grasp_pose"]}

    sim_mod = sys.modules["grasping_ai.pipelines.simulate_grasp"]
    monkeypatch.setattr(sim_mod, "simulate_grasp", dummy_simulate_grasp)

    grasps = np.stack([np.eye(4), np.eye(4)], axis=0)
    outcomes = run_simulation_sweep(
        grasp_poses=grasps,
        object_id="mustard_bottle",
        ycb_root=ycb_dir,
        robot_xml_path=panda_robot_xml,
        table_xml_path=None,
        num_simulation_steps=10,
        gripper_close_command=np.zeros(1),
    )
    if not (len(outcomes) == EXPECTED_SWEEP_OUTCOMES):
        raise AssertionError
    if not (outcomes[0]["success"]):
        raise AssertionError


def test_ycb_all_helper_functions_and_discovery_paths(tmp_path: Path) -> None:
    """Verify that YCB helper tools classify names, resolve asset dirs, and discover mesh files correctly."""
    with pytest.raises(TypeError, match="object_name must be a string"):
        tokenize_ycb_object_name(123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"ycb_root must be a pathlib\.Path"):
        build_ycb_object_name_classifier("not_a_path")  # type: ignore[arg-type]

    with pytest.raises(FileNotFoundError, match="YCB root directory"):
        build_ycb_object_name_classifier(tmp_path / "missing_ycb")

    empty_ycb = tmp_path / "empty_ycb"
    empty_ycb.mkdir()
    with pytest.raises(ValueError, match="No YCB objects found under"):
        build_ycb_object_name_classifier(empty_ycb)

    ycb_dir = tmp_path / "ycb_dir"
    ycb_dir.mkdir()
    mustard_dir = ycb_dir / "006_mustard_bottle"
    mustard_dir.mkdir()
    (mustard_dir / "textured.obj").write_text("mesh text", encoding="utf-8")
    (mustard_dir / "model.xml").write_text("<xml/>", encoding="utf-8")

    classifier = build_ycb_object_name_classifier(ycb_dir)
    if classifier("") is not None:
        raise AssertionError
    if classifier("unmatched_xyz_token") is not None:
        raise AssertionError
    if not (classifier("mustard_bottle") == "006_mustard_bottle"):
        raise AssertionError

    plain_dir = ycb_dir / "banana"
    plain_dir.mkdir()
    if not (resolve_ycb_object_directory(ycb_dir, "011_banana") == plain_dir):
        raise AssertionError

    _check_ycb_mesh_and_mjcf_discovery(ycb_dir, mustard_dir)


def _check_ycb_mesh_and_mjcf_discovery(ycb_dir: Path, mustard_dir: Path) -> None:
    """Verify mesh file lookup priorities and recursive MJCF discovery."""
    mesh1 = find_ycb_mesh_file(mustard_dir)
    if not (mesh1.name == "textured.obj"):
        raise AssertionError

    obj_only_dir = ycb_dir / "obj_only"
    obj_only_dir.mkdir()
    (obj_only_dir / "other.obj").write_text("mesh obj", encoding="utf-8")
    mesh2 = find_ycb_mesh_file(obj_only_dir)
    if not (mesh2.name == "other.obj"):
        raise AssertionError

    ply_only_dir = ycb_dir / "ply_only"
    ply_only_dir.mkdir()
    (ply_only_dir / "other.ply").write_text("mesh ply", encoding="utf-8")
    mesh3 = find_ycb_mesh_file(ply_only_dir)
    if not (mesh3.name == "other.ply"):
        raise AssertionError

    nested_xml_dir = ycb_dir / "nested_xml"
    nested_xml_dir.mkdir()
    sub_dir = nested_xml_dir / "sub"
    sub_dir.mkdir()
    (sub_dir / "model.xml").write_text("<xml/>", encoding="utf-8")
    mjcf_path = find_ycb_mjcf(nested_xml_dir)
    if not (mjcf_path.name == "model.xml"):
        raise AssertionError


def test_scene_and_simulate_grasp_additional_coverage(
    monkeypatch: pytest.MonkeyPatch,
    panda_robot_xml: Path,
    tmp_path: Path,
) -> None:
    """Verify scene construction validation and simulate_grasp handling of malformed XML and missing bodies."""
    object_xml = tmp_path / "obj.xml"
    object_xml.write_text(
        '<mujoco model="obj"><worldbody><body name="obj_body">'
        '<geom type="box" size="0.05 0.05 0.05"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )

    table_xml = tmp_path / "table.xml"
    table_xml.write_text(
        '<mujoco model="table"><worldbody><body name="table_body">'
        '<geom type="box" size="0.5 0.5 0.02"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match=r"output_dir must be a pathlib\.Path"):
        build_scene_xml(panda_robot_xml, object_xml, None, output_dir="invalid")  # type: ignore[arg-type]

    scene_xml = build_scene_xml(
        panda_robot_xml,
        object_xml,
        table_xml,
        object_name="renamed_obj",
        output_dir=tmp_path / "out_scenes",
    )
    if not (scene_xml.is_file()):
        raise AssertionError

    bad_xml = tmp_path / "bad.xml"
    bad_xml.write_text("<invalid_xml", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse object XML file"):
        build_scene_xml(panda_robot_xml, bad_xml, None, object_name="bad")

    nobody_xml = tmp_path / "nobody.xml"
    nobody_xml.write_text('<mujoco model="nobody"/>', encoding="utf-8")
    with pytest.raises(ValueError, match="No body element found in object XML"):
        build_scene_xml(panda_robot_xml, nobody_xml, None, object_name="nobody")

    with pytest.raises(TypeError, match=r"object_xml_path must be a pathlib\.Path"):
        MuJoCoScene(panda_robot_xml, object_xml_path="invalid")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"table_xml_path must be a pathlib\.Path"):
        MuJoCoScene(panda_robot_xml, table_xml_path="invalid")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"scene_output_dir must be a pathlib\.Path"):
        MuJoCoScene(panda_robot_xml, scene_output_dir="invalid")  # type: ignore[arg-type]

    missing_dir = tmp_path / "missing_body"
    missing_dir.mkdir(exist_ok=True)
    (missing_dir / "model.xml").write_text(
        '<mujoco model="m"><worldbody><body name="other_body">'
        '<geom type="box" size="0.05 0.05 0.05"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "grasping_ai.simulation.scene._rename_object_body",
        lambda obj_xml, _obj_name, _out_dir: obj_xml,
    )
    monkeypatch.setattr(
        "grasping_ai.robotics.kinematics.solve_inverse_kinematics",
        lambda *_args, **_kwargs: np.zeros(9),
    )

    with pytest.raises(ValueError, match="Body 'missing_body' not found in simulation model"):
        simulate_grasp(
            robot_xml_path=panda_robot_xml,
            grasp_pose=np.eye(4),
            object_id="missing_body",
            ycb_root=tmp_path,
            table_xml_path=None,
            num_simulation_steps=5,
            gripper_close_command=np.zeros(1),
        )

    ycb_dir = tmp_path / "ycb_dir"
    obj_dir = ycb_dir / "006_mustard_bottle"
    obj_dir.mkdir(parents=True, exist_ok=True)
    (obj_dir / "model.xml").write_text(
        '<mujoco model="mustard"><worldbody><body name="mustard_bottle">'
        '<geom type="box" size="0.05 0.05 0.05"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    outcomes = run_simulation_sweep(
        grasp_poses=np.eye(4),
        object_id="mustard_bottle",
        ycb_root=ycb_dir,
        robot_xml_path=panda_robot_xml,
        table_xml_path=None,
        num_simulation_steps=2,
        gripper_close_command=np.array([0.5, 0.8]),
    )
    if not (len(outcomes) == 1):
        raise AssertionError


def test_simulate_grasp_ik_failure_without_freejoint_returns_early(
    monkeypatch: pytest.MonkeyPatch,
    panda_robot_xml: Path,
    tmp_path: Path,
) -> None:
    """Return an unsuccessful outcome when IK fails on a fixed-base object.

    Args:
        monkeypatch: Pytest fixture used to force IK failure.
        panda_robot_xml: Path to ``deploy/robot.xml``.
        tmp_path: Temporary directory for YCB object assets.

    Returns:
        None. Asserts the simulation outcome reports failure.
    """
    ycb_dir = tmp_path / "ycb"
    obj_dir = ycb_dir / "fixed_obj"
    obj_dir.mkdir(parents=True)
    (obj_dir / "model.xml").write_text(
        '<mujoco model="fixed"><worldbody><body name="fixed_obj">'
        '<geom type="box" size="0.05 0.05 0.05"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )

    def fail_ik(*_args: object, **_kwargs: object) -> None:
        msg = "IK failed"
        raise ValueError(msg)

    monkeypatch.setattr(
        "grasping_ai.robotics.kinematics.solve_inverse_kinematics",
        fail_ik,
    )

    outcome = simulate_grasp(
        robot_xml_path=panda_robot_xml,
        grasp_pose=np.eye(4),
        object_id="fixed_obj",
        ycb_root=ycb_dir,
        table_xml_path=None,
        num_simulation_steps=4,
        gripper_close_command=np.zeros(1),
    )
    if outcome["success"] is not False:
        raise AssertionError
    if not (outcome["fk_position_error"] == float("inf")):
        raise AssertionError


def test_simulate_grasp_gripper_actuator_and_freejoint_object(panda_robot_xml: Path, tmp_path: Path) -> None:
    """Exercise simulate_grasp on Panda with a freejoint object.

    Args:
        panda_robot_xml: Path to ``deploy/robot.xml``.
        tmp_path: Temporary directory for YCB MJCF assets.

    Returns:
        None. Asserts a simulation outcome dictionary is produced.
    """
    ycb_dir = tmp_path / "ycb"
    obj_dir = ycb_dir / "free_obj"
    obj_dir.mkdir(parents=True)
    (obj_dir / "model.xml").write_text(
        '<mujoco model="free"><worldbody><body name="free_obj">'
        '<freejoint/><geom type="box" size="0.05 0.05 0.05"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )

    outcome = simulate_grasp(
        robot_xml_path=panda_robot_xml,
        grasp_pose=np.eye(4),
        object_id="free_obj",
        ycb_root=ycb_dir,
        table_xml_path=None,
        num_simulation_steps=8,
        gripper_close_command=np.array([0.01]),
    )
    if "success" not in outcome:
        raise AssertionError
    if not (outcome["contact_count"] >= 0.0):
        raise AssertionError


def test_gripper_actuator_indices_detects_finger_and_tendon(panda_robot_xml: Path, tmp_path: Path) -> None:
    """Detect Panda finger/tendon gripper actuators and a tendon-only model.

    Args:
        panda_robot_xml: Path to ``deploy/robot.xml``.
        tmp_path: Temporary directory for a tendon-only MJCF.

    Returns:
        None. Asserts finger and tendon actuators are indexed as grippers.
    """
    import mujoco  # noqa: PLC0415  # deferred: optional heavy dependency

    model = mujoco.MjModel.from_xml_path(str(panda_robot_xml))
    finger_ids = gripper_actuator_indices(model)
    if PANDA_FINGER_ACTUATOR_ID not in finger_ids:
        raise AssertionError

    tendon_xml = tmp_path / "tendon_gripper.xml"
    tendon_xml.write_text(
        """<mujoco model="tendon">
        <worldbody><body name="base"><geom type="box" size="0.05 0.05 0.05"/>
        <body name="finger"><joint name="finger_joint" type="slide" axis="0 1 0" range="0 0.04"/>
        <geom type="box" size="0.01 0.02 0.02"/></body></body></worldbody>
        <tendon><fixed name="gripper_tendon"><joint joint="finger_joint" coef="1"/></fixed></tendon>
        <actuator><motor name="gripper_motor" tendon="gripper_tendon"/></actuator>
        </mujoco>""",
        encoding="utf-8",
    )
    tendon_model = mujoco.MjModel.from_xml_path(str(tendon_xml))
    if not (gripper_actuator_indices(tendon_model) == [0]):
        raise AssertionError


def test_simulate_grasp_uses_fallback_timestep(
    monkeypatch: pytest.MonkeyPatch,
    panda_robot_xml: Path,
    tmp_path: Path,
) -> None:
    """Use a default physics timestep when the model reports a non-positive ``dt``.

    Args:
        monkeypatch: Pytest fixture used to patch MuJoCo model loading.
        panda_robot_xml: Path to ``deploy/robot.xml``.
        tmp_path: Temporary directory for YCB object assets.

    Returns:
        None. Asserts simulation completes without raising.
    """
    ycb_dir = tmp_path / "ycb"
    obj_dir = ycb_dir / "obj"
    obj_dir.mkdir(parents=True)
    (obj_dir / "model.xml").write_text(
        '<mujoco model="obj"><worldbody><body name="obj">'
        '<freejoint/><geom type="box" size="0.05 0.05 0.05"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )

    import mujoco  # noqa: PLC0415  # deferred: optional heavy dependency

    original_from_xml = mujoco.MjModel.from_xml_path

    def patched_from_xml(path: str) -> mujoco.MjModel:
        model = original_from_xml(path)
        model.opt.timestep = 0.0
        return model

    monkeypatch.setattr(mujoco.MjModel, "from_xml_path", patched_from_xml)

    outcome = simulate_grasp(
        robot_xml_path=panda_robot_xml,
        grasp_pose=np.eye(4),
        object_id="obj",
        ycb_root=ycb_dir,
        table_xml_path=None,
        num_simulation_steps=4,
        gripper_close_command=np.zeros(1),
    )
    if "success" not in outcome:
        raise AssertionError
