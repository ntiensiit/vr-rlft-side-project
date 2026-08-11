from pathlib import Path

import numpy as np
import pytest

from grasping_ai.robotics.gripper import (
    build_gripper_controller,
    load_gripper_model,
    make_close_command,
    make_open_command,
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


@pytest.fixture
def minimal_gripper_xml(tmp_path):
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
def minimal_robot_xml(tmp_path):
    xml_content = """
    <mujoco model="minimal_robot">
        <compiler angle="radian"/>
        <worldbody>
            <body name="base" pos="0 0 0">
                <geom name="base_geom" type="box" size="0.1 0.1 0.1"/>
                <body name="link1" pos="0 0 0.2">
                    <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14" limited="true"/>
                    <geom name="link1_geom" type="cylinder" size="0.05 0.1"/>
                    <body name="end_effector" pos="0 0 0.2"/>
                </body>
            </body>
        </worldbody>
    </mujoco>
    """
    path = tmp_path / "robot.xml"
    path.write_text(xml_content, encoding="utf-8")
    return path


# --- Transforms Tests ---

def test_transform_between_frames():
    """Verify transforming point and points between frames."""
    # Translation by [1, 2, 3]
    t = np.eye(4)
    t[:3, 3] = [1.0, 2.0, 3.0]

    # Single point
    pt = np.array([0.5, 0.5, 0.5])
    pt_t = transform_between_frames(t, pt)
    assert np.allclose(pt_t, [1.5, 2.5, 3.5])

    # Batch of points
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    pts_t = transform_between_frames(t, pts)
    assert np.allclose(pts_t, [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])

    # Invalid shape
    with pytest.raises(ValueError):
        transform_between_frames(t, np.array([1.0, 2.0]))

    # Invalid transform shape
    with pytest.raises(ValueError):
        transform_between_frames(np.eye(3), pt)


def test_transform_grasp_pose():
    """Test grasp pose composition."""
    g2w = np.eye(4)
    g2w[:3, 3] = [1.0, 0.0, 0.0]

    gr2g = np.eye(4)
    gr2g[:3, 3] = [0.0, 2.0, 0.0]

    gr2w = transform_grasp_pose(g2w, gr2g)
    assert np.allclose(gr2w[:3, 3], [1.0, 2.0, 0.0])


def test_convert_grasps_to_world_frame():
    """Verify batch grasp conversions."""
    o2w = np.eye(4)
    o2w[:3, 3] = [0.0, 0.0, 1.0]

    # Single grasp
    grasp = np.eye(4)
    grasp[:3, 3] = [0.1, 0.2, 0.3]
    grasp_w = convert_grasps_to_world_frame(grasp, o2w)
    assert np.allclose(grasp_w[:3, 3], [0.1, 0.2, 1.3])

    # Batch of grasps
    grasps = np.array([grasp, grasp])
    grasps_w = convert_grasps_to_world_frame(grasps, o2w)
    assert grasps_w.shape == (2, 4, 4)
    assert np.allclose(grasps_w[0, :3, 3], [0.1, 0.2, 1.3])
    assert np.allclose(grasps_w[1, :3, 3], [0.1, 0.2, 1.3])


def test_invert_rigid_transform():
    """Verify inverse rigid transforms."""
    t = np.eye(4)
    # Translation and 90 deg rotation around Z
    t[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    t[:3, 3] = [1.0, 2.0, 3.0]

    t_inv = invert_rigid_transform(t)
    identity = t @ t_inv
    assert np.allclose(identity, np.eye(4))


# --- Gripper Tests ---

def test_load_gripper_model(minimal_gripper_xml):
    """Test gripper loading and metadata mapping."""
    g_model = load_gripper_model(str(minimal_gripper_xml))
    assert isinstance(g_model, dict)
    assert g_model["nu"] == 1
    assert g_model["actuator_names"] == ["finger_actuator"]


def test_make_open_close_commands(minimal_gripper_xml):
    """Verify open and close command generation."""
    g_model = load_gripper_model(str(minimal_gripper_xml))

    open_cmd = make_open_command(g_model)
    close_cmd = make_close_command(g_model)

    assert np.allclose(open_cmd, [0.0])
    assert np.allclose(close_cmd, [0.04])


def test_build_gripper_controller(minimal_gripper_xml):
    """Verify command limits, mapping, and application to ctrl array."""
    import mujoco
    g_model = load_gripper_model(str(minimal_gripper_xml))
    controller = build_gripper_controller(g_model)

    # Bind data
    mj_model = g_model["model"]
    mj_data = mujoco.MjData(mj_model)
    g_model["model"] = mj_model
    g_model["data"] = mj_data

    # Valid command
    controller(np.array([0.02]))
    assert np.allclose(mj_data.ctrl[0], 0.02)

    # Invalid inputs
    with pytest.raises(ValueError, match="finite"):
        controller(np.array([np.nan]))
    with pytest.raises(ValueError, match="shape"):
        controller(np.array([0.01, 0.02]))


# --- Kinematics Tests ---

def test_load_robot_model(minimal_robot_xml):
    """Test robot model loading."""
    r_model = load_robot_model(str(minimal_robot_xml))
    assert isinstance(r_model, dict)
    assert r_model["nq"] == 1


def test_build_forward_kinematics(minimal_robot_xml):
    """Verify forward kinematics computation."""
    r_model = load_robot_model(str(minimal_robot_xml))
    fk = build_forward_kinematics(r_model)

    # Base at 0, link1 at z=0.2, ee at z=0.2 from link1 -> ee pos=[0, 0, 0.4] for joint=0
    pose = fk(np.array([0.0]))
    assert np.allclose(pose[:3, 3], [0.0, 0.0, 0.4])

    # Rotation test
    pose_rot = fk(np.array([np.pi / 2]))
    assert np.allclose(pose_rot[:3, 3], [0.0, 0.0, 0.4])
    # Z rotation matrix
    expected_rot = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    assert np.allclose(pose_rot[:3, :3], expected_rot)


def test_numerical_inverse_kinematics(minimal_robot_xml):
    """Test IK solver convergence, reachability, limit clipping, and invalid inputs."""
    r_model = load_robot_model(str(minimal_robot_xml))
    fk = build_forward_kinematics(r_model)
    ik_solver = build_inverse_kinematics(r_model, max_iterations=100, tolerance=1e-4)

    # Generate a target pose using FK
    target_joints = np.array([0.5])
    target_pose = fk(target_joints)

    # Solve from seed=0
    sol = solve_inverse_kinematics(ik_solver, target_pose, np.array([0.0]))
    assert np.allclose(sol, target_joints, atol=1e-3)

    # Unreachable target should raise ValueError
    unreachable_pose = np.eye(4)
    unreachable_pose[:3, 3] = [5.0, 5.0, 5.0]  # Far outside workspace
    with pytest.raises(ValueError, match="failed to converge"):
        solve_inverse_kinematics(ik_solver, unreachable_pose, np.array([0.0]))

    # Invalid input validations
    with pytest.raises(ValueError, match="finite"):
        solve_inverse_kinematics(ik_solver, target_pose, np.array([np.nan]))


def test_robotics_error_handling(minimal_gripper_xml, minimal_robot_xml):
    """Test all error-handling paths and parameter validations in robotics modules."""
    # 1. Transforms error-handling
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

    # 2. Gripper error-handling
    with pytest.raises(TypeError):
        load_gripper_model(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        load_gripper_model("")
    with pytest.raises(FileNotFoundError):
        load_gripper_model("non_existent_gripper.xml")
    with pytest.raises(ValueError, match="Failed to load"):
        bad_xml = Path("bad_gripper.xml")
        bad_xml.write_text("<invalid", encoding="utf-8")
        try:
            load_gripper_model(str(bad_xml))
        finally:
            if bad_xml.is_file():
                bad_xml.unlink()

    with pytest.raises(TypeError, match="gripper_model"):
        build_gripper_controller("not-a-dict")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="gripper_model"):
        make_open_command("not-a-dict")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="gripper_model"):
        make_close_command("not-a-dict")  # type: ignore[arg-type]

    g_model = load_gripper_model(str(minimal_gripper_xml))
    controller = build_gripper_controller(g_model)
    with pytest.raises(TypeError, match="command"):
        controller("not-a-numpy-array")  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="not bound"):
        controller(np.array([0.02]))

    # Test open/close commands with explicit override keys
    g_model_override = {
        "model": g_model["model"],
        "nu": g_model["nu"],
        "open_command": [0.01],
        "close_command": [0.03],
    }
    assert np.allclose(make_open_command(g_model_override), [0.01])
    assert np.allclose(make_close_command(g_model_override), [0.03])

    # 3. Kinematics error-handling
    with pytest.raises(TypeError):
        load_robot_model(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        load_robot_model("")
    with pytest.raises(FileNotFoundError):
        load_robot_model("non_existent_robot.xml")
    with pytest.raises(ValueError, match="Failed to load"):
        # Create an invalid XML file
        bad_xml = Path("bad_robot.xml")
        bad_xml.write_text("<invalid", encoding="utf-8")
        try:
            load_robot_model(str(bad_xml))
        finally:
            if bad_xml.is_file():
                bad_xml.unlink()

    with pytest.raises(TypeError, match="robot_model"):
        build_forward_kinematics("not-a-dict")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="robot_model"):
        build_inverse_kinematics("not-a-dict", 100, 1e-4)  # type: ignore[arg-type]

    r_model = load_robot_model(str(minimal_robot_xml))
    fk = build_forward_kinematics(r_model)
    with pytest.raises(TypeError, match="joints"):
        fk("not-a-numpy-array")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="joints shape"):
        fk(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="finite"):
        fk(np.array([np.nan]))

    with pytest.raises(ValueError, match="max_iterations"):
        build_inverse_kinematics(r_model, 0, 1e-4)
    with pytest.raises(ValueError, match="tolerance"):
        build_inverse_kinematics(r_model, 100, -1.0)

    ik_solver = build_inverse_kinematics(r_model, 100, 1e-4)
    with pytest.raises(ValueError, match="target_pose"):
        ik_solver(np.zeros((3, 3)), np.array([0.0]))
    with pytest.raises(ValueError, match="initial_joints"):
        ik_solver(np.eye(4), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="finite"):
        ik_solver(np.array([[np.nan]*4]*4), np.array([0.0]))

    with pytest.raises(TypeError, match="ik_solver"):
        solve_inverse_kinematics("not-callable", np.eye(4), np.array([0.0]))  # type: ignore[arg-type]

