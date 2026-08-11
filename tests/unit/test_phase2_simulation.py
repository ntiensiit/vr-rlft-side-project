from pathlib import Path

import numpy as np
import pytest

from grasping_ai.simulation.mujoco_env import (
    create_simulation,
    load_mujoco_model,
    read_body_pose,
    read_joint_positions,
    reset_simulation,
    set_joint_positions,
)
from grasping_ai.simulation.scene import (
    attach_object_to_scene,
    build_scene_xml,
    collect_contacts,
    step_scene,
)
from grasping_ai.simulation.ycb import (
    find_ycb_mesh_file,
    list_ycb_objects,
    resolve_ycb_object_directory,
    ycb_object_exists,
)


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


@pytest.fixture
def minimal_object_xml(tmp_path):
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


def test_mujoco_runtime_dependency_available():
    """Verify that mujoco package can be imported."""
    import mujoco
    assert mujoco.__version__ is not None


def test_phase1_package_import_remains_stable():
    """Verify that the package remains importable."""
    import grasping_ai
    assert grasping_ai.__name__ == "grasping_ai"


def test_simulation_config_file_exists():
    """Verify simulation configuration file existence."""
    path = Path("configs/simulation.yaml")
    assert path.is_file()


def test_robot_config_file_exists():
    """Verify robot configuration file existence."""
    path = Path("configs/robot.yaml")
    assert path.is_file()


def test_simulation_initializes_with_minimal_robot_description(minimal_robot_xml):
    """Test loading and initializing simulation with a minimal robot description."""
    model = load_mujoco_model(minimal_robot_xml)
    assert isinstance(model, dict)
    assert "mj_model" in model

    state, step, contacts = create_simulation(model)
    assert isinstance(state, dict)
    assert callable(step)
    assert callable(contacts)


def test_simulation_initialization_rejects_missing_robot_description():
    """Test simulation initialization with a missing robot description path."""
    with pytest.raises(FileNotFoundError):
        load_mujoco_model(Path("non_existent_file.xml"))

    with pytest.raises(TypeError):
        load_mujoco_model("not_a_path_object")  # type: ignore[arg-type]


def test_simulation_reset_returns_initial_observation(minimal_robot_xml):
    """Test environment reset returns observation and resets joints."""
    model = load_mujoco_model(minimal_robot_xml)
    state, _step, _contacts = create_simulation(model)

    # Modify joints
    set_joint_positions(state, np.array([1.5]))
    assert np.allclose(read_joint_positions(state), [1.5])

    reset_simulation(state)
    assert np.allclose(read_joint_positions(state), [0.0])


def test_simulation_step_accepts_valid_action(minimal_robot_xml):
    """Test that simulation steps correctly with positive finite dt."""
    model = load_mujoco_model(minimal_robot_xml)
    _state, step, _contacts = create_simulation(model)

    step(0.002)
    step(0.01)


def test_simulation_step_rejects_invalid_action_shape(minimal_robot_xml):
    """Test invalid joint position shape validation."""
    model = load_mujoco_model(minimal_robot_xml)
    state, _step, _contacts = create_simulation(model)

    with pytest.raises(ValueError, match="positions shape"):
        set_joint_positions(state, np.array([1.0, 2.0]))


def test_simulation_step_rejects_non_finite_action(minimal_robot_xml):
    """Test non-finite values are rejected by joint control setter."""
    model = load_mujoco_model(minimal_robot_xml)
    state, _step, _contacts = create_simulation(model)

    with pytest.raises(ValueError, match="finite"):
        set_joint_positions(state, np.array([np.nan]))
    with pytest.raises(ValueError, match="finite"):
        set_joint_positions(state, np.array([np.inf]))


def test_simulation_observation_shape_is_stable(minimal_robot_xml):
    """Verify observation shape and body pose return formats."""
    model = load_mujoco_model(minimal_robot_xml)
    state, _step, _contacts = create_simulation(model)

    q = read_joint_positions(state)
    assert q.shape == (1,)
    assert q.dtype == np.float64

    pose = read_body_pose(state, "end_effector")
    assert pose.shape == (4, 4)
    assert np.allclose(pose[3, :], [0, 0, 0, 1])


def test_simulation_state_does_not_leak_between_instances(minimal_robot_xml):
    """Verify that multiple simulation states are completely independent."""
    model1 = load_mujoco_model(minimal_robot_xml)
    state1, _step1, _contacts1 = create_simulation(model1)

    model2 = load_mujoco_model(minimal_robot_xml)
    state2, _step2, _contacts2 = create_simulation(model2)

    set_joint_positions(state1, np.array([1.0]))
    assert np.allclose(read_joint_positions(state1), [1.0])
    assert np.allclose(read_joint_positions(state2), [0.0])


def test_dynamic_object_attachment(minimal_robot_xml, minimal_object_xml):
    """Verify attaching object to scene, renaming its body and reloading state."""
    model = load_mujoco_model(minimal_robot_xml)
    state, _step, _contacts = create_simulation(model)

    set_joint_positions(state, np.array([1.2]))

    # Attach object and rename body to "my_object"
    attach_object_to_scene(state, minimal_object_xml, "my_object")

    # Joint positions should be copied and maintained
    assert np.allclose(read_joint_positions(state), [1.2])

    # End effector and new object should be present
    ee_pose = read_body_pose(state, "end_effector")
    obj_pose = read_body_pose(state, "my_object")
    assert ee_pose.shape == (4, 4)
    assert obj_pose.shape == (4, 4)


def test_scene_building_and_stepping(minimal_robot_xml, minimal_object_xml):
    """Test build_scene_xml and step_scene functions."""
    scene_xml = build_scene_xml(minimal_robot_xml, minimal_object_xml, None)
    assert scene_xml.is_file()

    model = load_mujoco_model(scene_xml)
    _state, step, _contacts = create_simulation(model)

    step_scene(step, 0.002, 10)


def test_contact_filtering():
    """Verify contact report filtering logic."""
    def dummy_reporter():
        return [
            {
                "position": np.zeros(3),
                "normal": np.array([0, 0, 1]),
                "force": np.zeros(6),
                "body_names": np.array(["bodyA", "bodyB"], dtype=object)
            },
            {
                "position": np.ones(3),
                "normal": np.array([0, 1, 0]),
                "force": np.zeros(6),
                "body_names": np.array(["bodyC", "bodyD"], dtype=object)
            }
        ]

    filtered = collect_contacts(dummy_reporter, {"bodyB", "bodyX"})
    assert len(filtered) == 1
    assert np.array_equal(filtered[0]["body_names"], ["bodyA", "bodyB"])

    filtered_empty = collect_contacts(dummy_reporter, {"bodyX", "bodyY"})
    assert len(filtered_empty) == 0


def test_ycb_dataset_path_resolution(tmp_path):
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
    assert objs == ["006_mustard_bottle", "banana"]

    # Exists check
    assert ycb_object_exists(ycb_root, "mustard_bottle")
    assert ycb_object_exists(ycb_root, "banana")
    assert not ycb_object_exists(ycb_root, "apple")

    # Path resolution
    path1 = resolve_ycb_object_directory(ycb_root, "mustard_bottle")
    assert path1 == obj1_dir

    path2 = resolve_ycb_object_directory(ycb_root, "banana")
    assert path2 == obj2_dir

    # Mesh file lookup
    file2 = find_ycb_mesh_file(path2)
    assert file2 == mesh2


def test_simulation_error_handling(minimal_robot_xml, tmp_path):
    """Test all error-handling paths and parameter validations in simulation modules."""
    # 1. load_mujoco_model type validation
    with pytest.raises(TypeError):
        load_mujoco_model("not-a-path")  # type: ignore[arg-type]

    # 2. State-getter/setter type validations
    with pytest.raises(TypeError, match="state"):
        reset_simulation("invalid-state")
    with pytest.raises(TypeError, match="state"):
        read_joint_positions("invalid-state")
    with pytest.raises(TypeError, match="state"):
        set_joint_positions("invalid-state", np.zeros(1))
    with pytest.raises(TypeError, match="state"):
        read_body_pose("invalid-state", "base")

    # 3. set_joint_positions input validation
    model = load_mujoco_model(minimal_robot_xml)
    state, step, contacts = create_simulation(model)
    with pytest.raises(TypeError, match="positions"):
        set_joint_positions(state, "not-an-array")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="body_name"):
        read_body_pose(state, 123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Body 'nonexistent' not found"):
        read_body_pose(state, "nonexistent")

    # 4. step function parameter validations
    with pytest.raises(TypeError, match="dt"):
        step("not-a-number")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        step(0.0)
    with pytest.raises(ValueError, match="positive"):
        step(-0.002)
    with pytest.raises(ValueError, match="finite"):
        step(np.nan)

    # 5. build_scene_xml validations
    with pytest.raises(TypeError):
        build_scene_xml("not-a-path", Path("obj.xml"), None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_scene_xml(Path("robot.xml"), "not-a-path", None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_scene_xml(Path("robot.xml"), Path("obj.xml"), "not-a-path")  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        build_scene_xml(Path("non_existent_robot.xml"), minimal_robot_xml, None)
    with pytest.raises(FileNotFoundError):
        build_scene_xml(minimal_robot_xml, Path("non_existent_obj.xml"), None)
    with pytest.raises(FileNotFoundError):
        build_scene_xml(minimal_robot_xml, minimal_robot_xml, Path("non_existent_table.xml"))

    # 6. attach_object_to_scene validations
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

    # 7. step_scene validations
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

    # 8. collect_contacts validations
    with pytest.raises(TypeError, match="contacts"):
        collect_contacts("not-callable", {"base"})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="body_names"):
        collect_contacts(contacts, "not-a-set")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="strings"):
        collect_contacts(contacts, {123})  # type: ignore[arg-type]

    # 9. YCB validations
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

