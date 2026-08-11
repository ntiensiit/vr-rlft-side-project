import numpy as np
import pytest

from grasping_ai.evaluation.collision import generate_analytical_contacts


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
