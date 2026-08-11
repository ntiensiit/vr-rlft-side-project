import numpy as np
import pytest

from grasping_ai.perception.geometry import (
    apply_transform,
    grasp_pose_to_transform,
    identity_transform,
    invert_transform,
    make_transform,
    rotation_matrix_from_axis_angle,
    rotation_matrix_to_axis_angle,
)


def test_se3_primitive_identity_behavior():
    """Verify that identity_transform returns identity matrix."""
    eye = identity_transform()
    assert eye.shape == (4, 4)
    assert np.allclose(eye, np.eye(4))


def test_se3_primitive_invalid_shape():
    """Verify type and shape errors raise appropriate exceptions."""
    with pytest.raises(ValueError, match="shape"):
        rotation_matrix_from_axis_angle(np.array([1, 0]), 0.0)
    with pytest.raises(TypeError, match="numpy array"):
        rotation_matrix_from_axis_angle("invalid", 0.0)
    with pytest.raises(ValueError, match="unit vector"):
        # non-unit vector
        rotation_matrix_from_axis_angle(np.array([1, 1, 1]), 0.0)

    # rotation_matrix_to_axis_angle
    with pytest.raises(ValueError, match="matrix"):
        rotation_matrix_to_axis_angle(np.eye(2))
    with pytest.raises(TypeError, match="numpy array"):
        rotation_matrix_to_axis_angle("invalid")

    # make_transform / grasp_pose_to_transform
    with pytest.raises(ValueError, match="shape"):
        make_transform(np.eye(3), np.zeros(2))
    with pytest.raises(TypeError, match="numpy array"):
        make_transform("invalid", np.zeros(3))

    # invert_transform
    with pytest.raises(ValueError, match="shape"):
        invert_transform(np.eye(3))
    with pytest.raises(TypeError, match="numpy array"):
        invert_transform("invalid")

    # apply_transform
    with pytest.raises(ValueError, match="shape"):
        apply_transform(np.zeros((10, 2)), np.eye(4))
    with pytest.raises(TypeError, match="numpy array"):
        apply_transform(np.zeros((10, 3)), "invalid")


def test_se3_primitive_non_finite_input():
    """Verify non-finite inputs behavior (NaN or inf)."""
    with pytest.raises(ValueError, match="unit vector"):
        rotation_matrix_from_axis_angle(np.array([np.nan, 0.0, 0.0]), 0.0)
    with pytest.raises(ValueError, match="unit vector"):
        rotation_matrix_from_axis_angle(np.array([np.inf, 0.0, 0.0]), 0.0)


def test_se3_primitive_no_global_side_effects():
    """Verify that calling functions twice with different inputs has no side effects."""
    axis1 = np.array([1.0, 0.0, 0.0])
    axis2 = np.array([0.0, 1.0, 0.0])
    r1 = rotation_matrix_from_axis_angle(axis1, 0.5)
    r2 = rotation_matrix_from_axis_angle(axis2, 1.0)
    r1_second = rotation_matrix_from_axis_angle(axis1, 0.5)
    assert np.allclose(r1, r1_second)
    assert not np.allclose(r1, r2)


def test_geometry_rotation_matrix_axis_angle():
    """Test rotation_matrix_from_axis_angle and rotation_matrix_to_axis_angle conversion."""
    axis = np.array([0.0, 0.0, 1.0])
    angle = np.pi / 2
    r = rotation_matrix_from_axis_angle(axis, angle)
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(r, expected)

    rec_axis, rec_angle = rotation_matrix_to_axis_angle(r)
    # Recaxis can be axis or opposite with negative angle
    if np.allclose(rec_axis, axis):
        assert np.allclose(rec_angle, angle)
    else:
        assert np.allclose(rec_axis, -axis)
        assert np.allclose(rec_angle, -angle)


def test_geometry_make_and_invert_transform():
    """Test make_transform, invert_transform and grasp_pose_to_transform."""
    rotation = np.eye(3)
    translation = np.array([1.0, 2.0, 3.0])
    t = make_transform(rotation, translation)
    assert t.shape == (4, 4)
    assert np.allclose(t[:3, 3], translation)
    assert np.allclose(t[:3, :3], rotation)
    assert np.allclose(t[3, :], [0, 0, 0, 1])

    t_grasp = grasp_pose_to_transform(rotation, translation)
    assert np.allclose(t, t_grasp)

    t_inv = invert_transform(t)
    identity = np.matmul(t, t_inv)
    assert np.allclose(identity, np.eye(4))


def test_geometry_apply_transform():
    """Test apply_transform on a set of points."""
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    translation = np.array([1.0, 2.0, 3.0])
    rotation = np.eye(3)
    t = make_transform(rotation, translation)

    transformed = apply_transform(points, t)
    expected = points + translation
    assert np.allclose(transformed, expected)
