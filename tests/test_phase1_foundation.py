from __future__ import annotations

import os

import hypothesis.strategies as st
import numpy as np
import pytest
import pytransform3d
import scipy  # type: ignore[import-untyped]
from hypothesis import HealthCheck, given, settings
from hypothesis.extra.numpy import arrays

import grasping_ai
from grasping_ai.perception.geometry import (
    apply_transform,
    grasp_pose_to_transform,
    identity_transform,
    invert_transform,
    make_transform,
    rotation_matrix_from_axis_angle,
    rotation_matrix_to_axis_angle,
)


def test_package_imports():
    """Verify that grasping_ai is importable."""
    assert grasping_ai.__name__ == "grasping_ai"


def test_math_dependencies_available():
    """Verify that core math libraries can be imported."""
    import theseus

    assert pytransform3d.__version__ is not None
    assert scipy.__version__ is not None
    assert np.__version__ is not None
    assert theseus.__version__ is not None


def test_base_config_file_exists():
    """Verify that configs/base.yaml exists."""
    config_path = os.path.join("configs", "base.yaml")
    assert os.path.isfile(config_path)


def test_base_config_contains_contract_keys():
    """Verify base.yaml contains the required contract keys by plain text."""
    config_path = os.path.join("configs", "base.yaml")
    with open(config_path, encoding="utf-8") as f:
        content = f.read()
    assert "seed:" in content or "random_seed:" in content
    assert "device:" in content
    assert "output_dir:" in content
    assert "paths:" in content


def test_pyproject_preserves_src_package_layout():
    """Verify pyproject.toml preserves src package layout."""
    with open("pyproject.toml", encoding="utf-8") as f:
        content = f.read()
    assert 'packages = ["src"]' in content or "packages = ['src']" in content or '"src"' in content


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


# --- Property-Based Testing (Hypothesis) ---


@st.composite
def unit_vectors(draw):
    """Generate a 3D unit vector."""
    vec = draw(
        arrays(
            np.float64,
            (3,),
            elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        )
    )
    norm = np.linalg.norm(vec)
    if norm < 1e-5:
        return np.array([1.0, 0.0, 0.0])
    return vec / norm


@st.composite
def rotation_matrices(draw):
    """Generate a valid 3x3 rotation matrix using random axis and angle."""
    axis = draw(unit_vectors())
    angle = draw(st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False))
    return rotation_matrix_from_axis_angle(axis, angle)


translations = arrays(
    np.float64,
    (3,),
    elements=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
)


@st.composite
def point_clouds(draw):
    """Generate a variable-length point cloud of shape (N, 3)."""
    n = draw(st.integers(min_value=1, max_value=100))
    return draw(
        arrays(
            np.float64,
            (n, 3),
            elements=st.floats(
                min_value=-1000.0,
                max_value=1000.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )


@given(
    axis=unit_vectors(),
    angle=st.floats(min_value=-2 * np.pi, max_value=2 * np.pi, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=1000, suppress_health_check=[HealthCheck.large_base_example])
def test_property_rotation_matrix_validity(axis, angle):
    """Property: rotation_matrix_from_axis_angle produces valid orthogonal matrix with det=1."""
    r = rotation_matrix_from_axis_angle(axis, angle)
    assert r.shape == (3, 3)
    # Check orthogonality: r * r.T = I
    assert np.allclose(np.matmul(r, r.T), np.eye(3), atol=1e-6)
    # Check determinant is ~ 1.0
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-6)


@given(
    axis=unit_vectors(),
    angle=st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=1000, suppress_health_check=[HealthCheck.large_base_example])
def test_property_rotation_matrix_axis_angle_roundtrip(axis, angle):
    """Property: converting matrix back to axis-angle reconstructs the same rotation matrix."""
    r = rotation_matrix_from_axis_angle(axis, angle)
    rec_axis, rec_angle = rotation_matrix_to_axis_angle(r)
    # Reconstruct matrix from recovered axis/angle and check equality with r
    r_rec = rotation_matrix_from_axis_angle(rec_axis, rec_angle)
    assert np.allclose(r, r_rec, atol=1e-6)


@given(rotation=rotation_matrices(), translation=translations)
@settings(max_examples=1000, suppress_health_check=[HealthCheck.large_base_example])
def test_property_make_and_invert_transform(rotation, translation):
    """Property: make_transform and invert_transform roundtrip and grasp pose consistency."""
    t = make_transform(rotation, translation)
    assert t.shape == (4, 4)
    assert np.allclose(t[:3, :3], rotation)
    assert np.allclose(t[:3, 3], translation)
    assert np.allclose(t[3, :], [0.0, 0.0, 0.0, 1.0])

    t_grasp = grasp_pose_to_transform(rotation, translation)
    assert np.allclose(t, t_grasp)

    t_inv = invert_transform(t)
    # Check roundtrip invert(invert(t)) = t
    assert np.allclose(invert_transform(t_inv), t, atol=1e-6)
    # Check matrix multiplication product is identity
    assert np.allclose(np.matmul(t, t_inv), np.eye(4), atol=1e-6)
    assert np.allclose(np.matmul(t_inv, t), np.eye(4), atol=1e-6)


@given(points=point_clouds(), rotation=rotation_matrices(), translation=translations)
@settings(max_examples=1000, suppress_health_check=[HealthCheck.large_base_example])
def test_property_apply_transform(points, rotation, translation):
    """Property: apply_transform correctly transforms points and invert_transform reverts them."""
    t = make_transform(rotation, translation)
    transformed = apply_transform(points, t)
    assert transformed.shape == points.shape

    # Explicit computation: P * R^T + t^T
    expected = np.matmul(points, rotation.T) + translation
    assert np.allclose(transformed, expected, atol=1e-6)

    # Inverted transform reverts the points
    t_inv = invert_transform(t)
    reverted = apply_transform(transformed, t_inv)
    assert np.allclose(reverted, points, atol=1e-6)


def test_geometry_additional_validation_branches() -> None:
    """Verify validation checks on invalid rotations, translations, axes, angles, and transform shapes."""
    # identity_transform
    assert np.allclose(identity_transform(), np.eye(4))

    # rotation_matrix_from_axis_angle
    with pytest.raises(TypeError, match="Axis must be a numpy array"):
        rotation_matrix_from_axis_angle("not_array", 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Axis must have shape"):
        rotation_matrix_from_axis_angle(np.array([1.0, 0.0]), 0.0)
    with pytest.raises(TypeError, match="Angle must be a float or integer"):
        rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]), "invalid_angle")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Axis must be a unit vector"):
        rotation_matrix_from_axis_angle(np.array([2.0, 0.0, 0.0]), 0.0)

    # rotation_matrix_to_axis_angle
    with pytest.raises(TypeError, match="Rotation must be a numpy array"):
        rotation_matrix_to_axis_angle("not_array")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Rotation must be a"):
        rotation_matrix_to_axis_angle(np.eye(4))

    # make_transform
    with pytest.raises(TypeError, match="Rotation must be a numpy array"):
        make_transform("not_array", np.zeros(3))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Translation must be a numpy array"):
        make_transform(np.eye(3), "invalid_translation")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Rotation must have shape"):
        make_transform(np.eye(4), np.zeros(3))
    with pytest.raises(ValueError, match="Translation must have shape"):
        make_transform(np.eye(3), np.zeros(4))

    # invert_transform
    with pytest.raises(TypeError, match="Transform must be a numpy array"):
        invert_transform("not_array")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Transform must have shape"):
        invert_transform(np.eye(3))

    # apply_transform
    with pytest.raises(TypeError, match="Points must be a numpy array"):
        apply_transform("invalid_points", np.eye(4))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Transform must be a numpy array"):
        apply_transform(np.zeros((5, 3)), "not_array")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Points must have shape"):
        apply_transform(np.zeros((5, 4)), np.eye(4))
    with pytest.raises(ValueError, match="Transform must have shape"):
        apply_transform(np.zeros((5, 3)), np.eye(3))

    # grasp_pose_to_transform
    assert np.allclose(grasp_pose_to_transform(np.eye(3), np.zeros(3)), np.eye(4))
