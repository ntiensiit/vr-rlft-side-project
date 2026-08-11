from pathlib import Path

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]
import pytest

from grasping_ai.data.pointcloud_dataset import generate_analytical_grasps
from grasping_ai.sensors.pointcloud_sensor import sample_point_cloud_from_mesh


@pytest.fixture
def temp_mesh_path(tmp_path) -> Path:
    # Create a simple box mesh
    mesh = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.1, depth=0.1)
    mesh_file = tmp_path / "cube.obj"
    o3d.io.write_triangle_mesh(str(mesh_file), mesh)
    return mesh_file


def test_sample_point_cloud_from_mesh(temp_mesh_path):
    rng = np.random.default_rng(42)
    num_samples = 200

    # Basic output shape and type checks
    points = sample_point_cloud_from_mesh(temp_mesh_path, num_samples, rng)
    assert isinstance(points, np.ndarray)
    assert points.shape == (num_samples, 3)
    assert points.dtype == np.float32

    # Determinism check
    rng1 = np.random.default_rng(100)
    rng2 = np.random.default_rng(100)
    pts1 = sample_point_cloud_from_mesh(temp_mesh_path, num_samples, rng1)
    pts2 = sample_point_cloud_from_mesh(temp_mesh_path, num_samples, rng2)
    assert np.allclose(pts1, pts2)


def test_sample_point_cloud_invalid_inputs(temp_mesh_path):
    rng = np.random.default_rng(42)

    with pytest.raises(FileNotFoundError):
        sample_point_cloud_from_mesh(Path("non_existent_file.obj"), 100, rng)

    with pytest.raises(TypeError):
        sample_point_cloud_from_mesh("not_a_path_object", 100, rng)

    with pytest.raises(ValueError):
        sample_point_cloud_from_mesh(temp_mesh_path, -10, rng)


def test_generate_analytical_grasps():
    rng = np.random.default_rng(42)

    # Simple sphere-like point cloud
    theta = rng.uniform(0, 2 * np.pi, 500)
    phi = rng.uniform(0, np.pi, 500)
    x = 0.05 * np.sin(phi) * np.cos(theta)
    y = 0.05 * np.sin(phi) * np.sin(theta)
    z = 0.05 * np.cos(phi)

    points = np.stack([x, y, z], axis=1)
    # Normals point outwards from center
    normals = points / np.linalg.norm(points, axis=1, keepdims=True)

    num_grasps = 10
    gripper_width = 0.12

    grasps = generate_analytical_grasps(points, normals, num_grasps, gripper_width, rng)

    assert isinstance(grasps, np.ndarray)
    assert grasps.ndim == 3
    assert grasps.shape[1:] == (4, 4)
    assert grasps.shape[0] <= num_grasps

    if grasps.shape[0] > 0:
        for i in range(grasps.shape[0]):
            pose = grasps[i]
            # Rotation matrix orthogonality
            r_rot = pose[:3, :3]
            assert np.allclose(r_rot @ r_rot.T, np.eye(3), atol=1e-3)
            # Determinant check
            assert np.abs(np.linalg.det(r_rot) - 1.0) < 1e-3


def test_generate_analytical_grasps_fallback():
    # If normals point in same direction, normal antipodal constraint fails
    # Let's verify fallback operates when standard constraints yield 0 grasps.
    rng = np.random.default_rng(42)
    points = np.array([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0]], dtype=np.float32)
    # Normals all pointing in +Z direction
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)

    grasps = generate_analytical_grasps(points, normals, num_grasps=2, gripper_width=0.05, rng=rng)
    # Even with same-direction normals, fallback allows grasp construction
    assert grasps.shape[0] > 0
