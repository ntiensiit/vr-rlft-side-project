import os
import subprocess
import sys
import tempfile
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
    # A pair whose normals oppose each other but not enough to satisfy the
    # strict antipodal threshold still yields grasps in relaxed mode.
    rng = np.random.default_rng(42)
    points = np.array([[0, 0, 0], [0.01, 0, 0]], dtype=np.float32)
    # n_i at 30 deg to the approach axis, n_j at 40 deg on the opposite side:
    # dot(n_i, -n_j) = cos(70 deg) ~ 0.34 (fails strict 0.5, passes relaxed 0.0).
    theta_i, theta_j = np.deg2rad(30), np.deg2rad(40)
    normals = np.array(
        [
            [np.cos(theta_i), 0.0, np.sin(theta_i)],
            [-np.cos(theta_j), 0.0, np.sin(theta_j)],
        ],
        dtype=np.float32,
    )

    grasps = generate_analytical_grasps(
        points, normals, num_grasps=2, gripper_width=0.05, rng=rng,
        allow_relaxed=True,
    )
    # The relaxed fallback allows grasp construction for opposing normals.
    assert grasps.shape[0] > 0

    # Without the relaxed fallback the same cloud yields no grasps.
    strict = generate_analytical_grasps(
        points, normals, num_grasps=2, gripper_width=0.05, rng=rng,
    )
    assert strict.shape == (0, 4, 4)


def test_generate_analytical_grasps_strict_policy():
    # The data policy prevents saving unconstrained grasps: clouds without
    # opposing normals yield no grasps even in relaxed mode.
    rng = np.random.default_rng(42)
    points = np.array([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0]], dtype=np.float32)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)

    grasps = generate_analytical_grasps(
        points, normals, num_grasps=2, gripper_width=0.05, rng=rng,
        allow_relaxed=True,
    )
    assert grasps.shape == (0, 4, 4)


def test_generate_analytical_grasps_validation():
    rng = np.random.default_rng(42)
    points = np.random.randn(10, 3)
    normals = np.random.randn(10, 3)

    with pytest.raises(ValueError, match="relaxed_antipodal_dot"):
        generate_analytical_grasps(
            points, normals, 2, 0.05, rng, relaxed_antipodal_dot=1.5
        )
    with pytest.raises(TypeError, match="allow_relaxed"):
        generate_analytical_grasps(
            points, normals, 2, 0.05, rng, allow_relaxed="yes"  # type: ignore[arg-type]
        )


def test_prepare_data_synthetic_pipeline(tmp_path):
    ycb_root = tmp_path / "ycb_raw"
    ycb_root.mkdir()

    obj_name = "006_mustard_bottle"
    obj_dir = ycb_root / obj_name
    obj_dir.mkdir()

    # Write a dummy textured.obj
    mesh = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.1, depth=0.05)
    o3d.io.write_triangle_mesh(str(obj_dir / "textured.obj"), mesh)

    dataset_root = tmp_path / "dataset"
    output_index = tmp_path / "custom_index.json"

    cmd = [
        sys.executable,
        "scripts/prepare_data.py",
        "--mode", "synthetic",
        "--ycb-root", str(ycb_root),
        "--dataset-root", str(dataset_root),
        "--output-index", str(output_index),
        "--num-samples", "100",
        "--num-grasps", "5",
        "--seed", "42"
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")

    subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)

    # Verify .npy file exists
    npy_file = dataset_root / f"{obj_name}.npy"
    assert npy_file.is_file()

    # Verify index.json exists
    assert output_index.is_file()

    # Load and check the data structure using load_grasp_sample
    from grasping_ai.data.pointcloud_dataset import load_grasp_sample
    sample = load_grasp_sample(npy_file)
    assert "point_cloud" in sample
    assert "grasp_poses" in sample
    assert "scores" in sample
    assert "object_id" in sample
    assert sample["point_cloud"].shape == (100, 3)


def test_generate_synthetic_dataset_skips_zero_grasp_objects(monkeypatch):
    """Verify zero-grasp objects are skipped rather than saved unusable."""
    from scripts.prepare_data import generate_synthetic_dataset

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ycb_root = tmp_path / "ycb_raw"
        ycb_root.mkdir()

        # A cloud with parallel normals produces no strict or relaxed grasps.
        obj_dir = ycb_root / "006_mustard_bottle"
        obj_dir.mkdir()
        mesh = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.1, depth=0.05)
        o3d.io.write_triangle_mesh(str(obj_dir / "textured.obj"), mesh)

        output_dir = tmp_path / "dataset"
        output_dir.mkdir()

        # Monkeypatch the grasp generator to force zero grasps regardless of mesh.
        def always_empty(points, normals, num_grasps, gripper_width, rng, **kwargs):
            return np.empty((0, 4, 4), dtype=np.float32)

        monkeypatch.setattr(
            "scripts.prepare_data.generate_analytical_grasps",
            always_empty,
        )
        generate_synthetic_dataset(
            ycb_root=ycb_root,
            output_dir=output_dir,
            num_samples=100,
            num_grasps=5,
            gripper_width=0.08,
            seed=42,
        )

        npy_file = output_dir / "006_mustard_bottle.npy"
        assert not npy_file.exists()
