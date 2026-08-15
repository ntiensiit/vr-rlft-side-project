from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import open3d as o3d  # type: ignore[import-untyped]
import numpy as np
import pytest

from grasping_ai.data.pointcloud_dataset import (
    generate_analytical_grasps,
    load_grasp_sample,
)
from grasping_ai.sensors.pointcloud_sensor import sample_point_cloud_from_mesh


@pytest.fixture
def temp_mesh_path(tmp_path) -> Path:
    """Fixture providing a temporary 3D mesh box file path using Open3D."""
    mesh = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.1, depth=0.1)
    mesh_file = tmp_path / "cube.obj"
    o3d.io.write_triangle_mesh(str(mesh_file), mesh)
    return mesh_file


def test_sample_point_cloud_from_mesh(temp_mesh_path):
    """Verify that sampling a point cloud from a mesh generates the expected size, dtype, and is deterministic."""
    rng = np.random.default_rng(42)
    num_samples = 200

    points = sample_point_cloud_from_mesh(temp_mesh_path, num_samples, rng)
    assert isinstance(points, np.ndarray)
    assert points.shape == (num_samples, 3)
    assert points.dtype == np.float32

    rng1 = np.random.default_rng(100)
    rng2 = np.random.default_rng(100)
    pts1 = sample_point_cloud_from_mesh(temp_mesh_path, num_samples, rng1)
    pts2 = sample_point_cloud_from_mesh(temp_mesh_path, num_samples, rng2)
    assert np.allclose(pts1, pts2)


def test_sample_point_cloud_invalid_inputs(temp_mesh_path):
    """Verify that sampling from a mesh raises appropriate exceptions for non-existent files and invalid parameters."""
    rng = np.random.default_rng(42)

    with pytest.raises(FileNotFoundError):
        sample_point_cloud_from_mesh(Path("non_existent_file.obj"), 100, rng)

    with pytest.raises(TypeError):
        sample_point_cloud_from_mesh("not_a_path_object", 100, rng)

    with pytest.raises(ValueError):
        sample_point_cloud_from_mesh(temp_mesh_path, -10, rng)


def test_generate_analytical_grasps():
    """Verify that generating analytical grasps from simulated antipodal points yields valid orthogonal SE3 poses."""
    rng = np.random.default_rng(42)

    theta = rng.uniform(0, 2 * np.pi, 500)
    phi = rng.uniform(0, np.pi, 500)
    x = 0.05 * np.sin(phi) * np.cos(theta)
    y = 0.05 * np.sin(phi) * np.sin(theta)
    z = 0.05 * np.cos(phi)

    points = np.stack([x, y, z], axis=1)
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
            r_rot = pose[:3, :3]
            assert np.allclose(r_rot @ r_rot.T, np.eye(3), atol=1e-3)
            assert np.abs(np.linalg.det(r_rot) - 1.0) < 1e-3


def test_generate_analytical_grasps_fallback():
    """Verify that relaxed antipodal filters allow grasp generation fallbacks when strict filters yield no candidate poses."""
    rng = np.random.default_rng(42)
    points = np.array([[0, 0, 0], [0.01, 0, 0]], dtype=np.float32)
    theta_i, theta_j = np.deg2rad(30), np.deg2rad(40)
    normals = np.array(
        [
            [np.cos(theta_i), 0.0, np.sin(theta_i)],
            [-np.cos(theta_j), 0.0, np.sin(theta_j)],
        ],
        dtype=np.float32,
    )

    grasps = generate_analytical_grasps(
        points,
        normals,
        num_grasps=2,
        gripper_width=0.05,
        rng=rng,
        allow_relaxed=True,
    )
    assert grasps.shape[0] > 0

    strict = generate_analytical_grasps(
        points,
        normals,
        num_grasps=2,
        gripper_width=0.05,
        rng=rng,
    )
    assert strict.shape == (0, 4, 4)


def test_generate_analytical_grasps_strict_policy():
    """Verify that generating grasps under strict constraints yields empty results for flat parallel normal vectors."""
    rng = np.random.default_rng(42)
    points = np.array([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0]], dtype=np.float32)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)

    grasps = generate_analytical_grasps(
        points,
        normals,
        num_grasps=2,
        gripper_width=0.05,
        rng=rng,
        allow_relaxed=True,
    )
    assert grasps.shape == (0, 4, 4)


def test_generate_analytical_grasps_validation():
    """Verify that generating analytical grasps validates parameter types and boundary ranges strictly."""
    rng = np.random.default_rng(42)
    points = np.random.randn(10, 3)
    normals = np.random.randn(10, 3)

    with pytest.raises(ValueError, match="relaxed_antipodal_dot"):
        generate_analytical_grasps(points, normals, 2, 0.05, rng, relaxed_antipodal_dot=1.5)
    with pytest.raises(TypeError, match="allow_relaxed"):
        generate_analytical_grasps(
            points,
            normals,
            2,
            0.05,
            rng,
            allow_relaxed="yes",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="strict_antipodal_dot"):
        generate_analytical_grasps(points, normals, 2, 0.05, rng, strict_antipodal_dot=1.5)
    with pytest.raises(ValueError, match="search_multiplier"):
        generate_analytical_grasps(points, normals, 2, 0.05, rng, search_multiplier=0)


def test_prepare_data_synthetic_pipeline(tmp_path):
    """Verify that the prepare_data script in synthetic mode runs via subprocess and exports grasp files correctly."""
    ycb_root = tmp_path / "ycb_raw"
    ycb_root.mkdir()

    obj_name = "006_mustard_bottle"
    obj_dir = ycb_root / obj_name
    obj_dir.mkdir()

    mesh = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.1, depth=0.05)
    o3d.io.write_triangle_mesh(str(obj_dir / "textured.obj"), mesh)

    dataset_root = tmp_path / "dataset"
    output_index = tmp_path / "custom_index.json"

    cmd = [
        sys.executable,
        "scripts/prepare_data.py",
        "--mode",
        "synthetic",
        "--ycb-root",
        str(ycb_root),
        "--dataset-root",
        str(dataset_root),
        "--output-index",
        str(output_index),
        "--num-samples",
        "100",
        "--num-grasps",
        "5",
        "--seed",
        "42",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")

    subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)

    npy_file = dataset_root / f"{obj_name}.npy"
    assert npy_file.is_file()
    assert output_index.is_file()

    sample = load_grasp_sample(npy_file)
    assert "point_cloud" in sample
    assert "grasp_poses" in sample
    assert "scores" in sample
    assert "object_id" in sample
    assert sample["point_cloud"].shape == (100, 3)
    assert isinstance(sample["scores"], np.ndarray)
    assert sample["scores"].shape[0] == sample["grasp_poses"].shape[0]


def test_generate_synthetic_dataset_skips_zero_grasp_objects(monkeypatch):
    """Verify that object folders resulting in zero valid analytical grasp candidates are skipped during dataset generation."""
    from scripts.prepare_data import generate_synthetic_dataset

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ycb_root = tmp_path / "ycb_raw"
        ycb_root.mkdir()

        obj_dir = ycb_root / "006_mustard_bottle"
        obj_dir.mkdir()
        mesh = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.1, depth=0.05)
        o3d.io.write_triangle_mesh(str(obj_dir / "textured.obj"), mesh)

        output_dir = tmp_path / "dataset"
        output_dir.mkdir()

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


def test_generate_synthetic_dataset_fail_fast_on_required_objects(monkeypatch):
    """Verify that dataset generation fails fast with a RuntimeError if required objects fail to yield grasps."""
    from scripts.prepare_data import generate_synthetic_dataset

    ycb_root = Path(tempfile.mkdtemp()) / "ycb"
    obj_name = "006_mustard_bottle"
    obj_dir = ycb_root / obj_name
    obj_dir.mkdir(parents=True)
    mesh = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.1, depth=0.05)
    o3d.io.write_triangle_mesh(str(obj_dir / "textured.obj"), mesh)

    output_dir = Path(tempfile.mkdtemp()) / "dataset"
    output_dir.mkdir()

    def always_empty(points, normals, num_grasps, gripper_width, rng, **kwargs):
        return np.empty((0, 4, 4), dtype=np.float32)

    monkeypatch.setattr(
        "scripts.prepare_data.generate_analytical_grasps",
        always_empty,
    )
    with pytest.raises(RuntimeError, match="Required YCB objects"):
        generate_synthetic_dataset(
            ycb_root=ycb_root,
            output_dir=output_dir,
            num_samples=100,
            num_grasps=5,
            gripper_width=0.08,
            seed=42,
            required_objects=[obj_name],
        )


def test_generate_synthetic_dataset_writes_quality_report(tmp_path):
    """Verify that the synthetic dataset pipeline outputs a quality JSON report mapping objects to quality stats."""
    from scripts.prepare_data import generate_synthetic_dataset

    ycb_root = tmp_path / "ycb_raw"
    ycb_root.mkdir()
    obj_name = "006_mustard_bottle"
    obj_dir = ycb_root / obj_name
    obj_dir.mkdir()
    mesh = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.1, depth=0.05)
    o3d.io.write_triangle_mesh(str(obj_dir / "textured.obj"), mesh)

    output_dir = tmp_path / "dataset"
    report_path = tmp_path / "quality.json"
    generate_synthetic_dataset(
        ycb_root=ycb_root,
        output_dir=output_dir,
        num_samples=100,
        num_grasps=4,
        gripper_width=0.08,
        seed=42,
        quality_report_path=report_path,
    )
    assert report_path.is_file()
    report_text = report_path.read_text(encoding="utf-8")
    assert obj_name in report_text


def test_generate_synthetic_dataset_sim_fallback_to_analytical(tmp_path, monkeypatch):
    """Verify that simulate-validation fallbacks to raw analytical candidates when simulation outcomes fail."""
    from scripts.prepare_data import generate_synthetic_dataset

    ycb_root = tmp_path / "ycb_raw"
    ycb_root.mkdir()
    obj_name = "006_mustard_bottle"
    obj_dir = ycb_root / obj_name
    obj_dir.mkdir()
    mesh = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.1, depth=0.05)
    o3d.io.write_triangle_mesh(str(obj_dir / "textured.obj"), mesh)

    output_dir = tmp_path / "dataset"

    def reject_all_sim(*args, **kwargs):
        return {
            "success": False,
            "fk_position_error": float("inf"),
            "contact_count": 0.0,
        }

    import importlib

    simulate_grasp_module = importlib.import_module("grasping_ai.pipelines.simulate_grasp")
    monkeypatch.setattr(simulate_grasp_module, "simulate_grasp", reject_all_sim)
    generate_synthetic_dataset(
        ycb_root=ycb_root,
        output_dir=output_dir,
        num_samples=100,
        num_grasps=4,
        gripper_width=0.08,
        seed=42,
        sim_validate=True,
        mjcf_root=tmp_path / "mjcf",
        robot_xml=Path("deploy/robot.xml"),
        gripper_close_command=np.array([0.0]),
        sim_validate_fallback_analytical=True,
    )
    assert (output_dir / f"{obj_name}.npy").is_file()


def test_audit_synthetic_labels(tmp_path):
    """Verify that the audit_synthetic_labels tool runs on generated datasets and computes quality metrics."""
    from scripts.audit_synthetic_labels import audit_synthetic_labels
    from scripts.prepare_data import generate_synthetic_dataset

    ycb_root = tmp_path / "ycb_raw"
    ycb_root.mkdir()
    obj_name = "006_mustard_bottle"
    obj_dir = ycb_root / obj_name
    obj_dir.mkdir()
    mesh = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.1, depth=0.05)
    o3d.io.write_triangle_mesh(str(obj_dir / "textured.obj"), mesh)

    output_dir = tmp_path / "dataset"
    generate_synthetic_dataset(
        ycb_root=ycb_root,
        output_dir=output_dir,
        num_samples=100,
        num_grasps=4,
        gripper_width=0.08,
        seed=42,
    )

    report = audit_synthetic_labels(
        dataset_root=output_dir,
        friction_coefficient=0.5,
        collision_clearance=0.005,
    )
    assert len(report) == 1
    assert report[0]["object_id"] == obj_name
    assert report[0]["num_grasps"] == 4
    assert float(report[0]["contact_scored_rate"]) > 0.0
    assert float(report[0]["mean_recomputed_score"]) > 0.0
