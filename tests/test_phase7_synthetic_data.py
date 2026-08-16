"""Phase 7 synthetic data generation tests."""

from __future__ import annotations

import importlib
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

EXPECTED_GRASP_NDIM = 3
ROT_ORTHOGONAL_TOL = 1e-3
EXPECTED_NUM_GRASPS = 4


@pytest.fixture
def temp_mesh_path(tmp_path: Path) -> Path:
    """Fixture providing a temporary 3D mesh box file path using Open3D."""
    mesh = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.1, depth=0.1)
    mesh_file = tmp_path / "cube.obj"
    o3d.io.write_triangle_mesh(str(mesh_file), mesh)
    return mesh_file


def test_sample_point_cloud_from_mesh(temp_mesh_path: Path) -> None:
    """Verify that sampling a point cloud from a mesh generates the expected size, dtype, and is deterministic."""
    rng = np.random.default_rng(42)
    num_samples = 200

    points = sample_point_cloud_from_mesh(temp_mesh_path, num_samples, rng)
    if not (isinstance(points, np.ndarray)):
        raise TypeError
    if not (points.shape == (num_samples, 3)):
        raise AssertionError
    if not (points.dtype == np.float32):
        raise AssertionError

    rng1 = np.random.default_rng(100)
    rng2 = np.random.default_rng(100)
    pts1 = sample_point_cloud_from_mesh(temp_mesh_path, num_samples, rng1)
    pts2 = sample_point_cloud_from_mesh(temp_mesh_path, num_samples, rng2)
    if not (np.allclose(pts1, pts2)):
        raise AssertionError


def test_sample_point_cloud_invalid_inputs(temp_mesh_path: Path) -> None:
    """Verify that sampling from a mesh raises appropriate exceptions for non-existent files and invalid parameters."""
    rng = np.random.default_rng(42)

    with pytest.raises(FileNotFoundError):
        sample_point_cloud_from_mesh(Path("non_existent_file.obj"), 100, rng)

    with pytest.raises(TypeError):
        sample_point_cloud_from_mesh("not_a_path_object", 100, rng)

    with pytest.raises(ValueError, match="num_samples must be a positive integer"):
        sample_point_cloud_from_mesh(temp_mesh_path, -10, rng)


def test_generate_analytical_grasps() -> None:
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

    if not (isinstance(grasps, np.ndarray)):
        raise TypeError
    if not (grasps.ndim == EXPECTED_GRASP_NDIM):
        raise AssertionError
    if not (grasps.shape[1:] == (4, 4)):
        raise AssertionError
    if not (grasps.shape[0] <= num_grasps):
        raise AssertionError

    if grasps.shape[0] > 0:
        for i in range(grasps.shape[0]):
            pose = grasps[i]
            r_rot = pose[:3, :3]
            if not (np.allclose(r_rot @ r_rot.T, np.eye(3), atol=ROT_ORTHOGONAL_TOL)):
                raise AssertionError
            if not (np.abs(np.linalg.det(r_rot) - 1.0) < ROT_ORTHOGONAL_TOL):
                raise AssertionError


def test_generate_analytical_grasps_fallback() -> None:
    """Verify relaxed antipodal fallbacks trigger when strict filters yield no candidate poses."""
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
    if not (grasps.shape[0] > 0):
        raise AssertionError

    strict = generate_analytical_grasps(
        points,
        normals,
        num_grasps=2,
        gripper_width=0.05,
        rng=rng,
        allow_relaxed=False,
    )
    if not (strict.shape == (0, 4, 4)):
        raise AssertionError


def test_generate_analytical_grasps_strict_policy() -> None:
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
    if not (grasps.shape == (0, 4, 4)):
        raise AssertionError


def test_generate_analytical_grasps_validation() -> None:
    """Verify that generating analytical grasps validates parameter types and boundary ranges strictly."""
    rng = np.random.default_rng(42)
    points = rng.standard_normal((10, 3))
    normals = rng.standard_normal((10, 3))

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


def test_prepare_data_synthetic_pipeline(tmp_path: Path) -> None:
    """Verify that synthetic dataset generation and indexing produce grasp files correctly."""
    ycb_root = tmp_path / "ycb_raw"
    ycb_root.mkdir()

    obj_name = "006_mustard_bottle"
    obj_dir = ycb_root / obj_name
    obj_dir.mkdir()

    mesh = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.1, depth=0.05)
    o3d.io.write_triangle_mesh(str(obj_dir / "textured.obj"), mesh)

    dataset_root = tmp_path / "dataset"
    output_index = tmp_path / "custom_index.json"

    # Deferred: importing the pipeline pulls in optional heavy deps (mujoco).
    from grasping_ai.pipelines.prepare_synthetic_data import (  # noqa: PLC0415
        generate_synthetic_dataset,
        prepare_data_index,
    )

    generate_synthetic_dataset(
        ycb_root=ycb_root,
        output_dir=dataset_root,
        num_samples=100,
        num_grasps=5,
        gripper_width=0.08,
        seed=42,
    )
    prepare_data_index(dataset_root, output_index)

    npz_file = dataset_root / f"{obj_name}.npz"
    if not (npz_file.is_file()):
        raise AssertionError
    if not (output_index.is_file()):
        raise AssertionError

    sample = load_grasp_sample(npz_file)
    if "point_cloud" not in sample:
        raise AssertionError
    if "grasp_poses" not in sample:
        raise AssertionError
    if "scores" not in sample:
        raise AssertionError
    if "object_id" not in sample:
        raise AssertionError
    if not (sample["point_cloud"].shape == (100, 3)):
        raise AssertionError
    if not (isinstance(sample["scores"], np.ndarray)):
        raise TypeError
    if not (sample["scores"].shape[0] == sample["grasp_poses"].shape[0]):
        raise AssertionError


def test_generate_synthetic_dataset_skips_zero_grasp_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that objects with zero valid analytical grasp candidates are skipped during dataset generation."""
    # Deferred: importing the pipeline pulls in optional heavy deps (mujoco).
    from grasping_ai.pipelines.prepare_synthetic_data import (  # noqa: PLC0415
        generate_synthetic_dataset,
    )

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

        def always_empty(
            points: np.ndarray,
            normals: np.ndarray,
            num_grasps: int,
            gripper_width: float,
            rng: np.random.Generator,
            **kwargs: object,
        ) -> np.ndarray:
            # Stub matches generate_analytical_grasps; args may be keyword-passed.
            _ = (points, normals, num_grasps, gripper_width, rng, kwargs)
            return np.empty((0, 4, 4), dtype=np.float32)

        monkeypatch.setattr(
            "grasping_ai.pipelines.prepare_synthetic_data.generate_analytical_grasps",
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

        npz_file = output_dir / "006_mustard_bottle.npz"
        if npz_file.exists():
            raise AssertionError


def test_generate_synthetic_dataset_fail_fast_on_required_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that dataset generation fails fast with a RuntimeError if required objects fail to yield grasps."""
    # Deferred: importing the pipeline pulls in optional heavy deps (mujoco).
    from grasping_ai.pipelines.prepare_synthetic_data import (  # noqa: PLC0415
        generate_synthetic_dataset,
    )

    ycb_root = Path(tempfile.mkdtemp()) / "ycb"
    obj_name = "006_mustard_bottle"
    obj_dir = ycb_root / obj_name
    obj_dir.mkdir(parents=True)
    mesh = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.1, depth=0.05)
    o3d.io.write_triangle_mesh(str(obj_dir / "textured.obj"), mesh)

    output_dir = Path(tempfile.mkdtemp()) / "dataset"
    output_dir.mkdir()

    def always_empty(
        points: np.ndarray,
        normals: np.ndarray,
        num_grasps: int,
        gripper_width: float,
        rng: np.random.Generator,
        **kwargs: object,
    ) -> np.ndarray:
        # Stub matches generate_analytical_grasps; args may be keyword-passed.
        _ = (points, normals, num_grasps, gripper_width, rng, kwargs)
        return np.empty((0, 4, 4), dtype=np.float32)

    monkeypatch.setattr(
        "grasping_ai.pipelines.prepare_synthetic_data.generate_analytical_grasps",
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


def test_generate_synthetic_dataset_writes_quality_report(tmp_path: Path) -> None:
    """Verify that the synthetic dataset pipeline outputs a quality JSON report mapping objects to quality stats."""
    # Deferred: importing the pipeline pulls in optional heavy deps (mujoco).
    from grasping_ai.pipelines.prepare_synthetic_data import (  # noqa: PLC0415
        generate_synthetic_dataset,
    )

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
    if not (report_path.is_file()):
        raise AssertionError
    report_text = report_path.read_text(encoding="utf-8")
    if obj_name not in report_text:
        raise AssertionError


def test_generate_synthetic_dataset_sim_fallback_to_analytical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that simulate-validation fallbacks to raw analytical candidates when simulation outcomes fail."""
    # Deferred: importing the pipeline pulls in optional heavy deps (mujoco).
    from grasping_ai.pipelines.prepare_synthetic_data import (  # noqa: PLC0415
        generate_synthetic_dataset,
    )

    ycb_root = tmp_path / "ycb_raw"
    ycb_root.mkdir()
    obj_name = "006_mustard_bottle"
    obj_dir = ycb_root / obj_name
    obj_dir.mkdir()
    mesh = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.1, depth=0.05)
    o3d.io.write_triangle_mesh(str(obj_dir / "textured.obj"), mesh)

    output_dir = tmp_path / "dataset"

    def reject_all_sim(*args: object, **kwargs: object) -> dict[str, bool | float]:
        _ = (args, kwargs)
        return {
            "success": False,
            "fk_position_error": float("inf"),
            "contact_count": 0.0,
        }

    simulate_grasp_module = importlib.import_module("grasping_ai.pipelines.prepare_synthetic_data")
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
    if not ((output_dir / f"{obj_name}.npz").is_file()):
        raise AssertionError


