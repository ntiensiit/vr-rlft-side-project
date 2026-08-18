"""Phase 3 data and perception tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from grasping_ai.data.grasp_vector import (
    se3_to_vec,
    vec_to_se3,
)
from grasping_ai.data.pointcloud_dataset import (
    discover_dataset_files,
    generate_analytical_grasps,
    iterate_grasp_dataset,
    load_grasp_sample,
    resolve_ycb_object_id,
    save_grasp_sample,
)
from grasping_ai.data.training_pairs import SupervisedGraspDataset, validate_grasp_dataset
from grasping_ai.data.transforms import (
    compose_transforms,
    make_random_rotation_jitter,
    make_translation_jitter,
    save_grasp_dataset_index,
)
from grasping_ai.perception.pointcloud import (
    build_kdtree,
    estimate_point_cloud_normals,
    farthest_point_sampling,
    normalize_point_cloud,
    sample_point_cloud,
    voxel_downsample,
)
from grasping_ai.sensors.pointcloud_sensor import (
    acquire_point_cloud_stream,
    merge_point_clouds,
    sample_point_cloud_from_mesh,
)

EXPECTED_INDEX_SAMPLES = 2
FARTHEST_POINT_INDEX = 2
EXPECTED_DOWNSAMPLED_POINTS = 2
EXPECTED_CLOUD_COUNT = 3
EXPECTED_TRAINING_PAIRS = 2
EXPECTED_FILTERED_PAIRS = 2


def test_numpy_runtime_dependency_available() -> None:
    """Verify that numpy package is available and can be imported."""
    import numpy as np  # noqa: PLC0415  # deferred: test verifies the dependency import

    if not (np.__version__ is not None):
        raise AssertionError


def test_phase1_package_import_remains_stable() -> None:
    """Verify that the package remains importable."""
    import grasping_ai  # noqa: PLC0415  # deferred: test verifies the package import

    if not (grasping_ai.__name__ == "grasping_ai"):
        raise AssertionError


def test_data_config_file_exists() -> None:
    """Verify data configuration file existence."""
    path = Path("configs/data/default.yaml")
    if not (path.is_file()):
        raise AssertionError


def test_object_config_file_exists() -> None:
    """Verify object configuration file existence."""
    path = Path("configs/object/default.yaml")
    if not (path.is_file()):
        raise AssertionError


def test_prepare_data_creates_index_from_minimal_dataset(tmp_path: Path) -> None:
    """Test discovering files and writing a JSON index."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    # Create dummy record
    rng = np.random.default_rng()
    record_path = dataset_root / "record_0.npz"
    save_grasp_sample(
        record_path,
        {
            "point_cloud": rng.random((10, 3)).astype(np.float32),
            "grasp_poses": rng.random((2, 4, 4)).astype(np.float32),
            "scores": rng.random(2).astype(np.float32),
            "object_id": "bottle",
        },
    )

    records = discover_dataset_files(dataset_root)
    if not (len(records) == 1):
        raise AssertionError
    if not (records[0] == record_path):
        raise AssertionError

    # Save index
    entries = [{"path": str(record)} for record in records]
    save_grasp_dataset_index(tmp_path, entries)

    index_file = tmp_path / "index.json"
    if not (index_file.is_file()):
        raise AssertionError

    with index_file.open(encoding="utf-8") as f:
        loaded_entries = json.load(f)
    if not (len(loaded_entries) == 1):
        raise AssertionError
    if not (loaded_entries[0]["path"] == str(record_path)):
        raise AssertionError


def test_save_grasp_dataset_index_honors_custom_filename() -> None:
    """Verify that a custom index filename is honored instead of defaulting to index.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root = Path(tmpdir)
        entries = [{"path": "record_0.npz"}]
        save_grasp_dataset_index(dataset_root, entries, "custom_index.json")

        index_file = dataset_root / "custom_index.json"
        if not (index_file.is_file()):
            raise AssertionError
        if (dataset_root / "index.json").exists():
            raise AssertionError

        with index_file.open(encoding="utf-8") as f:
            loaded_entries = json.load(f)
        if not (loaded_entries == entries):
            raise AssertionError


def test_prepare_data_rejects_missing_dataset_root() -> None:
    """Verify that discover_dataset_files raises FileNotFoundError for missing paths."""
    with pytest.raises(FileNotFoundError):
        discover_dataset_files(Path("non_existent_dataset_root"))
    with pytest.raises(TypeError):
        discover_dataset_files("not-a-path-object")  # type: ignore[arg-type]


def test_prepare_data_rejects_empty_dataset(tmp_path: Path) -> None:
    """Verify discover_dataset_files raises ValueError for empty directories."""
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(ValueError, match="No dataset record files"):
        discover_dataset_files(empty_root)


def test_discover_dataset_files_accepts_single_npz_record(tmp_path: Path) -> None:
    """A single processed record is a valid training dataset selection."""
    record = tmp_path / "003_cracker_box.npz"
    save_grasp_sample(
        record,
        {
            "point_cloud": np.zeros((4, 3), dtype=np.float32),
            "grasp_poses": np.tile(np.eye(4, dtype=np.float32), (1, 1, 1)),
            "scores": np.ones(1, dtype=np.float32),
            "object_id": "003_cracker_box",
        },
    )

    assert discover_dataset_files(record) == [record]


def test_index_loader_reads_prepared_index(tmp_path: Path) -> None:
    """Verify loading dataset via index iterate functions."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    # Create a couple dummy records
    rng = np.random.default_rng(42)
    record1 = {"point_cloud": rng.random((10, 3)), "object_id": "obj1"}
    record2 = {"point_cloud": rng.random((15, 3)), "object_id": "obj2"}

    path1 = dataset_root / "rec1.npz"
    path2 = dataset_root / "rec2.npz"
    save_grasp_sample(path1, record1)
    save_grasp_sample(path2, record2)

    samples = list(iterate_grasp_dataset(dataset_root))
    if not (len(samples) == EXPECTED_INDEX_SAMPLES):
        raise AssertionError
    if not (samples[0]["object_id"] == "obj1"):
        raise AssertionError
    if not (samples[1]["object_id"] == "obj2"):
        raise AssertionError
    if not (samples[0]["point_cloud"].shape == (10, 3)):
        raise AssertionError
    if not (samples[1]["point_cloud"].shape == (15, 3)):
        raise AssertionError


def test_index_loader_rejects_invalid_structure(tmp_path: Path) -> None:
    """Test loading invalid index inputs."""
    _ = tmp_path
    with pytest.raises(TypeError):
        list(iterate_grasp_dataset("not-a-path"))  # type: ignore[arg-type]


def test_save_grasp_sample_roundtrip(tmp_path: Path) -> None:
    """Verify save_grasp_sample writes pickle-free archives readable by load_grasp_sample."""
    path = tmp_path / "record.npz"
    rng = np.random.default_rng()
    sample = {
        "point_cloud": rng.standard_normal((12, 3)).astype(np.float32),
        "grasp_poses": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        "scores": np.array([0.1, 0.9], dtype=np.float32),
        "object_id": "mustard_bottle",
        "validation_version": "physical-lift-v4",
        "contact_sustained": np.array([True, False]),
        "initial_robot_object_collision_free": np.array([True, True]),
    }
    save_grasp_sample(path, sample)
    loaded = load_grasp_sample(path)
    if not (np.allclose(loaded["point_cloud"], sample["point_cloud"])):
        raise AssertionError
    if not (np.allclose(loaded["grasp_poses"], sample["grasp_poses"])):
        raise AssertionError
    if not (np.allclose(loaded["scores"], sample["scores"])):
        raise AssertionError
    if not (loaded["object_id"] == sample["object_id"]):
        raise AssertionError
    if loaded["validation_version"] != sample["validation_version"]:
        raise AssertionError
    if not np.array_equal(loaded["contact_sustained"], sample["contact_sustained"]):
        raise AssertionError
    if not np.array_equal(
        loaded["initial_robot_object_collision_free"],
        sample["initial_robot_object_collision_free"],
    ):
        raise AssertionError

    with np.load(path) as archive:
        if "point_cloud" not in archive:
            raise AssertionError
        if not (archive["point_cloud"].dtype == np.float32):
            raise AssertionError


def test_save_grasp_sample_validates_point_cloud(tmp_path: Path) -> None:
    """Verify save_grasp_sample rejects invalid point-cloud payloads before writing."""
    path = tmp_path / "invalid.npz"
    with pytest.raises(TypeError, match="'point_cloud' must be a numpy array"):
        save_grasp_sample(path, {"point_cloud": [[0.0, 0.0, 0.0]]})  # type: ignore[typeddict-item]

    with pytest.raises(ValueError, match="point_cloud must have shape"):
        save_grasp_sample(path, {"point_cloud": np.zeros((4, 2), dtype=np.float32)})


def test_load_grasp_sample_decodes_object_id_array(tmp_path: Path) -> None:
    """Verify object identifiers stored as unicode arrays decode to plain strings."""
    path = tmp_path / "object_id_array.npz"
    np.savez(
        path,
        point_cloud=np.zeros((3, 3), dtype=np.float32),
        object_id=np.array(["004_sugar_box"], dtype=np.str_),
    )
    sample = load_grasp_sample(path)
    if not (sample["object_id"] == "004_sugar_box"):
        raise AssertionError


def test_point_cloud_loader_reads_valid_npz_point_cloud(tmp_path: Path) -> None:
    """Test load_grasp_sample on valid file."""
    path = tmp_path / "sample.npz"
    save_grasp_sample(
        path,
        {
            "point_cloud": np.ones((5, 3), dtype=np.float32),
            "object_id": "box",
        },
    )

    sample = load_grasp_sample(path)
    if not (np.allclose(sample["point_cloud"], 1.0)):
        raise AssertionError
    if not (sample["object_id"] == "box"):
        raise AssertionError


def test_point_cloud_loader_rejects_wrong_shape(tmp_path: Path) -> None:
    """Verify loader validation on malformed shapes."""
    path = tmp_path / "bad.npz"

    # Missing point_cloud key
    np.savez(path, other=np.array([123], dtype=np.int64))
    with pytest.raises(ValueError, match="missing 'point_cloud'"):
        load_grasp_sample(path)

    # Wrong shape
    np.savez(path, point_cloud=np.zeros((10, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        load_grasp_sample(path)


def test_point_cloud_loader_rejects_non_finite_values(tmp_path: Path) -> None:
    """Verify loader validation rejects NaN and Inf."""
    path = tmp_path / "nan.npz"
    np.savez(path, point_cloud=np.array([[1.0, 2.0, np.nan]], dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        load_grasp_sample(path)


def test_resolve_ycb_object_id(tmp_path: Path) -> None:
    """Verify YCB mesh path lookup resolution."""
    ycb_root = tmp_path / "ycb"
    ycb_root.mkdir()

    # Create dummy folders
    obj_dir = ycb_root / "004_sugar_box"
    obj_dir.mkdir()
    mesh_file = obj_dir / "textured.obj"
    mesh_file.write_text("mesh content", encoding="utf-8")

    resolved = resolve_ycb_object_id(ycb_root, "sugar_box")
    if not (resolved == mesh_file):
        raise AssertionError

    # Non-existent object
    with pytest.raises(FileNotFoundError):
        resolve_ycb_object_id(ycb_root, "banana")


def test_make_random_rotation_jitter() -> None:
    """Verify SO(3) random rotation transform."""
    rng = np.random.default_rng(42)
    transform = make_random_rotation_jitter(rng)

    points = rng.random((5, 3))
    grasp_poses = rng.random((2, 4, 4))
    # Homogenize grasp poses
    grasp_poses[:, 3, :] = [0, 0, 0, 1]
    scores = np.array([0.9, 0.8])

    pts_rot, gps_rot, scs_rot = transform(points, grasp_poses, scores)

    if not (pts_rot.shape == (5, 3)):
        raise AssertionError
    if not (gps_rot is not None):
        raise AssertionError
    if not (gps_rot.shape == (2, 4, 4)):
        raise AssertionError
    if not (np.allclose(scs_rot, scores)):
        raise AssertionError

    # Verify that the relative distances between points are preserved (isometric transform)
    dist_orig = np.linalg.norm(points[0] - points[1])
    dist_rot = np.linalg.norm(pts_rot[0] - pts_rot[1])
    if not (np.isclose(dist_orig, dist_rot)):
        raise AssertionError


def test_make_translation_jitter() -> None:
    """Verify translation transform."""
    rng = np.random.default_rng(42)
    transform = make_translation_jitter(rng, scale=0.1)

    points = np.zeros((5, 3))
    grasp_poses = np.zeros((2, 4, 4))
    grasp_poses[:, 3, 3] = 1.0
    scores = np.array([0.5, 0.6])

    pts_t, gps_t, _ = transform(points, grasp_poses, scores)

    # Points should all have the same translation shift
    shift = pts_t[0]
    if not (np.allclose(pts_t, shift)):
        raise AssertionError
    if not (gps_t is not None):
        raise AssertionError
    if not (np.allclose(gps_t[:, :3, 3], shift)):
        raise AssertionError


def test_compose_transforms() -> None:
    """Verify sequential composition of transforms."""
    rng = np.random.default_rng(42)
    t1 = make_translation_jitter(rng, scale=0.01)
    t2 = make_translation_jitter(rng, scale=0.02)

    composed = compose_transforms(t1, t2)

    points = np.zeros((3, 3))
    pts_out, _, _ = composed(points, None, None)

    # Output should not be zero
    if np.allclose(pts_out, 0.0):
        raise AssertionError


def test_sample_point_cloud() -> None:
    """Verify point cloud fixed size sampling."""
    rng = np.random.default_rng(42)
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Downsample/upsample
    sampled_1 = sample_point_cloud(points, 1, rng)
    if not (sampled_1.shape == (1, 3)):
        raise AssertionError

    sampled_3 = sample_point_cloud(points, 3, rng)
    if not (sampled_3.shape == (3, 3)):
        raise AssertionError


def test_normalize_point_cloud() -> None:
    """Verify centering and unit scaling."""
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    normalized = normalize_point_cloud(points)

    # Centroid should be at origin
    if not (np.allclose(np.mean(normalized, axis=0), 0.0)):
        raise AssertionError
    # Max distance should be 1.0
    if not (np.allclose(np.max(np.linalg.norm(normalized, axis=1)), 1.0)):
        raise AssertionError


def test_farthest_point_sampling() -> None:
    """Verify Farthest Point Sampling indices selection."""
    rng = np.random.default_rng(42)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    indices = farthest_point_sampling(points, 2, rng)
    if not (indices.shape == (2,)):
        raise AssertionError
    # Farthest points should be selected (index 0 and 2, or index 1 and 2 depending on random start)
    if FARTHEST_POINT_INDEX not in indices:
        raise AssertionError


def test_estimate_point_cloud_normals() -> None:
    """Verify normal estimation using local PCA."""
    # Create flat point cloud in XY plane
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
    )
    normals = estimate_point_cloud_normals(points, neighborhood_size=4)
    # Normals should be perpendicular to XY plane (i.e. parallel to Z axis)
    for n in normals:
        if not (np.isclose(np.abs(n[2]), 1.0, atol=1e-4)):
            raise AssertionError
        if not (np.isclose(n[0], 0.0, atol=1e-4)):
            raise AssertionError
        if not (np.isclose(n[1], 0.0, atol=1e-4)):
            raise AssertionError


def test_voxel_downsample() -> None:
    """Verify voxel downsampling centroids grouping."""
    points = np.array(
        [
            [0.01, 0.01, 0.01],
            [0.02, 0.02, 0.02],
            [0.9, 0.9, 0.9],
        ],
    )
    downsampled = voxel_downsample(points, voxel_size=0.1)
    # The first two points should fall in the same voxel and be averaged
    if not (len(downsampled) == EXPECTED_DOWNSAMPLED_POINTS):
        raise AssertionError
    if not (np.allclose(downsampled[0], [0.015, 0.015, 0.015]) or np.allclose(downsampled[1], [0.015, 0.015, 0.015])):
        raise AssertionError


def test_build_kdtree() -> None:
    """Verify KDTree construction and neighbor querying."""
    points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    kdtree = build_kdtree(points)

    # Query nearest to [1.1, 0, 0]
    dist, idx = kdtree.query(np.array([1.1, 0.0, 0.0]))
    if not (idx == 0):
        raise AssertionError
    if not (np.isclose(dist, 0.1)):
        raise AssertionError


def test_data_functions_do_not_leak_global_state(tmp_path: Path) -> None:
    """Verify voxel downsampling is deterministic across calls and does not leak global state."""
    _ = tmp_path
    # Test voxel downsampling determinism
    rng = np.random.default_rng()
    points = rng.random((10, 3))
    out1 = voxel_downsample(points, 0.2)
    out2 = voxel_downsample(points, 0.2)
    if not (np.allclose(out1, out2)):
        raise AssertionError


def _assert_dataset_error_paths(tmp_path: Path) -> None:
    # --- pointcloud_dataset.py validations ---
    # discover_dataset_files validations
    with pytest.raises(TypeError, match="dataset_root"):
        discover_dataset_files("not-a-path")  # type: ignore[arg-type]
    file_path = tmp_path / "file.txt"
    file_path.write_text("not a dir", encoding="utf-8")
    with pytest.raises(ValueError, match="must use the .npz extension"):
        discover_dataset_files(file_path)

    # load_grasp_sample validations
    with pytest.raises(TypeError, match="record_path"):
        load_grasp_sample("not-a-path")  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError, match="not found"):
        load_grasp_sample(tmp_path / "non_existent.npz")
    with pytest.raises(ValueError, match="not a file"):
        load_grasp_sample(tmp_path)

    # Invalid np load (corrupted file)
    bad_npz = tmp_path / "corrupted.npz"
    bad_npz.write_bytes(b"PK\x03\x04this is not a valid npz archive")
    with pytest.raises(ValueError, match="Failed to load"):
        load_grasp_sample(bad_npz)

    # Legacy pickled record extension is rejected
    legacy_npy = tmp_path / "legacy.npy"
    np.save(legacy_npy, {"point_cloud": np.zeros((2, 3))}, allow_pickle=True)
    with pytest.raises(ValueError, match=r"must use the \.npz extension"):
        load_grasp_sample(legacy_npy)

    # Plain numeric .npy arrays are not dataset records
    array_npy = tmp_path / "array.npy"
    np.save(array_npy, np.zeros((2, 3)))
    with pytest.raises(ValueError, match=r"must use the \.npz extension"):
        load_grasp_sample(array_npy)

    with pytest.raises(ValueError, match=r"must use the \.npz extension"):
        save_grasp_sample(tmp_path / "invalid_suffix.npy", {"point_cloud": np.zeros((2, 3))})

    # resolve_ycb_object_id validations
    with pytest.raises(TypeError, match="ycb_root"):
        resolve_ycb_object_id("not-a-path", "sugar_box")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="object_name"):
        resolve_ycb_object_id(tmp_path, 123)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_ycb_object_id(tmp_path / "non_existent_ycb", "sugar_box")
    with pytest.raises(ValueError, match="not a directory"):
        resolve_ycb_object_id(file_path, "sugar_box")


def _assert_transform_error_paths(rng: np.random.Generator, tmp_path: Path) -> None:
    # --- transforms.py validations ---
    # make_random_rotation_jitter validations
    with pytest.raises(TypeError, match="rng"):
        make_random_rotation_jitter("not-a-generator")  # type: ignore[arg-type]
    rot_transform = make_random_rotation_jitter(rng)
    with pytest.raises(TypeError, match="points"):
        rot_transform("not-array", None, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="points shape"):
        rot_transform(np.zeros(3), None, None)
    with pytest.raises(ValueError, match="finite"):
        rot_transform(np.array([[np.nan, 2.0, 3.0]]), None, None)

    # grasp_poses validation in rotation
    with pytest.raises(TypeError, match="grasp_poses"):
        rot_transform(np.zeros((2, 3)), "not-array", None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="grasp_poses"):
        rot_transform(np.zeros((2, 3)), np.zeros((2, 3)), None)
    with pytest.raises(ValueError, match="finite"):
        rot_transform(np.zeros((2, 3)), np.array([[[np.nan] * 4] * 4]), None)

    # scores validation in rotation
    with pytest.raises(TypeError, match="scores"):
        rot_transform(np.zeros((2, 3)), None, "not-array")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scores"):
        rot_transform(np.zeros((2, 3)), None, np.array([np.nan]))

    # make_translation_jitter validations
    with pytest.raises(TypeError, match="rng"):
        make_translation_jitter("not-a-generator", 0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scale"):
        make_translation_jitter(rng, -0.1)
    with pytest.raises(ValueError, match="scale"):
        make_translation_jitter(rng, np.nan)
    trans_transform = make_translation_jitter(rng, 0.1)

    # grasp_poses validation in translation
    with pytest.raises(TypeError, match="grasp_poses"):
        trans_transform(np.zeros((2, 3)), "not-array", None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="grasp_poses"):
        trans_transform(np.zeros((2, 3)), np.zeros((2, 3)), None)
    with pytest.raises(ValueError, match="finite"):
        trans_transform(np.zeros((2, 3)), np.array([[[np.nan] * 4] * 4]), None)

    # scores validation in translation
    with pytest.raises(TypeError, match="scores"):
        trans_transform(np.zeros((2, 3)), None, "not-array")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scores"):
        trans_transform(np.zeros((2, 3)), None, np.array([np.nan]))

    # compose_transforms validation
    with pytest.raises(TypeError, match="callable"):
        compose_transforms("not-callable")(np.zeros((2, 3)), None, None)  # type: ignore[arg-type]

    # save_grasp_dataset_index validations
    with pytest.raises(TypeError, match="dataset_root"):
        save_grasp_dataset_index("not-a-path", [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="entries"):
        save_grasp_dataset_index(tmp_path, "not-a-list")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="dictionaries"):
        save_grasp_dataset_index(tmp_path, ["not-a-dict"])  # type: ignore[arg-type]


def _assert_sampling_error_paths(rng: np.random.Generator) -> None:
    # --- pointcloud.py validations ---
    # sample_point_cloud validations
    with pytest.raises(TypeError, match="points"):
        sample_point_cloud("not-array", 5, rng)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="points shape"):
        sample_point_cloud(np.zeros((2, 2)), 5, rng)
    with pytest.raises(ValueError, match="not be empty"):
        sample_point_cloud(np.zeros((0, 3)), 5, rng)
    with pytest.raises(ValueError, match="positive integer"):
        sample_point_cloud(np.zeros((2, 3)), 0, rng)
    with pytest.raises(TypeError, match="rng"):
        sample_point_cloud(np.zeros((2, 3)), 5, "not-generator")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        sample_point_cloud(np.array([[np.nan, 2.0, 3.0]]), 1, rng)

    # normalize_point_cloud validations
    with pytest.raises(TypeError, match="points"):
        normalize_point_cloud("not-array")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="points shape"):
        normalize_point_cloud(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="not be empty"):
        normalize_point_cloud(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="finite"):
        normalize_point_cloud(np.array([[np.nan, 2.0, 3.0]]))
    # Test zero distance normalization (single point)
    zero_dist_pc = np.array([[1.0, 2.0, 3.0]])
    if not (np.allclose(normalize_point_cloud(zero_dist_pc), 0.0)):
        raise AssertionError

    # farthest_point_sampling validations
    with pytest.raises(TypeError, match="points"):
        farthest_point_sampling("not-array", 5, rng)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="points shape"):
        farthest_point_sampling(np.zeros((2, 2)), 5, rng)
    with pytest.raises(ValueError, match="not be empty"):
        farthest_point_sampling(np.zeros((0, 3)), 5, rng)
    with pytest.raises(ValueError, match="positive integer"):
        farthest_point_sampling(np.zeros((2, 3)), 0, rng)
    with pytest.raises(TypeError, match="rng"):
        farthest_point_sampling(np.zeros((2, 3)), 5, "not-generator")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        farthest_point_sampling(np.array([[np.nan, 2.0, 3.0]]), 1, rng)


def _assert_normals_and_downsample_error_paths() -> None:
    # estimate_point_cloud_normals validations
    with pytest.raises(TypeError, match="points"):
        estimate_point_cloud_normals("not-array", 4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="points shape"):
        estimate_point_cloud_normals(np.zeros((2, 2)), 4)
    with pytest.raises(ValueError, match="not be empty"):
        estimate_point_cloud_normals(np.zeros((0, 3)), 4)
    with pytest.raises(ValueError, match="positive integer"):
        estimate_point_cloud_normals(np.zeros((2, 3)), 0)
    with pytest.raises(ValueError, match="finite"):
        estimate_point_cloud_normals(np.array([[np.nan, 2.0, 3.0]]), 4)
    # Estimate with neighborhood < 3 (returns default normal [0, 0, 1])
    small_normals = estimate_point_cloud_normals(np.zeros((2, 3)), neighborhood_size=2)
    if not (np.allclose(small_normals, [0, 0, 1])):
        raise AssertionError

    # voxel_downsample validations
    with pytest.raises(TypeError, match="points"):
        voxel_downsample("not-array", 0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="points shape"):
        voxel_downsample(np.zeros((2, 2)), 0.1)
    with pytest.raises(ValueError, match="positive float"):
        voxel_downsample(np.zeros((2, 3)), 0.0)
    with pytest.raises(ValueError, match="finite"):
        voxel_downsample(np.array([[np.nan, 2.0, 3.0]]), 0.1)

    # build_kdtree validations
    with pytest.raises(TypeError, match="points"):
        build_kdtree("not-array")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="points shape"):
        build_kdtree(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        build_kdtree(np.array([[np.nan, 2.0, 3.0]]))


def _assert_grasp_generation_error_paths(rng: np.random.Generator, tmp_path: Path) -> None:
    # generate_analytical_grasps validation check (strict_alignment_dot bounds)
    with pytest.raises(ValueError, match="strict_alignment_dot must be in"):
        generate_analytical_grasps(
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            num_grasps=1,
            gripper_width=0.05,
            strict_alignment_dot=2.0,
            rng=rng,
        )

    # Dataset and validation behavior for an empty dataset

    empty_dir = tmp_path / "empty_dataset"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No dataset record files"):
        SupervisedGraspDataset(empty_dir)
    with pytest.raises(ValueError, match="No dataset record files"):
        validate_grasp_dataset(empty_dir)


def test_data_perception_error_handling(tmp_path: Path) -> None:
    """Test all input validation and error handling paths in Phase 3 modules."""
    rng = np.random.default_rng(42)
    _assert_dataset_error_paths(tmp_path)
    _assert_transform_error_paths(rng, tmp_path)
    _assert_sampling_error_paths(rng)
    _assert_normals_and_downsample_error_paths()
    _assert_grasp_generation_error_paths(rng, tmp_path)


def test_resolve_ycb_object_id_exact_and_fallbacks(tmp_path: Path) -> None:
    """Test resolve_ycb_object_id for exact directory matches and different mesh formats."""
    ycb_root = tmp_path / "ycb"
    ycb_root.mkdir()

    # 1. Exact directory match & textured.obj exists
    obj_dir1 = ycb_root / "mustard_bottle"
    obj_dir1.mkdir()
    mesh1 = obj_dir1 / "textured.obj"
    mesh1.write_text("mesh mustard", encoding="utf-8")

    if not (resolve_ycb_object_id(ycb_root, "mustard_bottle") == mesh1):
        raise AssertionError

    # 2. Substring directory match & .ply fallback
    obj_dir2 = ycb_root / "003_cracker_box"
    obj_dir2.mkdir()
    mesh2 = obj_dir2 / "cracker_box.ply"
    mesh2.write_text("mesh cracker", encoding="utf-8")

    if not (resolve_ycb_object_id(ycb_root, "cracker_box") == mesh2):
        raise AssertionError

    # 3. Substring directory match & .obj fallback (no textured.obj or ply)
    obj_dir3 = ycb_root / "011_banana"
    obj_dir3.mkdir()
    mesh3 = obj_dir3 / "banana.obj"
    mesh3.write_text("mesh banana", encoding="utf-8")

    if not (resolve_ycb_object_id(ycb_root, "banana") == mesh3):
        raise AssertionError

    # 4. Directory matches but contains no mesh files (returns directory Path)
    obj_dir4 = ycb_root / "025_mug"
    obj_dir4.mkdir()

    if not (resolve_ycb_object_id(ycb_root, "mug") == obj_dir4):
        raise AssertionError


def test_transforms_optional_none_inputs() -> None:
    """Verify transforms when grasp_poses and scores are None."""
    rng = np.random.default_rng(42)
    rot = make_random_rotation_jitter(rng)
    trans = make_translation_jitter(rng, scale=0.1)

    points = rng.random((5, 3))

    p_rot, gp_rot, s_rot = rot(points, None, None)
    if not (p_rot.shape == (5, 3)):
        raise AssertionError
    if gp_rot is not None:
        raise AssertionError
    if s_rot is not None:
        raise AssertionError

    p_trans, gp_trans, s_trans = trans(points, None, None)
    if not (p_trans.shape == (5, 3)):
        raise AssertionError
    if gp_trans is not None:
        raise AssertionError
    if s_trans is not None:
        raise AssertionError


def test_perception_edge_cases() -> None:
    """Verify FPS and normals estimation edge cases."""
    rng = np.random.default_rng(42)

    # FPS with num_samples = 1
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    idx_1 = farthest_point_sampling(pts, 1, rng)
    if not (idx_1.shape == (1,)):
        raise AssertionError

    # FPS with num_samples > n
    idx_5 = farthest_point_sampling(pts, 5, rng)
    if not (idx_5.shape == (5,)):
        raise AssertionError

    # Normals with zero covariance (identical points)
    identical_pts = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
    )
    normals = estimate_point_cloud_normals(identical_pts, neighborhood_size=4)
    if not (np.allclose(normals, [0, 0, 1])):
        raise AssertionError


def test_acquire_point_cloud_stream_yields_saved_clouds() -> None:
    """Verify that acquire_point_cloud_stream correctly yields point clouds loaded from numpy array files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        paths = []
        rng = np.random.default_rng()
        for i in range(3):
            path = root / f"obs_{i}.npy"
            np.save(path, rng.standard_normal((10 + i, 3)).astype(np.float32))
            paths.append(path)
        clouds = list(acquire_point_cloud_stream(paths))
        if not (len(clouds) == EXPECTED_CLOUD_COUNT):
            raise AssertionError
        if not (clouds[0].shape == (10, 3)):
            raise AssertionError
        if not (clouds[2].shape == (12, 3)):
            raise AssertionError


def test_acquire_point_cloud_stream_rejects_invalid_input() -> None:
    """Verify acquire_point_cloud_stream raises TypeError for invalid paths and FileNotFoundError for missing files."""
    with pytest.raises(TypeError, match=r"list of pathlib\.Path"):
        list(acquire_point_cloud_stream(["obs.npy"]))  # type: ignore[list-item]

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        missing = root / "missing.npy"
        with pytest.raises(FileNotFoundError):
            list(acquire_point_cloud_stream([missing]))


def test_merge_point_clouds_concatenates() -> None:
    """Verify that merge_point_clouds correctly concatenates multiple point cloud arrays into a single array."""
    rng = np.random.default_rng()
    a = rng.standard_normal((5, 3)).astype(np.float32)
    b = rng.standard_normal((7, 3)).astype(np.float32)
    merged = merge_point_clouds([a, b])
    if not (merged.shape == (12, 3)):
        raise AssertionError
    if not (np.allclose(merged[:5], a)):
        raise AssertionError
    if not (np.allclose(merged[5:], b)):
        raise AssertionError


def test_merge_point_clouds_empty_list() -> None:
    """Verify that merge_point_clouds returns an empty array of correct shape when given an empty list."""
    merged = merge_point_clouds([])
    if not (merged.shape == (0, 3)):
        raise AssertionError


def test_merge_point_clouds_validation() -> None:
    """Verify that merge_point_clouds raises TypeError or ValueError for invalid list or array shapes."""
    with pytest.raises(TypeError, match="list of numpy arrays"):
        merge_point_clouds("not_a_list")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape \\(N, 3\\)"):
        merge_point_clouds([np.random.default_rng().standard_normal((4, 2))])


def test_training_pairs_validations_and_augmentation(tmp_path: Path) -> None:
    """Verify validations on dataset root paths and dataset data augmentation inside training pair utilities."""
    with pytest.raises(TypeError, match="dataset_root must be"):
        validate_grasp_dataset("not_a_path")  # type: ignore[arg-type]

    empty_root = tmp_path / "empty_dataset"
    empty_root.mkdir()
    with pytest.raises(ValueError, match="No dataset record files"):
        validate_grasp_dataset(empty_root)

    with pytest.raises(TypeError, match="dataset_root must be"):
        SupervisedGraspDataset("not_a_path")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="No dataset record files"):
        SupervisedGraspDataset(empty_root)

    dataset_dir = tmp_path / "valid_dataset"
    dataset_dir.mkdir()
    sample_file = dataset_dir / "sample_001.npz"
    rng = np.random.default_rng()
    pc = rng.standard_normal((20, 3)).astype(np.float32)
    grasps = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    save_grasp_sample(
        sample_file,
        {
            "point_cloud": pc,
            "grasp_poses": grasps,
        },
    )

    pairs = SupervisedGraspDataset(dataset_dir, augment=True, seed=42)
    if not (len(pairs) == EXPECTED_TRAINING_PAIRS):
        raise AssertionError
    if not (pairs[0][0].shape == (20, 3)):
        raise AssertionError
    if not (pairs[0][1].shape == (9,)):
        raise AssertionError

    scored_dir = tmp_path / "scored_dataset"
    scored_dir.mkdir()
    scored_file = scored_dir / "scored_sample.npz"
    scores = np.array([0.2, 0.9], dtype=np.float32)
    save_grasp_sample(
        scored_file,
        {
            "point_cloud": pc,
            "grasp_poses": grasps,
            "scores": scores,
        },
    )
    filtered_pairs = SupervisedGraspDataset(
        scored_dir,
        min_grasp_score=0.5,
        score_repeat_factor=2,
        score_repeat_power=1.0,
    )
    if not (len(filtered_pairs) == EXPECTED_FILTERED_PAIRS):
        raise AssertionError

    invalid_file = dataset_dir / "invalid_sample.npz"
    np.savez(
        invalid_file,
        point_cloud=np.array("not_array"),
        grasp_poses=grasps,
    )
    with pytest.raises(ValueError, match="point_cloud must have shape"):
        SupervisedGraspDataset(dataset_dir)

    with pytest.raises(TypeError, match="index must be an integer"):
        pairs["invalid"]  # type: ignore[index]
    with pytest.raises(ValueError, match="num_points must be positive"):
        SupervisedGraspDataset(scored_dir, num_points=-1)

    resampled_pairs = SupervisedGraspDataset(scored_dir, num_points=10)
    if resampled_pairs[0][0].shape != (10, 3):
        raise AssertionError

    low_score_dir = tmp_path / "low_score_dataset"
    low_score_dir.mkdir()
    save_grasp_sample(
        low_score_dir / "low_score.npz",
        {
            "point_cloud": pc,
            "grasp_poses": grasps,
            "scores": np.array([0.1, 0.2], dtype=np.float32),
        },
    )
    with pytest.raises(ValueError, match="no grasp poses above"):
        SupervisedGraspDataset(low_score_dir, min_grasp_score=0.9)


def test_generate_analytical_grasps_validations_and_fallbacks() -> None:
    """Verify type and value checks on antipodal dot limits, multipliers, and relaxed grasp fallbacks."""
    pts = np.array([[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float64)
    normals = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError, match="points must be of shape"):
        generate_analytical_grasps(np.zeros(3), normals, 2, 0.05, rng)

    with pytest.raises(ValueError, match="normals must be of shape"):
        generate_analytical_grasps(pts, np.zeros(3), 2, 0.05, rng)

    with pytest.raises(ValueError, match="same length"):
        generate_analytical_grasps(pts, normals[:1], 2, 0.05, rng)

    with pytest.raises(TypeError, match="rng must be"):
        generate_analytical_grasps(pts, normals, 2, 0.05, "not_rng")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="num_grasps must be positive"):
        generate_analytical_grasps(pts, normals, 0, 0.05, rng)

    with pytest.raises(TypeError, match="allow_relaxed"):
        generate_analytical_grasps(pts, normals, 2, 0.05, rng, allow_relaxed="yes")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="relaxed_antipodal_dot"):
        generate_analytical_grasps(pts, normals, 2, 0.05, rng, relaxed_antipodal_dot=2.0)

    grasps = generate_analytical_grasps(pts, normals, num_grasps=2, gripper_width=0.05, rng=rng)
    if not (len(grasps) > 0):
        raise AssertionError
    if not (grasps[0].shape == (4, 4)):
        raise AssertionError

    perp_normals = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    relaxed_grasps = generate_analytical_grasps(
        pts,
        perp_normals,
        num_grasps=2,
        gripper_width=0.05,
        rng=rng,
        allow_relaxed=True,
        relaxed_antipodal_dot=-1.0,
    )
    if not (isinstance(relaxed_grasps, np.ndarray)):
        raise TypeError


def test_training_pairs_validations_and_error_paths(tmp_path: Path) -> None:
    """Verify validation checks on dataset records with corrupt or empty point clouds and grasp arrays."""
    with pytest.raises(TypeError, match="dataset_root"):
        validate_grasp_dataset("invalid")  # type: ignore[arg-type]

    empty_dir = tmp_path / "empty_ds"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match=r"No dataset record files|contains no valid grasp samples"):
        validate_grasp_dataset(empty_dir)

    with pytest.raises(TypeError, match="dataset_root"):
        SupervisedGraspDataset("invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"No dataset record files|Dataset is empty"):
        SupervisedGraspDataset(empty_dir)

    corrupt_ds = tmp_path / "corrupt_ds"
    corrupt_ds.mkdir()
    np.savez(
        corrupt_ds / "sample_invalid_pc.npz",
        point_cloud=np.array("not_an_array"),
        grasp_poses=np.eye(4, dtype=np.float32)[None],
    )

    with pytest.raises(ValueError, match="point_cloud must have shape"):
        SupervisedGraspDataset(corrupt_ds)

    corrupt_ds2 = tmp_path / "corrupt_ds2"
    corrupt_ds2.mkdir()
    np.savez(
        corrupt_ds2 / "sample_invalid_grasps.npz",
        point_cloud=np.zeros((10, 3), dtype=np.float32),
    )

    with pytest.raises(TypeError, match=r"grasp poses must be a numpy array"):
        SupervisedGraspDataset(corrupt_ds2)

    corrupt_ds3 = tmp_path / "corrupt_ds3"
    corrupt_ds3.mkdir()
    save_grasp_sample(
        corrupt_ds3 / "sample_empty_grasps.npz",
        {
            "point_cloud": np.zeros((10, 3), dtype=np.float32),
            "grasp_poses": np.zeros((0, 4, 4), dtype=np.float32),
        },
    )

    with pytest.raises(ValueError, match="has no target grasp poses"):
        SupervisedGraspDataset(corrupt_ds3)


def test_grasp_vector_invalid_inputs() -> None:
    """Verify shape and value validation exceptions on grasp SE3 and 9D vector conversion functions."""
    import torch  # noqa: PLC0415  # deferred heavy import

    with pytest.raises(ValueError, match="t_matrix must be a"):
        se3_to_vec(np.zeros((3, 3)))

    with pytest.raises(ValueError, match="x must have shape"):
        vec_to_se3(torch.zeros(9))


def test_transforms_additional_validations(tmp_path: Path) -> None:
    """Verify translation jitter and dataset index saving raise validation errors on invalid inputs."""
    rng = np.random.default_rng(42)
    trans = make_translation_jitter(rng, scale=0.01)

    with pytest.raises(TypeError, match="points must be a numpy array"):
        trans("not_array", None, None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="points shape must be"):
        trans(np.zeros((5, 2)), None, None)

    with pytest.raises(ValueError, match="points must contain only finite values"):
        trans(np.array([[np.nan, 0.0, 0.0]]), None, None)

    dir_path = tmp_path / "is_dir"
    dir_path.mkdir()
    with pytest.raises(ValueError, match="Failed to write dataset index"):
        save_grasp_dataset_index(dir_path, [{"record": "r1"}], filename="")


def test_generate_analytical_grasps_degenerate_parallel_normals() -> None:
    """Verify that generating analytical grasps handles parallel collinear normal vectors successfully."""
    rng = np.random.default_rng(42)
    pts = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float64)
    normals = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)

    grasps = generate_analytical_grasps(pts, normals, num_grasps=1, gripper_width=0.05, rng=rng)
    if not (isinstance(grasps, np.ndarray)):
        raise TypeError


def test_pointcloud_sensor_additional_coverage(tmp_path: Path) -> None:
    """Verify mesh reading error scenarios, empty files, and random generators inside point cloud sensor functions."""
    mesh_file = tmp_path / "mesh.obj"
    mesh_file.write_text("v 0 0 0\nv 0 0 0\nv 0 0 0\nf 1 2 3\n", encoding="utf-8")

    rng = np.random.default_rng(42)

    with pytest.raises(TypeError, match="rng must be a numpy random Generator"):
        sample_point_cloud_from_mesh(mesh_file, 10, "not_rng")  # type: ignore[arg-type]

    empty_mesh = tmp_path / "empty.obj"
    empty_mesh.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="is empty or invalid"):
        sample_point_cloud_from_mesh(empty_mesh, 10, rng)

    no_tri = tmp_path / "no_tri.obj"
    no_tri.write_text("v 0 0 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"is empty or invalid|has no triangles"):
        sample_point_cloud_from_mesh(no_tri, 10, rng)

    pc = sample_point_cloud_from_mesh(mesh_file, 5, rng)
    if not (pc.shape == (5, 3)):
        raise AssertionError


def test_generate_analytical_grasps_relaxed_fallback_and_parallel_normals() -> None:
    """Verify relaxed fallbacks on parallel normal vectors and empty arrays for zero-distance grasp points."""
    rng = np.random.default_rng(42)

    pts = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    normals = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    grasps = generate_analytical_grasps(pts, normals, num_grasps=1, gripper_width=0.05, allow_relaxed=True, rng=rng)
    if not (len(grasps) >= 1):
        raise AssertionError

    pts_zero_dist = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    grasps_empty = generate_analytical_grasps(
        pts_zero_dist,
        normals,
        num_grasps=1,
        gripper_width=0.05,
        allow_relaxed=True,
        rng=rng,
    )
    if not (len(grasps_empty) == 0):
        raise AssertionError

    # Trigger relaxed search fallback for colinear z_axis and average normal (lines 294-298)
    pts_colinear = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    normals_colinear = np.array([[0.95, 0.312, 0.0], [-0.95, -0.312, 0.0]])
    grasps_col = generate_analytical_grasps(
        pts_colinear,
        normals_colinear,
        num_grasps=1,
        gripper_width=0.05,
        allow_relaxed=True,
        strict_alignment_dot=0.99,
        relaxed_antipodal_dot=0.5,
        rng=rng,
    )
    if not (len(grasps_col) >= 1):
        raise AssertionError
