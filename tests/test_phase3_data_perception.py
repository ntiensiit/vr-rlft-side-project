from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from grasping_ai.data.pointcloud_dataset import (
    discover_dataset_files,
    generate_analytical_grasps,
    iterate_grasp_dataset,
    load_grasp_sample,
    resolve_ycb_object_id,
    save_grasp_sample,
)
from grasping_ai.data.training_pairs import (
    build_supervised_training_pairs,
    validate_grasp_dataset,
)
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
)


def test_numpy_runtime_dependency_available():
    """Verify that numpy package is available and can be imported."""
    import numpy as np

    assert np.__version__ is not None


def test_phase1_package_import_remains_stable():
    """Verify that the package remains importable."""
    import grasping_ai

    assert grasping_ai.__name__ == "grasping_ai"


def test_data_config_file_exists():
    """Verify data configuration file existence."""
    path = Path("configs/data/default.yaml")
    assert path.is_file()


def test_object_config_file_exists():
    """Verify object configuration file existence."""
    path = Path("configs/object/default.yaml")
    assert path.is_file()


def test_prepare_data_creates_index_from_minimal_dataset(tmp_path):
    """Test discovering files and writing a JSON index."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    # Create dummy record
    record_path = dataset_root / "record_0.npz"
    save_grasp_sample(
        record_path,
        {
            "point_cloud": np.random.rand(10, 3).astype(np.float32),
            "grasp_poses": np.random.rand(2, 4, 4).astype(np.float32),
            "scores": np.random.rand(2).astype(np.float32),
            "object_id": "bottle",
        },
    )

    records = discover_dataset_files(dataset_root)
    assert len(records) == 1
    assert records[0] == record_path

    # Save index
    entries = [{"path": str(record)} for record in records]
    save_grasp_dataset_index(tmp_path, entries)

    index_file = tmp_path / "index.json"
    assert index_file.is_file()

    with open(index_file, encoding="utf-8") as f:
        loaded_entries = json.load(f)
    assert len(loaded_entries) == 1
    assert loaded_entries[0]["path"] == str(record_path)


def test_save_grasp_dataset_index_honors_custom_filename():
    """Verify that a custom index filename is honored instead of defaulting to index.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root = Path(tmpdir)
        entries = [{"path": "record_0.npz"}]
        save_grasp_dataset_index(dataset_root, entries, "custom_index.json")

        index_file = dataset_root / "custom_index.json"
        assert index_file.is_file()
        assert not (dataset_root / "index.json").exists()

        with open(index_file, encoding="utf-8") as f:
            loaded_entries = json.load(f)
        assert loaded_entries == entries


def test_prepare_data_rejects_missing_dataset_root():
    """Verify that discover_dataset_files raises FileNotFoundError for missing paths."""
    with pytest.raises(FileNotFoundError):
        discover_dataset_files(Path("non_existent_dataset_root"))
    with pytest.raises(TypeError):
        discover_dataset_files("not-a-path-object")  # type: ignore[arg-type]


def test_prepare_data_rejects_empty_dataset(tmp_path):
    """Verify discover_dataset_files raises ValueError for empty directories."""
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(ValueError, match="No dataset record files"):
        discover_dataset_files(empty_root)


def test_index_loader_reads_prepared_index(tmp_path):
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
    assert len(samples) == 2
    assert samples[0]["object_id"] == "obj1"
    assert samples[1]["object_id"] == "obj2"
    assert samples[0]["point_cloud"].shape == (10, 3)
    assert samples[1]["point_cloud"].shape == (15, 3)


def test_index_loader_rejects_invalid_structure(tmp_path):
    """Test loading invalid index inputs."""
    with pytest.raises(TypeError):
        list(iterate_grasp_dataset("not-a-path"))  # type: ignore[arg-type]


def test_save_grasp_sample_roundtrip(tmp_path):
    """Verify save_grasp_sample writes pickle-free archives readable by load_grasp_sample."""
    path = tmp_path / "record.npz"
    sample = {
        "point_cloud": np.random.randn(12, 3).astype(np.float32),
        "grasp_poses": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        "scores": np.array([0.1, 0.9], dtype=np.float32),
        "object_id": "mustard_bottle",
    }
    save_grasp_sample(path, sample)
    loaded = load_grasp_sample(path)
    assert np.allclose(loaded["point_cloud"], sample["point_cloud"])
    assert np.allclose(loaded["grasp_poses"], sample["grasp_poses"])
    assert np.allclose(loaded["scores"], sample["scores"])
    assert loaded["object_id"] == sample["object_id"]

    with np.load(path) as archive:
        assert "point_cloud" in archive
        assert archive["point_cloud"].dtype == np.float32


def test_save_grasp_sample_validates_point_cloud(tmp_path):
    """Verify save_grasp_sample rejects invalid point-cloud payloads before writing."""
    path = tmp_path / "invalid.npz"
    with pytest.raises(TypeError, match="'point_cloud' must be a numpy array"):
        save_grasp_sample(path, {"point_cloud": [[0.0, 0.0, 0.0]]})  # type: ignore[typeddict-item]

    with pytest.raises(ValueError, match="point_cloud must have shape"):
        save_grasp_sample(path, {"point_cloud": np.zeros((4, 2), dtype=np.float32)})


def test_load_grasp_sample_decodes_object_id_array(tmp_path):
    """Verify object identifiers stored as unicode arrays decode to plain strings."""
    path = tmp_path / "object_id_array.npz"
    np.savez(
        path,
        point_cloud=np.zeros((3, 3), dtype=np.float32),
        object_id=np.array(["004_sugar_box"], dtype=np.str_),
    )
    sample = load_grasp_sample(path)
    assert sample["object_id"] == "004_sugar_box"


def test_point_cloud_loader_reads_valid_npz_point_cloud(tmp_path):
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
    assert np.allclose(sample["point_cloud"], 1.0)
    assert sample["object_id"] == "box"


def test_point_cloud_loader_rejects_wrong_shape(tmp_path):
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


def test_point_cloud_loader_rejects_non_finite_values(tmp_path):
    """Verify loader validation rejects NaN and Inf."""
    path = tmp_path / "nan.npz"
    np.savez(path, point_cloud=np.array([[1.0, 2.0, np.nan]], dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        load_grasp_sample(path)


def test_resolve_ycb_object_id(tmp_path):
    """Verify YCB mesh path lookup resolution."""
    ycb_root = tmp_path / "ycb"
    ycb_root.mkdir()

    # Create dummy folders
    obj_dir = ycb_root / "004_sugar_box"
    obj_dir.mkdir()
    mesh_file = obj_dir / "textured.obj"
    mesh_file.write_text("mesh content", encoding="utf-8")

    resolved = resolve_ycb_object_id(ycb_root, "sugar_box")
    assert resolved == mesh_file

    # Non-existent object
    with pytest.raises(FileNotFoundError):
        resolve_ycb_object_id(ycb_root, "banana")


def test_make_random_rotation_jitter():
    """Verify SO(3) random rotation transform."""
    rng = np.random.default_rng(42)
    transform = make_random_rotation_jitter(rng)

    points = rng.random((5, 3))
    grasp_poses = rng.random((2, 4, 4))
    # Homogenize grasp poses
    grasp_poses[:, 3, :] = [0, 0, 0, 1]
    scores = np.array([0.9, 0.8])

    pts_rot, gps_rot, scs_rot = transform(points, grasp_poses, scores)

    assert pts_rot.shape == (5, 3)
    assert gps_rot is not None
    assert gps_rot.shape == (2, 4, 4)
    assert np.allclose(scs_rot, scores)

    # Verify that the relative distances between points are preserved (isometric transform)
    dist_orig = np.linalg.norm(points[0] - points[1])
    dist_rot = np.linalg.norm(pts_rot[0] - pts_rot[1])
    assert np.isclose(dist_orig, dist_rot)


def test_make_translation_jitter():
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
    assert np.allclose(pts_t, shift)
    assert gps_t is not None
    assert np.allclose(gps_t[:, :3, 3], shift)


def test_compose_transforms():
    """Verify sequential composition of transforms."""
    rng = np.random.default_rng(42)
    t1 = make_translation_jitter(rng, scale=0.01)
    t2 = make_translation_jitter(rng, scale=0.02)

    composed = compose_transforms(t1, t2)

    points = np.zeros((3, 3))
    pts_out, _, _ = composed(points, None, None)

    # Output should not be zero
    assert not np.allclose(pts_out, 0.0)


def test_sample_point_cloud():
    """Verify point cloud fixed size sampling."""
    rng = np.random.default_rng(42)
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Downsample/upsample
    sampled_1 = sample_point_cloud(points, 1, rng)
    assert sampled_1.shape == (1, 3)

    sampled_3 = sample_point_cloud(points, 3, rng)
    assert sampled_3.shape == (3, 3)


def test_normalize_point_cloud():
    """Verify centering and unit scaling."""
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    normalized = normalize_point_cloud(points)

    # Centroid should be at origin
    assert np.allclose(np.mean(normalized, axis=0), 0.0)
    # Max distance should be 1.0
    assert np.allclose(np.max(np.linalg.norm(normalized, axis=1)), 1.0)


def test_farthest_point_sampling():
    """Verify Farthest Point Sampling indices selection."""
    rng = np.random.default_rng(42)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    indices = farthest_point_sampling(points, 2, rng)
    assert indices.shape == (2,)
    # Farthest points should be selected (index 0 and 2, or index 1 and 2 depending on random start)
    assert 2 in indices


def test_estimate_point_cloud_normals():
    """Verify normal estimation using local PCA."""
    # Create flat point cloud in XY plane
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )
    normals = estimate_point_cloud_normals(points, neighborhood_size=4)
    # Normals should be perpendicular to XY plane (i.e. parallel to Z axis)
    for n in normals:
        assert np.isclose(np.abs(n[2]), 1.0, atol=1e-4)
        assert np.isclose(n[0], 0.0, atol=1e-4)
        assert np.isclose(n[1], 0.0, atol=1e-4)


def test_voxel_downsample():
    """Verify voxel downsampling centroids grouping."""
    points = np.array(
        [
            [0.01, 0.01, 0.01],
            [0.02, 0.02, 0.02],
            [0.9, 0.9, 0.9],
        ]
    )
    downsampled = voxel_downsample(points, voxel_size=0.1)
    # The first two points should fall in the same voxel and be averaged
    assert len(downsampled) == 2
    assert np.allclose(downsampled[0], [0.015, 0.015, 0.015]) or np.allclose(downsampled[1], [0.015, 0.015, 0.015])


def test_build_kdtree():
    """Verify KDTree construction and neighbor querying."""
    points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    kdtree = build_kdtree(points)

    # Query nearest to [1.1, 0, 0]
    dist, idx = kdtree.query(np.array([1.1, 0.0, 0.0]))
    assert idx == 0
    assert np.isclose(dist, 0.1)


def test_data_functions_do_not_leak_global_state(tmp_path):
    """Verify that calling voxel downsampling twice with the same inputs behaves deterministically and does not leak state."""
    # Test voxel downsampling determinism
    points = np.random.rand(10, 3)
    out1 = voxel_downsample(points, 0.2)
    out2 = voxel_downsample(points, 0.2)
    assert np.allclose(out1, out2)


def test_data_perception_error_handling(tmp_path):
    """Test all input validation and error handling paths in Phase 3 modules."""
    rng = np.random.default_rng(42)

    # --- pointcloud_dataset.py validations ---
    # discover_dataset_files validations
    with pytest.raises(TypeError, match="dataset_root"):
        discover_dataset_files("not-a-path")  # type: ignore[arg-type]
    file_path = tmp_path / "file.txt"
    file_path.write_text("not a dir", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
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
    assert np.allclose(normalize_point_cloud(zero_dist_pc), 0.0)

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
    assert np.allclose(small_normals, [0, 0, 1])

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

    # generate_analytical_grasps validation check (strict_alignment_dot bounds)
    with pytest.raises(ValueError, match="strict_alignment_dot must be in"):
        generate_analytical_grasps(
            np.zeros((2, 3)), np.zeros((2, 3)), num_grasps=1, gripper_width=0.05, strict_alignment_dot=2.0, rng=rng
        )

    # validate_grasp_dataset and build_supervised_training_pairs validations (empty dataset)
    from grasping_ai.data.training_pairs import build_supervised_training_pairs, validate_grasp_dataset

    empty_dir = tmp_path / "empty_dataset"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No dataset record files"):
        build_supervised_training_pairs(empty_dir)
    with pytest.raises(ValueError, match="No dataset record files"):
        validate_grasp_dataset(empty_dir)


def test_resolve_ycb_object_id_exact_and_fallbacks(tmp_path):
    """Test resolve_ycb_object_id for exact directory matches and different mesh formats."""
    ycb_root = tmp_path / "ycb"
    ycb_root.mkdir()

    # 1. Exact directory match & textured.obj exists
    obj_dir1 = ycb_root / "mustard_bottle"
    obj_dir1.mkdir()
    mesh1 = obj_dir1 / "textured.obj"
    mesh1.write_text("mesh mustard", encoding="utf-8")

    assert resolve_ycb_object_id(ycb_root, "mustard_bottle") == mesh1

    # 2. Substring directory match & .ply fallback
    obj_dir2 = ycb_root / "003_cracker_box"
    obj_dir2.mkdir()
    mesh2 = obj_dir2 / "cracker_box.ply"
    mesh2.write_text("mesh cracker", encoding="utf-8")

    assert resolve_ycb_object_id(ycb_root, "cracker_box") == mesh2

    # 3. Substring directory match & .obj fallback (no textured.obj or ply)
    obj_dir3 = ycb_root / "011_banana"
    obj_dir3.mkdir()
    mesh3 = obj_dir3 / "banana.obj"
    mesh3.write_text("mesh banana", encoding="utf-8")

    assert resolve_ycb_object_id(ycb_root, "banana") == mesh3

    # 4. Directory matches but contains no mesh files (returns directory Path)
    obj_dir4 = ycb_root / "025_mug"
    obj_dir4.mkdir()

    assert resolve_ycb_object_id(ycb_root, "mug") == obj_dir4


def test_transforms_optional_none_inputs():
    """Verify transforms when grasp_poses and scores are None."""
    rng = np.random.default_rng(42)
    rot = make_random_rotation_jitter(rng)
    trans = make_translation_jitter(rng, scale=0.1)

    points = rng.random((5, 3))

    p_rot, gp_rot, s_rot = rot(points, None, None)
    assert p_rot.shape == (5, 3)
    assert gp_rot is None
    assert s_rot is None

    p_trans, gp_trans, s_trans = trans(points, None, None)
    assert p_trans.shape == (5, 3)
    assert gp_trans is None
    assert s_trans is None


def test_perception_edge_cases():
    """Verify FPS and normals estimation edge cases."""
    rng = np.random.default_rng(42)

    # FPS with num_samples = 1
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    idx_1 = farthest_point_sampling(pts, 1, rng)
    assert idx_1.shape == (1,)

    # FPS with num_samples > n
    idx_5 = farthest_point_sampling(pts, 5, rng)
    assert idx_5.shape == (5,)

    # Normals with zero covariance (identical points)
    identical_pts = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    normals = estimate_point_cloud_normals(identical_pts, neighborhood_size=4)
    assert np.allclose(normals, [0, 0, 1])


def test_acquire_point_cloud_stream_yields_saved_clouds():
    """Verify that acquire_point_cloud_stream correctly yields point clouds loaded from numpy array files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        paths = []
        for i in range(3):
            path = root / f"obs_{i}.npy"
            np.save(path, np.random.randn(10 + i, 3).astype(np.float32))
            paths.append(path)
        clouds = list(acquire_point_cloud_stream(paths))
        assert len(clouds) == 3
        assert clouds[0].shape == (10, 3)
        assert clouds[2].shape == (12, 3)


def test_acquire_point_cloud_stream_rejects_invalid_input():
    """Verify that acquire_point_cloud_stream raises TypeError for invalid paths and FileNotFoundError for missing files."""
    with pytest.raises(TypeError, match=r"list of pathlib\.Path"):
        list(acquire_point_cloud_stream(["obs.npy"]))  # type: ignore[list-item]

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        missing = root / "missing.npy"
        with pytest.raises(FileNotFoundError):
            list(acquire_point_cloud_stream([missing]))


def test_merge_point_clouds_concatenates():
    """Verify that merge_point_clouds correctly concatenates multiple point cloud arrays into a single array."""
    a = np.random.randn(5, 3).astype(np.float32)
    b = np.random.randn(7, 3).astype(np.float32)
    merged = merge_point_clouds([a, b])
    assert merged.shape == (12, 3)
    assert np.allclose(merged[:5], a)
    assert np.allclose(merged[5:], b)


def test_merge_point_clouds_empty_list():
    """Verify that merge_point_clouds returns an empty array of correct shape when given an empty list."""
    merged = merge_point_clouds([])
    assert merged.shape == (0, 3)


def test_merge_point_clouds_validation():
    """Verify that merge_point_clouds raises TypeError or ValueError for invalid list or array shapes."""
    with pytest.raises(TypeError, match="list of numpy arrays"):
        merge_point_clouds("not_a_list")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape \\(N, 3\\)"):
        merge_point_clouds([np.random.randn(4, 2)])


def test_training_pairs_validations_and_augmentation(tmp_path: Path) -> None:
    """Verify validations on dataset root paths and dataset data augmentation inside training pair utilities."""
    with pytest.raises(TypeError, match="dataset_root must be"):
        validate_grasp_dataset("not_a_path")  # type: ignore[arg-type]

    empty_root = tmp_path / "empty_dataset"
    empty_root.mkdir()
    with pytest.raises(ValueError, match="No dataset record files"):
        validate_grasp_dataset(empty_root)

    with pytest.raises(TypeError, match="dataset_root must be"):
        build_supervised_training_pairs("not_a_path")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="No dataset record files"):
        build_supervised_training_pairs(empty_root)

    dataset_dir = tmp_path / "valid_dataset"
    dataset_dir.mkdir()
    sample_file = dataset_dir / "sample_001.npz"
    pc = np.random.randn(20, 3).astype(np.float32)
    grasps = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    save_grasp_sample(
        sample_file,
        {
            "point_cloud": pc,
            "grasp_poses": grasps,
        },
    )

    pairs = build_supervised_training_pairs(dataset_dir, augment=True, seed=42)
    assert len(pairs) == 2
    assert pairs[0][0].shape == (20, 3)
    assert pairs[0][1].shape == (9,)

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
    filtered_pairs = build_supervised_training_pairs(
        scored_dir, min_grasp_score=0.5, score_repeat_factor=2, score_repeat_power=1.0
    )
    assert len(filtered_pairs) == 2

    invalid_file = dataset_dir / "invalid_sample.npz"
    np.savez(
        invalid_file,
        point_cloud=np.array("not_array"),
        grasp_poses=grasps,
    )
    with pytest.raises(ValueError, match="point_cloud must have shape"):
        build_supervised_training_pairs(dataset_dir)


def test_generate_analytical_grasps_validations_and_fallbacks() -> None:
    """Verify type and value checks on antipodal dot limits, multipliers, and relaxed grasp fallbacks."""
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.04]], dtype=np.float64)
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float64)
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
    assert len(grasps) > 0
    assert grasps[0].shape == (4, 4)

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
    assert isinstance(relaxed_grasps, np.ndarray)


def test_training_pairs_validations_and_error_paths(tmp_path: Path) -> None:
    """Verify validation checks on dataset records with corrupt or empty point clouds and grasp arrays."""
    from grasping_ai.data.training_pairs import (
        build_supervised_training_pairs,
        validate_grasp_dataset,
    )

    with pytest.raises(TypeError, match="dataset_root"):
        validate_grasp_dataset("invalid")  # type: ignore[arg-type]

    empty_dir = tmp_path / "empty_ds"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match=r"No dataset record files|contains no valid grasp samples"):
        validate_grasp_dataset(empty_dir)

    with pytest.raises(TypeError, match="dataset_root"):
        build_supervised_training_pairs("invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"No dataset record files|Dataset is empty"):
        build_supervised_training_pairs(empty_dir)

    corrupt_ds = tmp_path / "corrupt_ds"
    corrupt_ds.mkdir()
    np.savez(
        corrupt_ds / "sample_invalid_pc.npz",
        point_cloud=np.array("not_an_array"),
        grasp_poses=np.eye(4, dtype=np.float32)[None],
    )

    with pytest.raises(ValueError, match="point_cloud must have shape"):
        build_supervised_training_pairs(corrupt_ds)

    corrupt_ds2 = tmp_path / "corrupt_ds2"
    corrupt_ds2.mkdir()
    np.savez(
        corrupt_ds2 / "sample_invalid_grasps.npz",
        point_cloud=np.zeros((10, 3), dtype=np.float32),
    )

    with pytest.raises(TypeError, match=r"grasp poses must be a numpy array"):
        build_supervised_training_pairs(corrupt_ds2)

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
        build_supervised_training_pairs(corrupt_ds3)


def test_grasp_vector_invalid_inputs() -> None:
    """Verify shape and value validation exceptions on grasp SE3 and 9D vector conversion functions."""
    import torch

    from grasping_ai.data.grasp_vector import se3_to_vec, vec_to_se3

    with pytest.raises(ValueError, match="t_matrix must be a"):
        se3_to_vec(np.zeros((3, 3)))

    with pytest.raises(ValueError, match="x must have shape"):
        vec_to_se3(torch.zeros(9))


def test_transforms_additional_validations(tmp_path: Path) -> None:
    """Verify that point cloud translation jitter and dataset index saving raise appropriate validation errors on invalid inputs."""
    from grasping_ai.data.transforms import (
        make_translation_jitter,
        save_grasp_dataset_index,
    )

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
    from grasping_ai.data.pointcloud_dataset import generate_analytical_grasps

    rng = np.random.default_rng(42)
    pts = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float64)
    normals = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)

    grasps = generate_analytical_grasps(pts, normals, num_grasps=1, gripper_width=0.05, rng=rng)
    assert isinstance(grasps, np.ndarray)


def test_pointcloud_sensor_additional_coverage(tmp_path: Path) -> None:
    """Verify mesh reading error scenarios, empty files, and random generators inside point cloud sensor functions."""
    from grasping_ai.sensors.pointcloud_sensor import sample_point_cloud_from_mesh

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
    assert pc.shape == (5, 3)


def test_generate_analytical_grasps_relaxed_fallback_and_parallel_normals() -> None:
    """Verify relaxed fallbacks on parallel normal vectors and empty arrays for zero-distance grasp points."""
    from grasping_ai.data.pointcloud_dataset import generate_analytical_grasps

    rng = np.random.default_rng(42)

    pts = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    normals = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    grasps = generate_analytical_grasps(pts, normals, num_grasps=1, gripper_width=0.05, allow_relaxed=True, rng=rng)
    assert len(grasps) >= 1

    pts_zero_dist = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    grasps_empty = generate_analytical_grasps(
        pts_zero_dist,
        normals,
        num_grasps=1,
        gripper_width=0.05,
        allow_relaxed=True,
        rng=rng,
    )
    assert len(grasps_empty) == 0

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
    assert len(grasps_col) >= 1
