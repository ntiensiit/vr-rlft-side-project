from pathlib import Path

import numpy as np
import pytest

import grasping_ai.pipelines.generate_grasps as gg
from grasping_ai.inference.grasp_inference_runtime import (
    build_grasp_generator_from_checkpoint,
    load_inference_point_cloud,
    run_single_object_grasp_inference,
)
from grasping_ai.pipelines.generate_grasps import (
    generate_grasps_for_dataset,
    load_generated_grasps,
    write_generated_grasps,
    write_generated_grasps_array,
)


def _sample_grasps(count: int = 2) -> np.ndarray:
    grasps = np.tile(np.eye(4), (count, 1, 1))
    grasps[:, :3, 3] = np.arange(count)[:, None]
    return grasps


def test_load_generated_grasps_plain_array(tmp_path: Path) -> None:
    grasps = _sample_grasps(3)
    path = tmp_path / "grasps.npy"
    np.save(path, grasps)

    loaded = load_generated_grasps(path)
    assert loaded.shape == (3, 4, 4)
    assert np.allclose(loaded, grasps)


def test_load_generated_grasps_single_object_dict(tmp_path: Path) -> None:
    grasps = _sample_grasps(2)
    path = tmp_path / "grasps.npy"
    write_generated_grasps(path, {"003_cracker_box": grasps})

    loaded = load_generated_grasps(path)
    assert np.allclose(loaded, grasps)


def test_load_generated_grasps_multi_object_dict_requires_key(tmp_path: Path) -> None:
    path = tmp_path / "grasps.npy"
    write_generated_grasps(
        path,
        {
            "003_cracker_box": _sample_grasps(2),
            "004_sugar_box": _sample_grasps(1),
        },
    )

    with pytest.raises(ValueError, match="object_key is required"):
        load_generated_grasps(path)

    keyed = load_generated_grasps(path, object_key="004_sugar_box")
    assert keyed.shape == (1, 4, 4)


def test_write_generated_grasps_array_validates_shape(tmp_path: Path) -> None:
    path = tmp_path / "grasps.npy"
    with pytest.raises(ValueError, match="grasp_poses must have shape"):
        write_generated_grasps_array(path, np.zeros((2, 3)))


def test_load_inference_point_cloud_from_observation(tmp_path: Path) -> None:
    cloud = np.random.randn(64, 3).astype(np.float32)
    obs_path = tmp_path / "obs.npy"
    np.save(obs_path, cloud)

    loaded = load_inference_point_cloud(obs_path, None, None, num_grasps=4, seed=0)
    assert np.allclose(loaded, cloud)


def test_load_inference_point_cloud_rejects_conflicting_inputs(tmp_path: Path) -> None:
    obs_path = tmp_path / "obs.npy"
    np.save(obs_path, np.zeros((8, 3)))

    with pytest.raises(ValueError, match="not both"):
        load_inference_point_cloud(
            obs_path, tmp_path, "003_cracker_box", num_grasps=4, seed=0
        )


def test_build_grasp_generator_from_checkpoint_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method must be"):
        build_grasp_generator_from_checkpoint(
            "invalid", {}, feature_dim=16, num_steps=4, device="cpu"
        )


def test_build_grasp_generator_from_checkpoint_diffusion_and_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    called_methods = []

    def mock_diffusion(checkpoint, feature_dim, num_steps, device, seed):
        called_methods.append("diffusion")
        return "diffusion_generator"

    def mock_flow(checkpoint, feature_dim, num_steps, device, seed):
        called_methods.append("flow")
        return "flow_generator"

    monkeypatch.setattr(
        "grasping_ai.inference.grasp_inference_runtime.build_diffusion_grasp_generator",
        mock_diffusion,
    )
    monkeypatch.setattr(
        "grasping_ai.inference.grasp_inference_runtime.build_flow_grasp_generator",
        mock_flow,
    )

    gen_diff = build_grasp_generator_from_checkpoint("diffusion", {}, 16, 4, "cpu", 42)
    assert gen_diff == "diffusion_generator"

    gen_flow = build_grasp_generator_from_checkpoint("flow", {}, 16, 4, "cpu", 42)
    assert gen_flow == "flow_generator"

    assert called_methods == ["diffusion", "flow"]


def test_load_inference_point_cloud_validations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Provide either observation_path or both"):
        load_inference_point_cloud(None, None, None, 4, 0)

    non_existent = tmp_path / "missing.npy"
    with pytest.raises(FileNotFoundError, match="Observation file not found"):
        load_inference_point_cloud(non_existent, None, None, 4, 0)

    with pytest.raises(FileNotFoundError, match="YCB root directory not found"):
        load_inference_point_cloud(None, tmp_path / "missing_dir", "003_cracker_box", 4, 0)

    bad_cloud_path = tmp_path / "bad.npy"
    np.save(bad_cloud_path, np.zeros((10,)))
    with pytest.raises(ValueError, match="point_cloud must have shape"):
        load_inference_point_cloud(bad_cloud_path, None, None, 4, 0)


def test_load_inference_point_cloud_from_ycb_mesh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ycb_dir = tmp_path / "ycb"
    ycb_dir.mkdir()

    fake_mesh_path = ycb_dir / "mesh.ply"
    fake_mesh_path.touch()

    monkeypatch.setattr(
        "grasping_ai.inference.grasp_inference_runtime.resolve_ycb_object_id",
        lambda root, obj_id: fake_mesh_path,
    )
    monkeypatch.setattr(
        "grasping_ai.inference.grasp_inference_runtime.sample_point_cloud_from_mesh",
        lambda mesh_path, count, rng: np.ones((count, 3), dtype=np.float32),
    )

    pc = load_inference_point_cloud(None, ycb_dir, "003_cracker_box", num_grasps=2, seed=42)
    assert pc.shape == (16, 3)


def test_run_single_object_grasp_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    obs_path = tmp_path / "obs.npy"
    np.save(obs_path, np.zeros((32, 3)))

    out_path = tmp_path / "output" / "grasps.npy"
    ckpt_path = tmp_path / "ckpt.pt"
    ckpt_path.touch()

    dummy_grasps = _sample_grasps(2)

    monkeypatch.setattr(
        "grasping_ai.inference.grasp_inference_runtime.load_grasp_model_checkpoint",
        lambda path, device: {},
    )
    monkeypatch.setattr(
        "grasping_ai.inference.grasp_inference_runtime.build_grasp_generator_from_checkpoint",
        lambda method, ckpt, f_dim, steps, dev, seed: "dummy_gen",
    )
    monkeypatch.setattr(
        "grasping_ai.inference.grasp_generator.generate_candidate_grasps",
        lambda gen, pc, n: dummy_grasps,
    )

    res = run_single_object_grasp_inference(
        ckpt_path,
        out_path,
        "diffusion",
        feature_dim=16,
        num_steps=4,
        num_grasps=2,
        device="cpu",
        seed=42,
        observation_path=obs_path,
    )
    assert np.allclose(res, dummy_grasps)
    assert out_path.is_file()
    assert np.allclose(np.load(out_path), dummy_grasps)


def test_generate_grasps_for_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    pc1 = np.zeros((10, 3))
    pc2 = np.ones((10, 3))
    dummy_gen = "dummy_gen"

    def mock_generate(gen, pc, count):
        return _sample_grasps(count)

    monkeypatch.setattr(gg, "generate_candidate_grasps", mock_generate)

    results = generate_grasps_for_dataset([pc1, pc2], dummy_gen, 2)  # type: ignore[arg-type]
    assert len(results) == 2
    assert results[0].shape == (2, 4, 4)


def test_generate_grasps_type_and_value_errors(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="grasps_path must be"):
        load_generated_grasps("not_a_path")  # type: ignore[arg-type]

    dict_path = tmp_path / "dict_grasps.npy"
    write_generated_grasps(dict_path, {"obj1": _sample_grasps(1), "obj2": _sample_grasps(1)})
    with pytest.raises(ValueError, match="Object key 'missing_key' not found"):
        load_generated_grasps(dict_path, object_key="missing_key")

    arr_4d_path = tmp_path / "arr_4d.npy"
    np.save(arr_4d_path, np.zeros((1, 3, 4, 4)))
    loaded_4d = load_generated_grasps(arr_4d_path)
    assert loaded_4d.shape == (3, 4, 4)

    with pytest.raises(TypeError, match="output_path must be"):
        write_generated_grasps("not_a_path", {})  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="output_path must be"):
        write_generated_grasps_array("not_a_path", _sample_grasps(1))  # type: ignore[arg-type]
