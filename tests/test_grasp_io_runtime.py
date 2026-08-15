from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

from grasping_ai.inference.grasp_generator import generate_candidate_grasps
from grasping_ai.inference.grasp_inference_runtime import (
    run_single_object_grasp_inference,
)
from grasping_ai.pipelines.generate_grasps import (
    load_generated_grasps,
    write_generated_grasps,
    write_generated_grasps_array,
)


def _sample_grasps(count: int = 2) -> np.ndarray:
    grasps = np.tile(np.eye(4), (count, 1, 1))
    grasps[:, :3, 3] = np.arange(count)[:, None]
    return grasps


def test_load_generated_grasps_plain_array(tmp_path: Path) -> None:
    """Verify loading generated grasps from a plain numpy array file."""
    grasps = _sample_grasps(3)
    path = tmp_path / "grasps.npy"
    np.save(path, grasps)

    loaded = load_generated_grasps(path)
    assert loaded.shape == (3, 4, 4)
    assert np.allclose(loaded, grasps)


def test_load_generated_grasps_single_object_dict(tmp_path: Path) -> None:
    """Verify loading generated grasps from a dictionary containing a single object key."""
    grasps = _sample_grasps(2)
    path = tmp_path / "grasps.npy"
    write_generated_grasps(path, {"003_cracker_box": grasps})

    loaded = load_generated_grasps(path)
    assert np.allclose(loaded, grasps)


def test_load_generated_grasps_multi_object_dict_requires_key(tmp_path: Path) -> None:
    """Verify that loading multi-object dictionaries requires specifying a target object key."""
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
    """Verify that write_generated_grasps_array raises ValueError on invalid array shapes."""
    path = tmp_path / "grasps.npy"
    with pytest.raises(ValueError, match="grasp_poses must have shape"):
        write_generated_grasps_array(path, np.zeros((2, 3)))


def test_run_single_object_grasp_inference_from_observation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that grasp inference runtime executes correctly on input observation point cloud files."""
    obs_path = tmp_path / "obs.npy"
    np.save(obs_path, np.random.randn(64, 3).astype(np.float32))

    out_path = tmp_path / "output" / "grasps.npy"
    ckpt_path = tmp_path / "ckpt.pt"
    ckpt_path.touch()
    dummy_grasps = _sample_grasps(2)

    monkeypatch.setattr(
        "grasping_ai.inference.grasp_inference_runtime.load_grasp_model_checkpoint",
        lambda path, device: {},
    )
    monkeypatch.setattr(
        "grasping_ai.inference.grasp_inference_runtime.build_diffusion_grasp_generator",
        lambda checkpoint, feature_dim, num_steps, device, seed: "dummy_gen",
    )
    monkeypatch.setattr(
        "grasping_ai.inference.grasp_inference_runtime.generate_candidate_grasps",
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


def test_run_single_object_grasp_inference_rejects_conflicting_inputs(
    tmp_path: Path,
) -> None:
    """Verify that grasp inference runtime raises ValueError if both observation_path and YCB inputs are provided."""
    obs_path = tmp_path / "obs.npy"
    np.save(obs_path, np.zeros((8, 3)))
    ckpt_path = tmp_path / "ckpt.pt"
    ckpt_path.touch()

    with pytest.raises(ValueError, match="not both"):
        run_single_object_grasp_inference(
            ckpt_path,
            tmp_path / "out.npy",
            "diffusion",
            feature_dim=16,
            num_steps=4,
            num_grasps=2,
            device="cpu",
            seed=0,
            observation_path=obs_path,
            ycb_root=tmp_path,
            object_id="003_cracker_box",
        )


def test_run_single_object_grasp_inference_rejects_unknown_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that grasp inference runtime raises ValueError on unknown inference methods."""
    obs_path = tmp_path / "obs.npy"
    np.save(obs_path, np.zeros((8, 3)))
    ckpt_path = tmp_path / "ckpt.pt"
    ckpt_path.touch()
    monkeypatch.setattr(
        "grasping_ai.inference.grasp_inference_runtime.load_grasp_model_checkpoint",
        lambda path, device: {},
    )

    with pytest.raises(ValueError, match="method must be"):
        run_single_object_grasp_inference(
            ckpt_path,
            tmp_path / "out.npy",
            "invalid",
            feature_dim=16,
            num_steps=4,
            num_grasps=2,
            device="cpu",
            seed=42,
            observation_path=obs_path,
        )


def test_run_single_object_grasp_inference_diffusion_and_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that grasp inference runtime can run both diffusion and flow matching pipelines."""
    obs_path = tmp_path / "obs.npy"
    np.save(obs_path, np.zeros((32, 3)))
    ckpt_path = tmp_path / "ckpt.pt"
    ckpt_path.touch()
    called_methods: list[str] = []

    def mock_diffusion(checkpoint, feature_dim, num_steps, device, seed):
        called_methods.append("diffusion")
        return "diffusion_generator"

    def mock_flow(checkpoint, feature_dim, num_steps, device, seed):
        called_methods.append("flow")
        return "flow_generator"

    monkeypatch.setattr(
        "grasping_ai.inference.grasp_inference_runtime.load_grasp_model_checkpoint",
        lambda path, device: {},
    )
    monkeypatch.setattr(
        "grasping_ai.inference.grasp_inference_runtime.build_diffusion_grasp_generator",
        mock_diffusion,
    )
    monkeypatch.setattr(
        "grasping_ai.inference.grasp_inference_runtime.build_flow_grasp_generator",
        mock_flow,
    )
    monkeypatch.setattr(
        "grasping_ai.inference.grasp_inference_runtime.generate_candidate_grasps",
        lambda gen, pc, n: _sample_grasps(n),
    )

    run_single_object_grasp_inference(
        ckpt_path,
        tmp_path / "diff.npy",
        "diffusion",
        feature_dim=16,
        num_steps=4,
        num_grasps=2,
        device="cpu",
        seed=42,
        observation_path=obs_path,
    )
    run_single_object_grasp_inference(
        ckpt_path,
        tmp_path / "flow.npy",
        "flow",
        feature_dim=16,
        num_steps=4,
        num_grasps=2,
        device="cpu",
        seed=42,
        observation_path=obs_path,
    )
    assert called_methods == ["diffusion", "flow"]


def test_run_single_object_grasp_inference_validations(tmp_path: Path) -> None:
    """Verify that grasp inference runtime validates missing paths and invalid point cloud shapes."""
    ckpt_path = tmp_path / "ckpt.pt"
    ckpt_path.touch()

    with pytest.raises(ValueError, match="Provide either observation_path or both"):
        run_single_object_grasp_inference(
            ckpt_path,
            tmp_path / "out.npy",
            "diffusion",
            feature_dim=16,
            num_steps=4,
            num_grasps=2,
            device="cpu",
            seed=0,
        )

    non_existent = tmp_path / "missing.npy"
    with pytest.raises(FileNotFoundError, match="Observation file not found"):
        run_single_object_grasp_inference(
            ckpt_path,
            tmp_path / "out.npy",
            "diffusion",
            feature_dim=16,
            num_steps=4,
            num_grasps=2,
            device="cpu",
            seed=0,
            observation_path=non_existent,
        )

    with pytest.raises(FileNotFoundError, match="YCB root directory not found"):
        run_single_object_grasp_inference(
            ckpt_path,
            tmp_path / "out.npy",
            "diffusion",
            feature_dim=16,
            num_steps=4,
            num_grasps=2,
            device="cpu",
            seed=0,
            ycb_root=tmp_path / "missing_dir",
            object_id="003_cracker_box",
        )

    bad_cloud_path = tmp_path / "bad.npy"
    np.save(bad_cloud_path, np.zeros((10,)))
    with pytest.raises(ValueError, match="point_cloud must have shape"):
        run_single_object_grasp_inference(
            ckpt_path,
            tmp_path / "out.npy",
            "diffusion",
            feature_dim=16,
            num_steps=4,
            num_grasps=2,
            device="cpu",
            seed=0,
            observation_path=bad_cloud_path,
        )


def test_run_single_object_grasp_inference_from_ycb_mesh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that grasp inference runtime successfully samples meshes for YCB objects to run inference."""
    ycb_dir = tmp_path / "ycb"
    ycb_dir.mkdir()
    fake_mesh_path = ycb_dir / "mesh.ply"
    fake_mesh_path.touch()
    ckpt_path = tmp_path / "ckpt.pt"
    ckpt_path.touch()

    monkeypatch.setattr(
        "grasping_ai.inference.grasp_inference_runtime.resolve_ycb_object_id",
        lambda root, obj_id: fake_mesh_path,
    )
    monkeypatch.setattr(
        "grasping_ai.inference.grasp_inference_runtime.sample_point_cloud_from_mesh",
        lambda mesh_path, count, rng: np.ones((count, 3), dtype=np.float32),
    )
    monkeypatch.setattr(
        "grasping_ai.inference.grasp_inference_runtime.load_grasp_model_checkpoint",
        lambda path, device: {},
    )
    monkeypatch.setattr(
        "grasping_ai.inference.grasp_inference_runtime.build_diffusion_grasp_generator",
        lambda checkpoint, feature_dim, num_steps, device, seed: "gen",
    )
    monkeypatch.setattr(
        "grasping_ai.inference.grasp_inference_runtime.generate_candidate_grasps",
        lambda gen, pc, n: _sample_grasps(n),
    )

    run_single_object_grasp_inference(
        ckpt_path,
        tmp_path / "out.npy",
        "diffusion",
        feature_dim=16,
        num_steps=4,
        num_grasps=2,
        device="cpu",
        seed=42,
        ycb_root=ycb_dir,
        object_id="003_cracker_box",
    )


def test_generate_candidate_grasps_batch() -> None:
    """Verify that generate_candidate_grasps successfully loops over point cloud batches."""
    pc1 = np.zeros((10, 3))
    pc2 = np.ones((10, 3))

    def dummy_gen(point_cloud: np.ndarray, count: int) -> np.ndarray:
        return _sample_grasps(count)

    results = [generate_candidate_grasps(dummy_gen, pc, 2) for pc in [pc1, pc2]]
    assert len(results) == 2
    assert results[0].shape == (2, 4, 4)


def test_generate_grasps_type_and_value_errors(tmp_path: Path) -> None:
    """Verify error handling on loading/saving files with invalid paths, missing dictionary keys, or 4D formats."""
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


def test_write_generated_grasps_exception_and_array_writer(monkeypatch, tmp_path: Path) -> None:
    """Verify that disk save failures raise custom ValueErrors, and that plain array saves function correctly."""

    def bad_save(*args, **kwargs):
        msg = "Disk write error"
        raise OSError(msg)

    monkeypatch.setattr(np, "save", bad_save)

    with pytest.raises(ValueError, match="Failed to write generated grasps"):
        write_generated_grasps(tmp_path / "out.npy", {"obj1": _sample_grasps(1)})

    monkeypatch.undo()
    arr_file = tmp_path / "grasps_plain.npy"
    write_generated_grasps_array(arr_file, _sample_grasps(2))
    assert arr_file.is_file()
