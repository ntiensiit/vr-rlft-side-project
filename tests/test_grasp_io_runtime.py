from pathlib import Path

import numpy as np
import pytest

from grasping_ai.inference.grasp_inference_runtime import (
    build_grasp_generator_from_checkpoint,
    load_inference_point_cloud,
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
