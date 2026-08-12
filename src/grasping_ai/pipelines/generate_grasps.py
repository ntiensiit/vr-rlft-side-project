from pathlib import Path
from typing import Any, cast

import numpy as np

from grasping_ai.inference.grasp_generator import (
    GraspPoseGenerator,
    generate_candidate_grasps,
)


def generate_grasps_for_dataset(
    dataset_point_clouds: list[np.ndarray],
    grasp_generator: GraspPoseGenerator,
    num_candidates: int,
) -> list[np.ndarray]:
    """Generate grasp candidates for a list of object point clouds."""
    return [
        generate_candidate_grasps(grasp_generator, pc, num_candidates)
        for pc in dataset_point_clouds
    ]


def load_generated_grasps(
    grasps_path: Path,
    object_key: str | None = None,
) -> np.ndarray:
    """Load generated grasps from either on-disk format.

    Supports:
    * Plain array with shape ``(K, 4, 4)`` (runtime workflow / simulation).
    * Pickled dict mapping object identifiers to grasp arrays (artifact chain).
    """
    if not isinstance(grasps_path, Path):
        raise TypeError("grasps_path must be a pathlib.Path instance")

    loaded = np.load(grasps_path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.dtype == object:
        data: object = loaded.item()
    else:
        data = loaded

    if isinstance(data, dict):
        keyed = cast(dict[str, np.ndarray], data)
        if object_key is not None:
            if object_key not in keyed:
                raise ValueError(
                    f"Object key '{object_key}' not found in grasp dictionary: "
                    f"{list(keyed.keys())}"
                )
            return keyed[object_key]
        if len(keyed) == 1:
            return next(iter(keyed.values()))
        raise ValueError(
            "object_key is required when the grasp file contains multiple objects"
        )

    grasps = cast(np.ndarray, data)
    if grasps.ndim == 4 and grasps.shape[0] == 1:
        return grasps[0]
    return grasps


def write_generated_grasps(
    output_path: Path,
    grasps_by_object: dict[str, np.ndarray],
) -> None:
    """Persist multi-object generated grasps as a pickled dict (artifact-chain format)."""
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path instance")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.save(output_path, cast(Any, grasps_by_object), allow_pickle=True)
    except Exception as e:
        raise ValueError(f"Failed to write generated grasps: {e}") from e


def write_generated_grasps_array(output_path: Path, grasp_poses: np.ndarray) -> None:
    """Persist a plain ``(K, 4, 4)`` grasp array (runtime-workflow format)."""
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path instance")
    if grasp_poses.ndim != 3 or grasp_poses.shape[1:] != (4, 4):
        raise ValueError(
            f"grasp_poses must have shape (K, 4, 4), got {grasp_poses.shape}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, grasp_poses)
