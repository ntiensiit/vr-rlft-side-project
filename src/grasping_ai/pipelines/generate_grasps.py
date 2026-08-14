from pathlib import Path
from typing import Any, cast

import numpy as np


def load_generated_grasps(
    grasps_path: Path,
    object_key: str | None = None,
) -> np.ndarray:
    """Load generated grasps from either on-disk format.

    Supports plain ``(K, 4, 4)`` arrays and pickled dicts mapping object
    identifiers to grasp arrays.

    Args:
        grasps_path: Path to a ``.npy`` grasp file.
        object_key: Required when the file contains multiple object entries.

    Returns:
        Grasp poses with shape ``(K, 4, 4)``.

    Raises:
        TypeError: If ``grasps_path`` is not a ``pathlib.Path`` instance.
        ValueError: If the file format or ``object_key`` selection is invalid.
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
    """Persist multi-object generated grasps as a pickled dict.

    Args:
        output_path: Destination ``.npy`` path.
        grasps_by_object: Mapping from object identifier to grasp arrays.

    Raises:
        TypeError: If ``output_path`` is not a ``pathlib.Path`` instance.
        ValueError: If writing the file fails.
    """
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path instance")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.save(output_path, cast(Any, grasps_by_object), allow_pickle=True)
    except Exception as e:
        raise ValueError(f"Failed to write generated grasps: {e}") from e


def write_generated_grasps_array(output_path: Path, grasp_poses: np.ndarray) -> None:
    """Persist a plain ``(K, 4, 4)`` grasp array.

    Args:
        output_path: Destination ``.npy`` path.
        grasp_poses: Candidate grasp poses to serialize.

    Raises:
        TypeError: If ``output_path`` is not a ``pathlib.Path`` instance.
        ValueError: If ``grasp_poses`` shape is invalid.
    """
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path instance")
    if grasp_poses.ndim != 3 or grasp_poses.shape[1:] != (4, 4):
        raise ValueError(
            f"grasp_poses must have shape (K, 4, 4), got {grasp_poses.shape}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, grasp_poses)
