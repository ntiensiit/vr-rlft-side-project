"""Generate grasps from object point clouds."""

from __future__ import annotations

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG

from grasping_ai.utils.path_validation import require_path

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

GRASP_OBJECT_BATCH_NDIM = int(FLATTENED_YAML_CONFIG.get("grasp.object_batch_ndim", 4))
GRASP_POSES_NDIM = int(FLATTENED_YAML_CONFIG.get("grasp.poses_ndim", 3))
SE3_MATRIX_SHAPE = tuple(int(v) for v in FLATTENED_YAML_CONFIG.get("grasp.se3_matrix_shape", [4, 4]))

if TYPE_CHECKING:
    from pathlib import Path

def _parse_grasp_dict(data: dict[object, object]) -> dict[str, np.ndarray]:
    """Validate a pickled object-id to grasp-array mapping.

    Args:
        data: Raw dictionary loaded from a multi-object grasp file.

    Returns:
        Mapping from object identifier strings to grasp arrays.

    Raises:
        TypeError: If any key is not a string or any value is not a
            ``numpy.ndarray``.
    """
    parsed: dict[str, np.ndarray] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise TypeError("Grasp dictionary keys must be strings")
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Grasp dictionary value for '{key}' must be a numpy array")
        parsed[key] = value
    return parsed

def _parse_grasp_array(data: object) -> np.ndarray:
    """Validate a plain grasp pose array payload.

    Args:
        data: Deserialized grasp file contents.

    Returns:
        Grasp pose array, typically with shape ``(K, 4, 4)``.

    Raises:
        TypeError: If ``data`` is not a ``numpy.ndarray``.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Grasp file must contain a numpy array or object dictionary")
    return data

def _numpy_pickle_payload(obj: object) -> np.ndarray:
    """Wrap an arbitrary Python object for ``np.save(..., allow_pickle=True)``.

    Args:
        obj: Python object to persist, such as a grasp dictionary.

    Returns:
        Zero-dimensional ``object`` dtype array containing ``obj``.
    """
    payload = np.empty((), dtype=object)
    payload[()] = obj
    return payload

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
    require_path(grasps_path, "grasps_path")

    loaded = np.load(grasps_path, allow_pickle=True)
    logger.info("Loaded grasps from: {}", grasps_path)
    if isinstance(loaded, np.ndarray) and loaded.dtype == object:
        data: object = loaded.item()
    else:
        data = loaded

    if isinstance(data, dict):
        keyed = _parse_grasp_dict(data)
        if object_key is not None:
            if object_key not in keyed:
                raise ValueError(f"Object key '{object_key}' not found in grasp dictionary: {list(keyed.keys())}")
            return keyed[object_key]
        if len(keyed) == 1:
            return next(iter(keyed.values()))
        raise ValueError("object_key is required when the grasp file contains multiple objects")

    grasps = _parse_grasp_array(data)
    if grasps.ndim == GRASP_OBJECT_BATCH_NDIM and grasps.shape[0] == 1:
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
    require_path(output_path, "output_path")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        np.save(output_path, _numpy_pickle_payload(grasps_by_object), allow_pickle=True)
        logger.info("Saved generated grasps to: {}", output_path)
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
    require_path(output_path, "output_path")
    if grasp_poses.ndim != GRASP_POSES_NDIM or grasp_poses.shape[1:] != SE3_MATRIX_SHAPE:
        raise ValueError(f"grasp_poses must have shape (K, 4, 4), got {grasp_poses.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, grasp_poses)
