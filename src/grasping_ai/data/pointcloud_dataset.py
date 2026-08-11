from collections.abc import Iterator
from pathlib import Path

import numpy as np

type GraspSample = dict[str, np.ndarray | str | None]


def discover_dataset_files(dataset_root: Path) -> list[Path]:
    """List dataset record files under a dataset root directory.

    Args:
        dataset_root: Root directory containing processed dataset records.

    Returns:
        A sorted list of file paths representing individual dataset records.
    """
    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be a pathlib.Path instance")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root directory '{dataset_root}' does not exist")
    if not dataset_root.is_dir():
        raise ValueError(f"Dataset root '{dataset_root}' is not a directory")

    records = sorted([p for p in dataset_root.rglob("*.npy") if p.is_file()])
    if not records:
        raise ValueError(f"No dataset record files (.npy) found under '{dataset_root}'")
    return records


def load_grasp_sample(record_path: Path) -> GraspSample:
    """Load a single grasp-pose dataset record from disk.

    Args:
        record_path: Path to a serialized record file.

    Returns:
        A ``GraspSample`` containing a point cloud, grasp poses, optional scores,
        and an object identifier.

    Raises:
        FileNotFoundError: If ``record_path`` does not exist.
    """
    if not isinstance(record_path, Path):
        raise TypeError("record_path must be a pathlib.Path instance")
    if not record_path.exists():
        raise FileNotFoundError(f"Record file '{record_path}' not found")
    if not record_path.is_file():
        raise ValueError(f"Record path '{record_path}' is not a file")

    try:
        data = np.load(record_path, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"Failed to load record file: {e}") from e

    if data.ndim == 0:
        sample = data.item()
    else:
        raise ValueError("Invalid record file format: expected a serialized dictionary")

    if not isinstance(sample, dict):
        raise TypeError("Serialized record is not a dictionary")

    if "point_cloud" not in sample:
        raise ValueError("Record is missing 'point_cloud' key")

    pc = sample["point_cloud"]
    if not isinstance(pc, np.ndarray):
        raise TypeError("'point_cloud' must be a numpy array")
    if pc.ndim != 2 or pc.shape[1] != 3:
        raise ValueError(f"point_cloud must have shape (N, 3), got {pc.shape}")
    if not np.isfinite(pc).all():
        raise ValueError("point_cloud must contain only finite values")

    return {
        "point_cloud": pc,
        "grasp_poses": sample.get("grasp_poses"),
        "scores": sample.get("scores"),
        "object_id": sample.get("object_id"),
    }


def iterate_grasp_dataset(dataset_root: Path) -> Iterator[GraspSample]:
    """Iterate over all grasp-pose samples in a dataset directory.

    Args:
        dataset_root: Root directory containing processed dataset records.

    Yields:
        ``GraspSample`` records loaded one at a time.
    """
    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be a pathlib.Path instance")

    records = discover_dataset_files(dataset_root)
    for record in records:
        yield load_grasp_sample(record)


def resolve_ycb_object_id(ycb_root: Path, object_name: str) -> Path:
    """Resolve the on-disk path of a YCB object mesh file.

    Args:
        ycb_root: Root directory of the YCB object set.
        object_name: Logical YCB object identifier such as ``"mustard_bottle"``.

    Returns:
        Path to the YCB object resource directory or mesh file.
    """
    if not isinstance(ycb_root, Path):
        raise TypeError("ycb_root must be a pathlib.Path instance")
    if not isinstance(object_name, str):
        raise TypeError("object_name must be a string")
    if not ycb_root.exists():
        raise FileNotFoundError(f"YCB root directory '{ycb_root}' does not exist")
    if not ycb_root.is_dir():
        raise ValueError(f"YCB root '{ycb_root}' is not a directory")

    target_dir = ycb_root / object_name
    if not target_dir.is_dir():
        found_dir = None
        for p in ycb_root.iterdir():
            if p.is_dir() and object_name in p.name:
                found_dir = p
                break
        if found_dir is None:
            raise FileNotFoundError(f"YCB object '{object_name}' not found under '{ycb_root}'")
        target_dir = found_dir

    mesh_path = None
    for p in target_dir.rglob("textured.obj"):
        if p.is_file():
            mesh_path = p
            break
    if mesh_path is None:
        for p in target_dir.rglob("*.ply"):
            if p.is_file():
                mesh_path = p
                break
    if mesh_path is None:
        for p in target_dir.rglob("*.obj"):
            if p.is_file():
                mesh_path = p
                break

    return mesh_path if mesh_path is not None else target_dir
