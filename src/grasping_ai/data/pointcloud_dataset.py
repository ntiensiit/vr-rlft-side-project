from collections.abc import Iterator
from pathlib import Path

import numpy as np

from grasping_ai.perception.geometry import make_transform
from grasping_ai.perception.pointcloud import build_kdtree

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

    from loguru import logger

    records = sorted([p for p in dataset_root.rglob("*.npy") if p.is_file()])
    if not records:
        raise ValueError(f"No dataset record files (.npy) found under '{dataset_root}'")
    logger.info("Discovered {} dataset record files under {}", len(records), dataset_root)
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

    from grasping_ai.simulation.ycb import find_ycb_mesh_file, resolve_ycb_object_directory

    object_dir = resolve_ycb_object_directory(ycb_root, object_name)
    try:
        return find_ycb_mesh_file(object_dir)
    except FileNotFoundError:
        return object_dir


def generate_analytical_grasps(
    points: np.ndarray,
    normals: np.ndarray,
    num_grasps: int,
    gripper_width: float,
    rng: np.random.Generator,
    allow_relaxed: bool = False,
    relaxed_antipodal_dot: float = 0.0,
    strict_antipodal_dot: float = 0.5,
    strict_alignment_dot: float = 0.5,
    search_multiplier: int = 20,
) -> np.ndarray:
    """Generate analytical antipodal grasps from a point cloud with normals.

    Data policy: by default (``allow_relaxed=False``) only geometrically
    constrained antipodal grasps are produced, so no unconstrained grasps can
    be saved. When the strict antipodal search yields no grasps and
    ``allow_relaxed=True``, a relaxed search is used that still requires the
    contact normals to face each other beyond ``relaxed_antipodal_dot``.

    Args:
        points: Point cloud of shape ``(N, 3)``.
        normals: Point cloud normals of shape ``(N, 3)``.
        num_grasps: Maximum number of grasps to generate.
        gripper_width: Maximum allowed distance between contact points.
        rng: Random generator.
        allow_relaxed: Whether to permit the relaxed antipodal fallback when
            the strict search produces no grasps.
        relaxed_antipodal_dot: Cosine threshold on ``dot(n_i, -n_j)`` used by
            the relaxed fallback. Values must lie in ``[-1, 1]``.
        strict_antipodal_dot: Cosine threshold on ``dot(n_i, -n_j)`` for the
            strict antipodal search.
        strict_alignment_dot: Cosine threshold on action-line alignment with
            contact normals during the strict search.
        search_multiplier: Number of random attempts per requested grasp.

    Returns:
        Array of grasp poses of shape ``(K, 4, 4)`` where K <= num_grasps.
    """
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be of shape (N, 3)")
    if not isinstance(normals, np.ndarray) or normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("normals must be of shape (N, 3)")
    if points.shape[0] != normals.shape[0]:
        raise ValueError("points and normals must have the same length")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy random Generator")
    if num_grasps <= 0:
        raise ValueError("num_grasps must be positive")
    if not isinstance(allow_relaxed, bool):
        raise TypeError("allow_relaxed must be a boolean")
    if not -1.0 <= relaxed_antipodal_dot <= 1.0:
        raise ValueError("relaxed_antipodal_dot must be in [-1, 1]")
    if not -1.0 <= strict_antipodal_dot <= 1.0:
        raise ValueError("strict_antipodal_dot must be in [-1, 1]")
    if not -1.0 <= strict_alignment_dot <= 1.0:
        raise ValueError("strict_alignment_dot must be in [-1, 1]")
    if not isinstance(search_multiplier, int) or search_multiplier <= 0:
        raise ValueError("search_multiplier must be a positive integer")

    tree = build_kdtree(points)
    valid_grasps: list[np.ndarray] = []
    n = points.shape[0]

    attempts = num_grasps * search_multiplier
    for _ in range(attempts):
        if len(valid_grasps) >= num_grasps:
            break

        i = int(rng.choice(n))
        p_i = points[i]
        n_i = normals[i]

        # Find neighbors within gripper_width
        neighbors = tree.query_ball_point(p_i, r=gripper_width)
        if not neighbors:
            continue

        # Shuffle neighbors using rng
        rng.shuffle(neighbors)

        for j in neighbors:
            if j == i:
                continue
            p_j = points[j]
            n_j = normals[j]

            d = p_j - p_i
            dist = np.linalg.norm(d)
            if dist < 1e-4:
                continue
            d = d / dist

            # Antipodal condition: normals face each other, and action line aligns with normals
            if (
                np.dot(n_i, -n_j) > strict_antipodal_dot
                and np.dot(n_i, d) > strict_alignment_dot
                and np.dot(n_j, -d) > strict_alignment_dot
            ):
                z_axis = d
                avg_normal = 0.5 * (n_i + n_j)
                x_axis = np.cross(z_axis, avg_normal)
                x_norm = np.linalg.norm(x_axis)
                if x_norm < 1e-4:
                    ref = np.array([1.0, 0.0, 0.0])
                    if np.abs(np.dot(ref, z_axis)) > 0.9:
                        ref = np.array([0.0, 1.0, 0.0])
                    x_axis = np.cross(z_axis, ref)
                    x_norm = np.linalg.norm(x_axis)
                x_axis = x_axis / x_norm
                y_axis = np.cross(z_axis, x_axis)
                y_axis = y_axis / np.linalg.norm(y_axis)

                pose = make_transform(
                    np.column_stack([x_axis, y_axis, z_axis]),
                    0.5 * (p_i + p_j),
                )

                # Ensure determinant is +1 within tolerance
                det = np.linalg.det(pose[:3, :3])
                if np.abs(det - 1.0) < 1e-4:
                    valid_grasps.append(pose)
                    break

    # Fallback if no valid grasps found: relax constraints only when explicitly
    # allowed. The relaxed search still requires normals to face each other
    # beyond relaxed_antipodal_dot so unconstrained grasps are never saved.
    if not valid_grasps and allow_relaxed:
        for _ in range(attempts):
            if len(valid_grasps) >= num_grasps:
                break

            i = int(rng.choice(n))
            p_i = points[i]
            n_i = normals[i]

            neighbors = tree.query_ball_point(p_i, r=gripper_width)
            if not neighbors:
                continue

            rng.shuffle(neighbors)
            for j in neighbors:
                if j == i:
                    continue
                p_j = points[j]
                n_j = normals[j]

                d = p_j - p_i
                dist = np.linalg.norm(d)
                if dist < 1e-4:
                    continue
                d = d / dist

                # Relaxed antipodal condition
                if np.dot(n_i, -n_j) > relaxed_antipodal_dot:
                    z_axis = d
                    avg_normal = 0.5 * (n_i + n_j)
                    x_axis = np.cross(z_axis, avg_normal)
                    x_norm = np.linalg.norm(x_axis)
                    if x_norm < 1e-4:
                        ref = np.array([1.0, 0.0, 0.0])
                        if np.abs(np.dot(ref, z_axis)) > 0.9:
                            ref = np.array([0.0, 1.0, 0.0])
                        x_axis = np.cross(z_axis, ref)
                        x_norm = np.linalg.norm(x_axis)
                    x_axis = x_axis / x_norm
                    y_axis = np.cross(z_axis, x_axis)
                    y_axis = y_axis / np.linalg.norm(y_axis)

                    pose = make_transform(
                        np.column_stack([x_axis, y_axis, z_axis]),
                        0.5 * (p_i + p_j),
                    )

                    det = np.linalg.det(pose[:3, :3])
                    if np.abs(det - 1.0) < 1e-4:
                        valid_grasps.append(pose)
                        break

    if not valid_grasps:
        return np.empty((0, 4, 4), dtype=np.float32)

    return np.stack(valid_grasps, axis=0).astype(np.float32)
