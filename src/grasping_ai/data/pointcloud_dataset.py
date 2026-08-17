"""Point-cloud grasp dataset loading and NPZ I/O."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast

import numpy as np
from loguru import logger

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
from grasping_ai.perception.geometry import make_transform
from grasping_ai.perception.pointcloud import build_kdtree
from grasping_ai.simulation.ycb import (
    find_ycb_mesh_file,
    resolve_ycb_object_directory,
)
from grasping_ai.utils.path_validation import require_path

ALIGNMENT_DOT_THRESHOLD = float(FLATTENED_YAML_CONFIG.get("tolerances.alignment_dot_threshold", 0.9))
GRASP_DISTANCE_EPS = float(FLATTENED_YAML_CONFIG.get("tolerances.grasp_distance_eps", 1e-4))
POINT_CLOUD_NDIM = int(FLATTENED_YAML_CONFIG.get("geometry.point_cloud_ndim", 2))
ROTATION_DET_EPS = float(FLATTENED_YAML_CONFIG.get("tolerances.rotation_det_eps", 1e-4))
SPATIAL_DIM = int(FLATTENED_YAML_CONFIG.get("geometry.spatial_dim", 3))
ALLOW_RELAXED = bool(FLATTENED_YAML_CONFIG.get("synthetic.allow_relaxed", False))
RELAXED_ANTIPODAL_DOT = float(FLATTENED_YAML_CONFIG.get("synthetic.relaxed_antipodal_dot", 0.3))
STRICT_ANTIPODAL_DOT = float(FLATTENED_YAML_CONFIG.get("synthetic.strict_antipodal_dot", 0.5))
STRICT_ALIGNMENT_DOT = float(FLATTENED_YAML_CONFIG.get("synthetic.strict_alignment_dot", 0.5))
SEARCH_MULTIPLIER = int(FLATTENED_YAML_CONFIG.get("synthetic.search_multiplier", 50))
MAX_VERTICAL_CLOSING_AXIS_COMPONENT = 0.35
MAX_OBJECT_EXTENT_TOWARD_HAND = float(
    FLATTENED_YAML_CONFIG.get("robot.gripper.max_object_extent_toward_hand", 0.04),
)
PHYSICAL_VALIDATION_VERSION = "physical-lift-v4"

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class GraspSample(TypedDict):
    """Single grasp-dataset record loaded from disk."""

    point_cloud: np.ndarray
    grasp_poses: np.ndarray | None
    scores: np.ndarray | None
    object_id: str | None
    grasp_pose_format: NotRequired[str]
    validation_version: NotRequired[str]
    validation_lift_distance: NotRequired[float]
    sim_validated: NotRequired[np.ndarray]
    ik_converged: NotRequired[np.ndarray]
    contact_counts: NotRequired[np.ndarray]
    bilateral_contacts: NotRequired[np.ndarray]
    fk_position_errors: NotRequired[np.ndarray]
    table_collision_free: NotRequired[np.ndarray]
    lift_ik_converged: NotRequired[np.ndarray]
    lift_height_gains: NotRequired[np.ndarray]
    stable: NotRequired[np.ndarray]
    contact_sustained: NotRequired[np.ndarray]
    initial_robot_object_collision_free: NotRequired[np.ndarray]


def _validate_point_cloud(point_cloud: object) -> np.ndarray:
    """Validate a loaded point cloud array.

    Args:
        point_cloud: Deserialized ``point_cloud`` payload from a dataset record.

    Returns:
        The validated point cloud array.

    Raises:
        TypeError: If ``point_cloud`` is not a ``numpy.ndarray``.
        ValueError: If the array shape or values are invalid.
    """
    if not isinstance(point_cloud, np.ndarray):
        msg = "'point_cloud' must be a numpy array"
        raise TypeError(msg)
    if point_cloud.ndim != POINT_CLOUD_NDIM or point_cloud.shape[1] != SPATIAL_DIM:
        msg = f"point_cloud must have shape (N, 3), got {point_cloud.shape}"
        raise ValueError(msg)
    if not np.isfinite(point_cloud).all():
        msg = "point_cloud must contain only finite values"
        raise ValueError(msg)
    return point_cloud


def _decode_object_id(object_id: object) -> str | None:
    """Decode an optional object identifier stored in an NPZ archive.

    Args:
        object_id: Deserialized ``object_id`` payload from a dataset record.

    Returns:
        The object identifier string, or ``None`` when absent.
    """
    if object_id is None:
        return None
    if isinstance(object_id, np.ndarray):
        if object_id.shape == ():
            return str(object_id.item())
        if object_id.ndim == 1 and object_id.dtype.kind in {"U", "S"}:
            return str(object_id.item())
    if isinstance(object_id, str):
        return object_id
    return str(object_id)


def save_grasp_sample(record_path: Path, sample: GraspSample) -> None:
    """Persist a grasp-pose dataset record as a pickle-free NPZ archive.

    Args:
        record_path: Destination ``.npz`` path for the serialized record.
        sample: Dataset record containing point-cloud and grasp metadata.

    Raises:
        TypeError: If ``point_cloud`` is not a ``numpy.ndarray``.
        ValueError: If ``point_cloud`` has an invalid shape or non-finite values.
    """
    require_path(record_path, "record_path")
    if record_path.suffix != ".npz":
        msg = f"Record path '{record_path}' must use the .npz extension"
        raise ValueError(msg)

    point_cloud = _validate_point_cloud(sample["point_cloud"])

    archive_fields: dict[str, np.ndarray] = {
        "point_cloud": np.asarray(point_cloud, dtype=np.float32),
    }

    grasp_poses = sample.get("grasp_poses")
    if grasp_poses is not None:
        archive_fields["grasp_poses"] = np.asarray(grasp_poses, dtype=np.float32)

    scores = sample.get("scores")
    if scores is not None:
        archive_fields["scores"] = np.asarray(scores, dtype=np.float32)

    object_id = sample.get("object_id")
    if object_id is not None:
        archive_fields["object_id"] = np.asarray(object_id, dtype=np.str_)

    for key in ("grasp_pose_format", "validation_version"):
        value = sample.get(key)  # type: ignore[literal-required]
        if value is not None:
            archive_fields[key] = np.asarray(value, dtype=np.str_)
    validation_lift_distance = sample.get("validation_lift_distance")
    if validation_lift_distance is not None:
        archive_fields["validation_lift_distance"] = np.asarray(
            validation_lift_distance,
            dtype=np.float32,
        )
    for key in (
        "sim_validated",
        "ik_converged",
        "contact_counts",
        "bilateral_contacts",
        "fk_position_errors",
        "table_collision_free",
        "lift_ik_converged",
        "lift_height_gains",
        "stable",
        "contact_sustained",
        "initial_robot_object_collision_free",
    ):
        value = sample.get(key)  # type: ignore[literal-required]
        if value is not None:
            archive_fields[key] = np.asarray(value)

    record_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(record_path, **cast("Any", archive_fields))


def discover_dataset_files(dataset_root: Path) -> list[Path]:
    """List dataset record files under a dataset root directory.

    Args:
        dataset_root: Root directory containing processed dataset records.

    Returns:
        A sorted list of file paths representing individual dataset records.
    """
    require_path(dataset_root, "dataset_root")
    if not dataset_root.exists():
        msg = f"Dataset root directory '{dataset_root}' does not exist"
        raise FileNotFoundError(msg)
    if not dataset_root.is_dir():
        msg = f"Dataset root '{dataset_root}' is not a directory"
        raise ValueError(msg)

    records = sorted([p for p in dataset_root.rglob("*.npz") if p.is_file()])
    if not records:
        msg = f"No dataset record files (.npz) found under '{dataset_root}'"
        raise ValueError(msg)
    logger.info("Discovered {} dataset record files under {}", len(records), dataset_root)
    return records


def _read_grasp_sample_archive(archive: np.lib.npyio.NpzFile) -> GraspSample:
    """Parse a loaded NPZ archive into a ``GraspSample``.

    Args:
        archive: Open NPZ archive containing dataset record arrays.

    Returns:
        Parsed grasp dataset record.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    if "point_cloud" not in archive:
        msg = "Record is missing 'point_cloud' key"
        raise ValueError(msg)

    point_cloud = _validate_point_cloud(archive["point_cloud"])
    grasp_poses = archive["grasp_poses"] if "grasp_poses" in archive.files else None
    scores = archive["scores"] if "scores" in archive.files else None
    object_id = _decode_object_id(archive["object_id"]) if "object_id" in archive.files else None
    sample: GraspSample = {
        "point_cloud": point_cloud,
        "grasp_poses": grasp_poses,
        "scores": scores,
        "object_id": object_id,
    }
    for key in ("grasp_pose_format", "validation_version"):
        if key in archive.files:
            sample[key] = str(archive[key].item())  # type: ignore[literal-required]
    if "validation_lift_distance" in archive.files:
        sample["validation_lift_distance"] = float(archive["validation_lift_distance"].item())
    for key in (
        "sim_validated",
        "ik_converged",
        "contact_counts",
        "bilateral_contacts",
        "fk_position_errors",
        "table_collision_free",
        "lift_ik_converged",
        "lift_height_gains",
        "stable",
        "contact_sustained",
        "initial_robot_object_collision_free",
    ):
        if key in archive.files:
            sample[key] = archive[key]  # type: ignore[literal-required]
    return sample


def load_grasp_sample(record_path: Path) -> GraspSample:
    """Load a single grasp-pose dataset record from disk.

    Args:
        record_path: Path to a serialized ``.npz`` record file.

    Returns:
        A ``GraspSample`` containing a point cloud, grasp poses, optional scores,
        and an object identifier.

    Raises:
        FileNotFoundError: If ``record_path`` does not exist.
        ValueError: If the record format is invalid or missing required fields.
    """
    require_path(record_path, "record_path")
    if not record_path.exists():
        msg = f"Record file '{record_path}' not found"
        raise FileNotFoundError(msg)
    if not record_path.is_file():
        msg = f"Record path '{record_path}' is not a file"
        raise ValueError(msg)
    if record_path.suffix != ".npz":
        msg = f"Record path '{record_path}' must use the .npz extension"
        raise ValueError(msg)

    try:
        with np.load(record_path) as archive:
            return _read_grasp_sample_archive(archive)
    except ValueError:
        raise
    except Exception as e:
        msg = f"Failed to load record file: {e}"
        raise ValueError(msg) from e


def iterate_grasp_dataset(dataset_root: Path) -> Iterator[GraspSample]:
    """Iterate over all grasp-pose samples in a dataset directory.

    Args:
        dataset_root: Root directory containing processed dataset records.

    Yields:
        ``GraspSample`` records loaded one at a time.
    """
    require_path(dataset_root, "dataset_root")

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
    require_path(ycb_root, "ycb_root")
    if not isinstance(object_name, str):
        msg = "object_name must be a string"
        raise TypeError(msg)
    if not ycb_root.exists():
        msg = f"YCB root directory '{ycb_root}' does not exist"
        raise FileNotFoundError(msg)
    if not ycb_root.is_dir():
        msg = f"YCB root '{ycb_root}' is not a directory"
        raise ValueError(msg)

    object_dir = resolve_ycb_object_directory(ycb_root, object_name)
    try:
        return find_ycb_mesh_file(object_dir)
    except FileNotFoundError:
        return object_dir


@dataclass(frozen=True)
class _ContactPair:
    """Antipodal contact candidate: contact positions and outward normals."""

    p_i: np.ndarray
    n_i: np.ndarray
    p_j: np.ndarray
    n_j: np.ndarray


@dataclass(frozen=True)
class _AntipodalSearchConfig:
    """Tunable knobs for one randomized antipodal contact-pair search pass."""

    num_grasps: int
    gripper_width: float
    attempts: int
    antipodal_dot: float
    alignment_dot: float | None = None


def _antipodal_grasp_from_contacts(
    pair: _ContactPair,
    d_unit: np.ndarray,
    antipodal_dot: float,
    alignment_dot: float | None = None,
) -> np.ndarray | None:
    """Build a single antipodal grasp pose from a contact pair when constraints pass.

    Args:
        pair: Contact positions and outward normals of the contact pair.
        d_unit: Unit vector from the first to the second contact.
        antipodal_dot: Minimum cosine similarity between the two normals.
        alignment_dot: Optional minimum alignment between normals and ``d_unit``.

    Returns:
        A valid ``(4, 4)`` grasp transform, or ``None`` when constraints fail.
    """
    if np.dot(pair.n_i, -pair.n_j) <= antipodal_dot:
        return None
    if alignment_dot is not None and (
        np.dot(pair.n_i, -d_unit) <= alignment_dot or np.dot(pair.n_j, d_unit) <= alignment_dot
    ):
        return None

    # The Panda contact frame's x axis is its finger closing direction.
    # Align it with the contact pair so the gripper closes across the object,
    # rather than approaching along the line between the contacts.
    x_axis = d_unit
    # The contact frame's +z axis points from the Panda hand toward the
    # contact midpoint.  For objects supported by a horizontal work surface,
    # choose the downward hemisphere so the hand is above the object.  The
    # previous +Z reference inverted this convention and generated grasps
    # whose hand frame sat below the contacts, forcing links through the
    # table even though the antipodal contact pair itself was valid.
    reference = np.array([0.0, 0.0, -1.0])
    if np.abs(np.dot(reference, x_axis)) > ALIGNMENT_DOT_THRESHOLD:
        reference = np.array([0.0, 1.0, 0.0])
    z_axis = reference - np.dot(reference, x_axis) * x_axis
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    if z_axis[2] > 0.0:
        # The alternate reference used for nearly vertical closing axes can
        # project into either horizontal hemisphere.  Keep the convention
        # deterministic and never generate an upward-facing approach.
        y_axis = -y_axis
        z_axis = -z_axis

    pose = make_transform(
        np.column_stack([x_axis, y_axis, z_axis]),
        0.5 * (pair.p_i + pair.p_j),
    )
    det = np.linalg.det(pose[:3, :3])
    if np.abs(det - 1.0) >= ROTATION_DET_EPS:
        return None
    return pose


def _search_antipodal_grasps(
    points: np.ndarray,
    normals: np.ndarray,
    tree: object,
    rng: np.random.Generator,
    config: _AntipodalSearchConfig,
) -> list[np.ndarray]:
    """Collect antipodal grasps from randomized contact-pair search.

    Args:
        points: Point cloud of shape ``(N, 3)``.
        normals: Point normals of shape ``(N, 3)``.
        tree: KD-tree built over ``points``.
        rng: Random generator for sampling and shuffling.
        config: Search pass knobs (grasp count, width, attempts, thresholds).

    Returns:
        List of valid grasp transforms, each with shape ``(4, 4)``.
    """
    valid_grasps: list[np.ndarray] = []
    n = points.shape[0]

    for _ in range(config.attempts):
        if len(valid_grasps) >= config.num_grasps:
            break

        i = int(rng.choice(n))
        p_i = points[i]
        n_i = normals[i]

        neighbors = tree.query_ball_point(p_i, r=config.gripper_width)  # type: ignore[attr-defined]
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
            if dist < GRASP_DISTANCE_EPS:
                continue
            d_unit = d / dist
            if abs(float(d_unit[2])) > MAX_VERTICAL_CLOSING_AXIS_COMPONENT:
                # A top-down parallel-jaw pick needs side contacts around the
                # object's body.  Vertically stacked contact pairs instead
                # describe a pinch from underneath/above, not a top-down pick.
                continue

            pose = _antipodal_grasp_from_contacts(
                _ContactPair(p_i=p_i, n_i=n_i, p_j=p_j, n_j=n_j),
                d_unit,
                config.antipodal_dot,
                config.alignment_dot,
            )
            if pose is not None:
                # For a top-down Panda grasp, points extending too far from
                # the contact midpoint toward the hand collide with the palm
                # before the fingers can close. Filter by the actual gripper
                # clearance while continuing the search for upper contacts.
                toward_hand = -pose[:3, 2]
                extent_toward_hand = float(np.max((points - pose[:3, 3]) @ toward_hand))
                if extent_toward_hand > MAX_OBJECT_EXTENT_TOWARD_HAND:
                    continue
                valid_grasps.append(pose)
                break

    return valid_grasps


def _validate_cloud_and_rng(points: np.ndarray, normals: np.ndarray, rng: np.random.Generator) -> None:
    """Validate point/normal arrays and the random generator.

    Args:
        points: Point cloud expected to have shape ``(N, 3)``.
        normals: Point normals expected to match ``points`` in length.
        rng: Random generator, must be a ``numpy.random.Generator``.
    """
    if not isinstance(points, np.ndarray) or points.ndim != POINT_CLOUD_NDIM or points.shape[1] != SPATIAL_DIM:
        msg = "points must be of shape (N, 3)"
        raise ValueError(msg)
    if not isinstance(normals, np.ndarray) or normals.ndim != POINT_CLOUD_NDIM or normals.shape[1] != SPATIAL_DIM:
        msg = "normals must be of shape (N, 3)"
        raise ValueError(msg)
    if points.shape[0] != normals.shape[0]:
        msg = "points and normals must have the same length"
        raise ValueError(msg)
    if not isinstance(rng, np.random.Generator):
        msg = "rng must be a numpy random Generator"
        raise TypeError(msg)


# Public API: threshold knobs stay individual keyword arguments because
# pipelines and tests pass them by name.
def generate_analytical_grasps(  # noqa: PLR0913
    points: np.ndarray,
    normals: np.ndarray,
    num_grasps: int,
    gripper_width: float,
    rng: np.random.Generator,
    *,
    allow_relaxed: bool = ALLOW_RELAXED,
    relaxed_antipodal_dot: float = RELAXED_ANTIPODAL_DOT,
    strict_antipodal_dot: float = STRICT_ANTIPODAL_DOT,
    strict_alignment_dot: float = STRICT_ALIGNMENT_DOT,
    search_multiplier: int = SEARCH_MULTIPLIER,
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
    _validate_cloud_and_rng(points, normals, rng)
    if num_grasps <= 0:
        msg = "num_grasps must be positive"
        raise ValueError(msg)
    if not isinstance(allow_relaxed, bool):
        msg = "allow_relaxed must be a boolean"
        raise TypeError(msg)
    if not -1.0 <= relaxed_antipodal_dot <= 1.0:
        msg = "relaxed_antipodal_dot must be in [-1, 1]"
        raise ValueError(msg)
    if not -1.0 <= strict_antipodal_dot <= 1.0:
        msg = "strict_antipodal_dot must be in [-1, 1]"
        raise ValueError(msg)
    if not -1.0 <= strict_alignment_dot <= 1.0:
        msg = "strict_alignment_dot must be in [-1, 1]"
        raise ValueError(msg)
    if not isinstance(search_multiplier, int) or search_multiplier <= 0:
        msg = "search_multiplier must be a positive integer"
        raise ValueError(msg)

    tree = build_kdtree(points)
    search = _AntipodalSearchConfig(
        num_grasps=num_grasps,
        gripper_width=gripper_width,
        attempts=num_grasps * search_multiplier,
        antipodal_dot=strict_antipodal_dot,
        alignment_dot=strict_alignment_dot,
    )
    valid_grasps = _search_antipodal_grasps(points, normals, tree, rng, search)

    if not valid_grasps and allow_relaxed:
        relaxed_search = replace(search, antipodal_dot=relaxed_antipodal_dot, alignment_dot=None)
        valid_grasps = _search_antipodal_grasps(points, normals, tree, rng, relaxed_search)

    if not valid_grasps:
        return np.empty((0, 4, 4), dtype=np.float32)

    return np.stack(valid_grasps, axis=0).astype(np.float32)
