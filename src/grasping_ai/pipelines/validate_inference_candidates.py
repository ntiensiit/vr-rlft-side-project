"""Physically validate raw inference grasp candidates in MuJoCo."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from grasping_ai.data.pointcloud_dataset import (
    PHYSICAL_VALIDATION_VERSION,
    GraspSample,
    save_grasp_sample,
)
from grasping_ai.pipelines.prepare_synthetic_data import sim_validation_passes
from grasping_ai.pipelines.simulate_grasp import simulate_grasp
from grasping_ai.simulation.scene import (
    MuJoCoScene,
    place_freejoint_body_on_surface,
    set_freejoint_body_pose,
)
from grasping_ai.simulation.ycb import find_ycb_mjcf, resolve_ycb_object_directory

POSE_NDIM = 3
POINT_CLOUD_NDIM = 2
SE3_SHAPE = (4, 4)
SPATIAL_SHAPE = (3,)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def _load_object_frame_candidates(candidate_path: Path) -> np.ndarray:
    """Load and validate raw object-frame SE(3) candidates from a NPY file."""
    if candidate_path.suffix.lower() != ".npy":
        raise ValueError("candidate_path must be a raw .npy grasp-candidate file")
    if not candidate_path.is_file():
        raise FileNotFoundError(f"candidate file not found: {candidate_path}")
    candidates = np.asarray(np.load(candidate_path, allow_pickle=False), dtype=np.float64)
    if candidates.ndim != POSE_NDIM or candidates.shape[1:] != SE3_SHAPE or candidates.shape[0] == 0:
        raise ValueError("candidate file must contain one or more poses with shape (K, 4, 4)")
    if not np.isfinite(candidates).all():
        raise ValueError("candidate poses must contain only finite values")
    if not np.allclose(candidates[:, 3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise ValueError("candidate poses must use homogeneous last row [0, 0, 0, 1]")
    rotations = candidates[:, :3, :3]
    orthogonality = rotations.transpose(0, 2, 1) @ rotations
    if not np.allclose(orthogonality, np.eye(3), atol=1e-5):
        raise ValueError("candidate pose rotations must be orthonormal")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-5):
        raise ValueError("candidate pose rotations must have determinant +1")
    return candidates


def _placed_object_pose(
    *,
    object_id: str,
    ycb_root: Path,
    robot_xml: Path,
    table_xml: Path | None,
    object_position: np.ndarray,
) -> np.ndarray:
    """Return the assembled scene's object-to-world transform after placement."""
    object_xml = find_ycb_mjcf(resolve_ycb_object_directory(ycb_root, object_id))
    scene = MuJoCoScene(robot_xml, object_xml, table_xml, object_name=object_id)
    set_freejoint_body_pose(scene.model, scene.data, object_id, object_position)
    if table_xml is not None:
        place_freejoint_body_on_surface(scene.model, scene.data, object_id)
    return scene.body_pose(object_id)


def _failure_reason(outcome: Mapping[str, object], *, passed: bool) -> str:
    """Return an auditable first failure reason for one physical validation."""
    if passed:
        return ""
    checks = (
        ("ik_converged", "IK did not converge"),
        ("lift_ik_converged", "lift IK did not converge"),
        ("initial_robot_object_collision_free", "robot intersects object before grasp"),
        ("table_collision_free", "robot intersects table"),
        ("bilateral_contact", "both fingertip pads did not contact object"),
        ("contact_sustained", "gripper contact was not sustained during lift"),
        ("stable", "object was not stable after lift"),
        ("success", "object did not complete physical lift"),
    )
    for key, reason in checks:
        if not bool(outcome.get(key, False)):
            return reason
    return "did not meet configured physical-validation thresholds"


def validate_inference_candidates(  # noqa: PLR0913
    *,
    candidate_path: Path,
    observation_path: Path,
    output_path: Path,
    object_id: str,
    ycb_root: Path,
    robot_xml: Path,
    table_xml: Path | None,
    object_position: np.ndarray,
    num_simulation_steps: int,
    gripper_close_command: np.ndarray,
    gripper_width: float,
    lift_height_threshold: float,
    lift_distance: float,
    max_linear_velocity: float,
    max_angular_velocity: float,
    min_contacts: float,
    require_ik: bool = True,
    require_lift: bool = True,
) -> GraspSample:
    """Validate exact raw candidates and save their per-candidate outcomes as NPZ.

    Raw candidates remain in the object frame.  The conversion to the assembled
    MuJoCo scene frame is performed exactly once, after the object's final
    table-surface placement is known.
    """
    if not object_id:
        raise ValueError("object_id is required")
    if observation_path.suffix.lower() != ".npy" or not observation_path.is_file():
        raise FileNotFoundError(f"observation file not found: {observation_path}")
    point_cloud = np.asarray(np.load(observation_path, allow_pickle=False), dtype=np.float32)
    if (
        point_cloud.ndim != POINT_CLOUD_NDIM
        or point_cloud.shape[1:] != SPATIAL_SHAPE
        or not np.isfinite(point_cloud).all()
    ):
        raise ValueError("observation must contain a finite point cloud with shape (N, 3)")
    object_position = np.asarray(object_position, dtype=np.float64)
    if object_position.shape != (3,) or not np.isfinite(object_position).all():
        raise ValueError("object_position must be finite with shape (3,)")

    candidates = _load_object_frame_candidates(candidate_path)
    object_to_world = _placed_object_pose(
        object_id=object_id,
        ycb_root=ycb_root,
        robot_xml=robot_xml,
        table_xml=table_xml,
        object_position=object_position,
    )

    outcomes: list[Mapping[str, object]] = []
    validated: list[bool] = []
    for object_pose in candidates:
        outcome = cast(
            "Mapping[str, object]",
            simulate_grasp(
                object_to_world @ object_pose,
                object_id,
                ycb_root,
                robot_xml,
                table_xml,
                num_simulation_steps,
                gripper_close_command,
                lift_height_threshold=lift_height_threshold,
                lift_distance=lift_distance,
                max_linear_velocity=max_linear_velocity,
                max_angular_velocity=max_angular_velocity,
                grasp_width=gripper_width,
                object_position=object_to_world[:3, 3],
                quiet=True,
            ),
        )
        outcomes.append(outcome)
        validated.append(
            sim_validation_passes(
                outcome,
                sim_validate_require_ik=require_ik,
                sim_validate_require_lift=require_lift,
                sim_validate_min_contacts=min_contacts,
                lift_height_threshold=lift_height_threshold,
            ),
        )

    def values(*, key: str, default: object, dtype: object) -> np.ndarray:
        return np.asarray([outcome.get(key, default) for outcome in outcomes], dtype=dtype)

    validation = np.asarray(validated, dtype=np.bool_)
    sample: GraspSample = {
        "point_cloud": point_cloud,
        "grasp_poses": candidates.astype(np.float32),
        "scores": np.zeros(candidates.shape[0], dtype=np.float32),
        "object_id": object_id,
        "grasp_pose_format": "object",
        "validation_version": PHYSICAL_VALIDATION_VERSION,
        "validation_lift_distance": float(lift_distance),
        "sim_validated": validation,
        "ik_converged": values(key="ik_converged", default=False, dtype=np.bool_),
        "contact_counts": values(key="contact_count", default=0.0, dtype=np.float32),
        "bilateral_contacts": values(key="bilateral_contact", default=False, dtype=np.bool_),
        "fk_position_errors": values(key="fk_position_error", default=float("inf"), dtype=np.float32),
        "table_collision_free": values(key="table_collision_free", default=False, dtype=np.bool_),
        "lift_ik_converged": values(key="lift_ik_converged", default=False, dtype=np.bool_),
        "lift_height_gains": np.asarray(
            [
                float(outcome.get("final_height", 0.0)) - float(outcome.get("initial_height", 0.0))
                for outcome in outcomes
            ],
            dtype=np.float32,
        ),
        "stable": values(key="stable", default=False, dtype=np.bool_),
        "contact_sustained": values(key="contact_sustained", default=False, dtype=np.bool_),
        "initial_robot_object_collision_free": values(
            key="initial_robot_object_collision_free",
            default=False,
            dtype=np.bool_,
        ),
        "candidate_indices": np.arange(candidates.shape[0], dtype=np.int32),
        "validation_failure_reasons": np.asarray(
            [_failure_reason(outcome, passed=passed) for outcome, passed in zip(outcomes, validated, strict=True)],
            dtype=np.str_,
        ),
    }
    save_grasp_sample(output_path, sample)
    return sample
