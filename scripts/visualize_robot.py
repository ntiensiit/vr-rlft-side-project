"""Visualize robot in MuJoCo."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
from loguru import logger

from grasping_ai.config import FLATTENED_YAML_CONFIG, SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.data.pointcloud_dataset import PHYSICAL_VALIDATION_VERSION, load_grasp_sample
from grasping_ai.pipelines.visualize_robot import (
    apply_grasp_pose,
    build_contact_lift_trajectory,
    convert_grasp_pose_to_world,
    load_visualization_scene,
    run_robot_viewer,
)

ROBOT_XML_PATH = Path(str(FLATTENED_YAML_CONFIG.get("script.robot_xml", "deploy/robot.xml")))
YCB_ROOT = Path(str(FLATTENED_YAML_CONFIG.get("script.ycb_root", "data/raw/ycb")))
OBJECT_ID = FLATTENED_YAML_CONFIG.get("script.object_id")
GRASP_POSES_NDIM = int(FLATTENED_YAML_CONFIG.get("script.poses_ndim", 3))

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _apply_grasp_selection(  # noqa: PLR0913,PLR0917
    mj_model: object,
    mj_data: object,
    robot_xml_path: Path,
    grasp_poses: np.ndarray,
    object_id: object,
    pose_format: str,
    grasp_index: int,
    *,
    auto_select_reachable: bool,
    allow_ik_failure: bool,
    close_gripper: bool,
    candidate_validity: np.ndarray,
    lift_height: float | None,
    gripper_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int | None]:
    """Select a physically validated candidate that can complete the requested lift."""
    candidate_indices = (
        np.flatnonzero(candidate_validity).tolist() if auto_select_reachable else [grasp_index]
    )
    failures: list[str] = []
    for candidate_index in candidate_indices:
        try:
            grasp_pose = convert_grasp_pose_to_world(
                mj_model,
                mj_data,
                grasp_poses[candidate_index],
                object_id=str(object_id) if object_id is not None else None,
                pose_format=pose_format,
            )
            q_target = apply_grasp_pose(
                mj_model,
                mj_data,
                robot_xml_path,
                grasp_pose,
                allow_ik_failure=False if auto_select_reachable else allow_ik_failure,
                close_gripper=close_gripper,
            )
            lift_trajectory = None
            lift_start_index = None
            if lift_height is not None:
                lift_trajectory, lift_start_index = build_contact_lift_trajectory(
                    robot_xml_path,
                    grasp_pose,
                    q_target,
                    lift_height,
                    gripper_width=gripper_width,
                )
            logger.info("Selected collision-free grasp candidate {}", candidate_index)
            return grasp_pose, q_target, lift_trajectory, lift_start_index  # noqa: TRY300
        except ValueError as exc:
            if not auto_select_reachable:
                raise
            failures.append(f"{candidate_index}: {exc}")
    msg = f"No reachable grasp candidates found: {'; '.join(failures)}"
    raise ValueError(msg)


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/visualize_robot")
def main(cfg: DictConfig) -> None:  # noqa: C901, PLR0912, PLR0915
    """Open the MuJoCo viewer for the configured robot scene."""
    yaml_config = FlattenedYAMLConfig(cfg)
    robot_xml_path = yaml_config.value(
        "robot_xml", "robot", "description", value_type=Path, script_or=True, default=ROBOT_XML_PATH,
    )
    object_id = yaml_config.value(
        "object_id", "script", "object_id", value_type=object, default=OBJECT_ID, script_or=True,
    )
    table_xml = yaml_config.value("table_xml", "env", "table_xml", value_type=Path, script_or=True)
    grasp_file = yaml_config.value(
        "grasp_file", "script", "grasp_file", value_type=Path, script_or=True, default=None,
    )
    grasp_poses: np.ndarray | None = None
    grasp_validation: np.ndarray | None = None
    archive_pose_format: str | None = None
    validation_lift_distance: float | None = None
    selected_grasp_pose: np.ndarray | None = None
    q_target: np.ndarray | None = None
    lift_trajectory: np.ndarray | None = None
    lift_start_index: int | None = None
    if grasp_file is not None:
        if not grasp_file.is_file():
            raise FileNotFoundError(f"grasp_file not found: {grasp_file}")
        if grasp_file.suffix == ".npz":
            sample = load_grasp_sample(grasp_file)
            grasp_poses = sample["grasp_poses"]
            validation_version = sample.get("validation_version")
            if validation_version != PHYSICAL_VALIDATION_VERSION:
                raise ValueError(
                    f"Grasp archive '{grasp_file}' uses validation version {validation_version!r}; "
                    f"expected {PHYSICAL_VALIDATION_VERSION!r}. Regenerate it with prepare_data.py",
                )
            validation_lift_distance = sample.get("validation_lift_distance")
            if validation_lift_distance is None or validation_lift_distance <= 0:
                raise ValueError("Grasp archive is missing its physical lift validation distance")
            grasp_validation = np.asarray(sample.get("sim_validated", []), dtype=np.bool_)
            validation_fields = (
                "ik_converged",
                "lift_ik_converged",
                "bilateral_contacts",
                "table_collision_free",
                "lift_height_gains",
                "stable",
                "contact_sustained",
                "initial_robot_object_collision_free",
            )
            for field in validation_fields:
                values = np.asarray(sample.get(field, []))  # type: ignore[literal-required]
                if grasp_poses is not None and values.shape != (grasp_poses.shape[0],):
                    raise ValueError(
                        f"Grasp archive validation field '{field}' does not match its candidate count",
                    )
            archive_pose_format = sample.get("grasp_pose_format")
            sample_object_id = sample.get("object_id")
            if object_id is not None and sample_object_id is not None and str(object_id) != str(sample_object_id):
                raise ValueError(
                    f"Grasp archive object_id '{sample_object_id}' does not match requested object_id '{object_id}'",
                )
            if object_id is None:
                object_id = sample_object_id
        else:
            grasp_poses = np.load(grasp_file, allow_pickle=False)
    mj_model, mj_data = load_visualization_scene(
        robot_xml_path,
        object_id=object_id,
        ycb_root=yaml_config.value("ycb_root", "paths", "ycb_root", value_type=Path, script_or=True, default=YCB_ROOT),
        table_xml_path=Path(table_xml) if table_xml is not None else None,
        object_position=np.asarray(
            yaml_config.value(
                "object_position",
                "synthetic",
                "sim_object_position",
                value_type=list[float],
                script_or=True,
            ),
            dtype=np.float64,
        ),
    )
    lift_object = yaml_config.value(
        "lift_object", "script", "lift_object", value_type=bool, script_or=True, default=False,
    )
    lift_height = yaml_config.value(
        "lift_height", "script", "lift_height", value_type=float, script_or=True, default=0.1,
    )
    if lift_object and validation_lift_distance is not None and lift_height > validation_lift_distance + 1e-6:
        raise ValueError(
            f"Requested lift_height={lift_height} exceeds the archive's physically validated "
            f"distance {validation_lift_distance}",
        )
    gripper_width = yaml_config.value(
        "gripper_width", "script", "gripper_width", value_type=float, script_or=True, default=0.072,
    )
    close_gripper = yaml_config.value(
        "close_gripper", "script", "close_gripper", value_type=bool, script_or=True, default=True,
    )
    if lift_object and not close_gripper:
        raise ValueError("script.lift_object=true requires script.close_gripper=true for physical contact")
    if grasp_poses is not None:
        grasp_index = yaml_config.value(
            "grasp_index", "script", "grasp_index", value_type=int, script_or=True, default=0,
        )
        if grasp_poses.ndim != GRASP_POSES_NDIM or grasp_poses.shape[1:] != (4, 4):
            msg = f"grasp_file must contain poses with shape (K, 4, 4), got {grasp_poses.shape}"
            raise ValueError(msg)
        if not 0 <= grasp_index < grasp_poses.shape[0]:
            msg = f"grasp_index {grasp_index} is outside [0, {grasp_poses.shape[0]})"
            raise IndexError(msg)
        if grasp_validation is None or grasp_validation.shape != (grasp_poses.shape[0],):
            raise ValueError("Grasp archive validation metadata does not match its candidate count")
        if not bool(np.any(grasp_validation)):
            raise ValueError("Grasp archive contains no physically validated candidates")
        if not yaml_config.value(
            "auto_select_reachable", "script", "auto_select_reachable", value_type=bool, script_or=True, default=False,
        ) and not bool(grasp_validation[grasp_index]):
            raise ValueError(f"Grasp candidate {grasp_index} did not pass physical simulation validation")
        grasp_pose_format = str(
            yaml_config.value(
                "grasp_pose_format",
                "script",
                "grasp_pose_format",
                value_type=object,
                script_or=True,
                default=archive_pose_format or "object",
            ),
        )
        selected_grasp_pose, q_target, lift_trajectory, lift_start_index = _apply_grasp_selection(
            mj_model,
            mj_data,
            robot_xml_path,
            grasp_poses,
            object_id,
            grasp_pose_format,
            grasp_index,
            auto_select_reachable=yaml_config.value(
                "auto_select_reachable",
                "script",
                "auto_select_reachable",
                value_type=bool,
                script_or=True,
                default=False,
            ),
            allow_ik_failure=yaml_config.value(
                "allow_ik_failure", "script", "allow_ik_failure", value_type=bool, script_or=True, default=True,
            ),
            close_gripper=close_gripper,
            candidate_validity=grasp_validation,
            lift_height=lift_height if lift_object else None,
            gripper_width=gripper_width,
        )
    animation_duration = yaml_config.value(
        "animation_duration", "script", "animation_duration", value_type=float, script_or=True, default=1.0,
    )
    if lift_object and (selected_grasp_pose is None or q_target is None):
        raise ValueError("script.lift_object=true requires script.grasp_file")
    run_robot_viewer(
        mj_model,
        mj_data,
        lift_trajectory=lift_trajectory,
        object_id=str(object_id) if object_id is not None else None,
        trajectory_duration_s=animation_duration,
        lift_start_index=lift_start_index,
        require_gripper_contact=lift_object,
    )

if __name__ == "__main__":
    main()
