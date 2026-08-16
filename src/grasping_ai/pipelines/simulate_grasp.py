"""Simulate grasps in MuJoCo."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco  # type: ignore[import-untyped]
import numpy as np
from loguru import logger

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
from grasping_ai.evaluation.metrics import (
    build_lift_outcome_judge,
    build_stability_judge,
    evaluate_lift_success,
    evaluate_stability,
)
from grasping_ai.perception.geometry import invert_transform
from grasping_ai.robotics.gripper import (
    gripper_actuator_indices,
    load_gripper_model,
    make_close_command,
    make_open_command,
    panda_hand_to_contact_transform,
    panda_width_to_finger_joints,
)
from grasping_ai.robotics.kinematics import (
    build_forward_kinematics,
    build_inverse_kinematics,
    load_robot_model,
    robot_model_mj_model,
    robot_model_nq,
    solve_inverse_kinematics,
)
from grasping_ai.robotics.transforms import transform_grasp_pose
from grasping_ai.simulation.scene import (
    MuJoCoScene,
    collect_contacts,
    step_scene,
)
from grasping_ai.simulation.ycb import (
    find_ycb_mjcf,
    resolve_ycb_object_directory,
)

if TYPE_CHECKING:
    from grasping_ai.robotics.kinematics import ForwardKinematics

FALLBACK_TIMESTEP = float(FLATTENED_YAML_CONFIG.get("fallback_timestep"))
SE3_MATRIX_SHAPE = tuple(int(v) for v in FLATTENED_YAML_CONFIG.get("grasp.se3_matrix_shape", [4, 4]))
IK_MAX_ITERATIONS = int(FLATTENED_YAML_CONFIG.get("robot.ik.max_iterations"))
IK_TOLERANCE = float(FLATTENED_YAML_CONFIG.get("robot.ik.tolerance"))
GRIPPER_JOINT_RANGES = tuple(
    (name, tuple(FLATTENED_YAML_CONFIG.get_path("robot", "gripper", "joint_ranges", name)))
    for name in ("finger_joint1", "finger_joint2")
)
GRIPPER_DUAL_COUNT = int(FLATTENED_YAML_CONFIG.get("robot.gripper.dual_count", 2))
LIFT_HEIGHT_THRESHOLD = float(FLATTENED_YAML_CONFIG.get("metrics.lift_height_threshold"))
MAX_LINEAR_VELOCITY = float(FLATTENED_YAML_CONFIG.get("limits.max_linear_velocity"))
MAX_ANGULAR_VELOCITY = float(FLATTENED_YAML_CONFIG.get("limits.max_angular_velocity"))
PRE_GRASP_STEPS = int(FLATTENED_YAML_CONFIG.get("pre_grasp_steps"))
POINT_CLOUD_NDIM = int(FLATTENED_YAML_CONFIG.get("geometry.point_cloud_ndim", 2))
GRASP_POSES_NDIM = int(FLATTENED_YAML_CONFIG.get("grasp.poses_ndim", 3))

def _validate_simulate_grasp_args(  # noqa: PLR0913, PLR0917  # validation keeps related simulation inputs together
    grasp_pose: np.ndarray,
    robot_xml_path: Path,
    ycb_root: Path,
    num_simulation_steps: int,
    lift_height_threshold: float,
    max_linear_velocity: float,
    max_angular_velocity: float,
    grasp_width: float | None,
) -> None:
    """Validate grasp pose shape, asset paths, and simulation thresholds."""
    if grasp_pose.shape != SE3_MATRIX_SHAPE:
        msg = f"grasp_pose must have shape (4, 4), got {grasp_pose.shape}"
        raise ValueError(msg)
    if not isinstance(robot_xml_path, Path) or not robot_xml_path.is_file():
        msg = f"robot_xml_path not found: {robot_xml_path}"
        raise FileNotFoundError(msg)
    if not isinstance(ycb_root, Path) or not ycb_root.is_dir():
        msg = f"ycb_root not found: {ycb_root}"
        raise FileNotFoundError(msg)
    if num_simulation_steps <= 0:
        msg = "num_simulation_steps must be positive"
        raise ValueError(msg)
    if lift_height_threshold < 0:
        msg = "lift_height_threshold must be non-negative"
        raise ValueError(msg)
    if max_linear_velocity < 0:
        msg = "max_linear_velocity must be non-negative"
        raise ValueError(msg)
    if max_angular_velocity < 0:
        msg = "max_angular_velocity must be non-negative"
        raise ValueError(msg)
    if grasp_width is not None and grasp_width < 0:
        msg = "grasp_width must be non-negative when provided"
        raise ValueError(msg)


def _solve_grasp_ik(
    robot_xml_path: Path,
    hand_pose: np.ndarray,
    *,
    quiet: bool,
    max_iterations: int = IK_MAX_ITERATIONS,
    tolerance: float = IK_TOLERANCE,
) -> tuple[np.ndarray, bool, ForwardKinematics]:
    """Solve IK for the hand pose, falling back to the home keyframe joints."""
    robot_model = load_robot_model(str(robot_xml_path))
    nq_robot = robot_model_nq(robot_model)
    ik_solver = build_inverse_kinematics(
        robot_model,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    fk_solver = build_forward_kinematics(robot_model)
    robot_mj = robot_model_mj_model(robot_model)
    if int(robot_mj.nkey) > 0:
        initial_joints = np.array(robot_mj.key_qpos[0, :nq_robot], dtype=np.float64)
    else:
        initial_joints = np.zeros(nq_robot)

    ik_failed = False
    try:
        q_target = solve_inverse_kinematics(ik_solver, hand_pose, initial_joints)
    except ValueError as exc:
        if not quiet:
            logger.warning("IK failed: {}", exc)
        q_target = initial_joints
        ik_failed = True
    return q_target, ik_failed, fk_solver


def _place_object_at_gripper(
    scene: MuJoCoScene,
    object_id: str,
    ee_pose: np.ndarray,
    hand_pose: np.ndarray,
    *,
    quiet: bool,
) -> bool:
    """Place the object at the gripper after IK failure; return ``False`` when impossible."""
    mj_model = scene.model
    mj_data = scene.data
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, object_id)
    qposadr: int | None = None
    if body_id != -1:
        joint_adr = int(mj_model.body_jntadr[body_id])
        joint_count = int(mj_model.body_jntnum[body_id])
        if joint_adr >= 0 and joint_count > 0:
            for offset in range(joint_count):
                joint_id = joint_adr + offset
                if mj_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                    qposadr = int(mj_model.jnt_qposadr[joint_id])
                    break
    if qposadr is None:
        if not quiet:
            logger.warning("IK failed and object has no freejoint; skipping physics for this grasp.")
        return False
    object_pose = scene.body_pose(object_id)
    new_object_pose = ee_pose @ invert_transform(hand_pose) @ object_pose
    mj_data.qpos[qposadr : qposadr + 3] = new_object_pose[:3, 3]
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(
        quat,
        np.ascontiguousarray(new_object_pose[:3, :3], dtype=np.float64).reshape(9),
    )
    mj_data.qpos[qposadr + 3 : qposadr + 7] = quat
    for joint_id in range(mj_model.njnt):
        if int(mj_model.jnt_qposadr[joint_id]) == qposadr:
            dofadr = int(mj_model.jnt_dofadr[joint_id])
            mj_data.qvel[dofadr : dofadr + 6] = 0.0
            break
    mujoco.mj_forward(mj_model, mj_data)
    if not quiet:
        logger.warning("Placing object at the gripper because the arm cannot reach this pose.")
    return True


def _apply_finger_width_overrides(  # noqa: PLR0913, PLR0917
    mj_model: mujoco.MjModel,
    gripper_ids: list[int],
    open_cmd: np.ndarray,
    close_cmd: np.ndarray,
    grasp_width: float,
    joint_ranges: tuple[tuple[str, tuple[float, ...]], ...] = GRIPPER_JOINT_RANGES,
) -> None:
    """Override finger open/close targets from a desired gripper width."""
    open_q1, open_q2 = panda_width_to_finger_joints(grasp_width)
    ranges = dict(joint_ranges)
    close_q1 = float(ranges["finger_joint1"][0])
    close_q2 = float(ranges["finger_joint2"][0])
    finger_open = {"finger_joint1": open_q1, "finger_joint2": open_q2}
    finger_close = {"finger_joint1": close_q1, "finger_joint2": close_q2}
    for idx in gripper_ids:
        joint_id = int(mj_model.actuator_trnid[idx, 0])
        joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        for key, value in finger_open.items():
            if key in joint_name:
                open_cmd[idx] = value
        for key, value in finger_close.items():
            if key in joint_name:
                close_cmd[idx] = value


def _gripper_control_ranges(mj_model: mujoco.MjModel, gripper_ids: list[int]) -> tuple[list[float], list[float]]:
    """Read per-actuator open (hi) and close (lo) targets from control ranges."""
    open_vals = []
    close_vals = []
    for idx in gripper_ids:
        if mj_model.actuator_ctrllimited[idx]:
            lo, hi = mj_model.actuator_ctrlrange[idx]
        else:
            lo, hi = 0.0, 1.0
        open_vals.append(float(hi))
        close_vals.append(float(lo))
    return open_vals, close_vals


def _actuator_gripper_commands(  # noqa: PLR0913, PLR0917
    mj_model: mujoco.MjModel,
    gripper_ids: list[int],
    q_target: np.ndarray,
    gripper_close_command: np.ndarray,
    grasp_width: float | None,
    dual_count: int = GRIPPER_DUAL_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    """Build open/close commands from gripper actuator control ranges."""
    open_vals, close_vals = _gripper_control_ranges(mj_model, gripper_ids)
    open_cmd = np.zeros(int(mj_model.nu), dtype=np.float64)
    close_cmd = np.zeros(int(mj_model.nu), dtype=np.float64)
    gripper_set = set(gripper_ids)
    for i in range(int(mj_model.nu)):
        if i in gripper_set:
            open_cmd[i] = open_vals[0]
            close_cmd[i] = close_vals[0]
            continue
        if mj_model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            joint_id = int(mj_model.actuator_trnid[i, 0])
            qadr = int(mj_model.jnt_qposadr[joint_id])
            if qadr < q_target.shape[0]:
                open_cmd[i] = q_target[qadr]
                close_cmd[i] = q_target[qadr]
    for i, idx in enumerate(gripper_ids):
        open_cmd[idx] = open_vals[i]
        close_cmd[idx] = close_vals[i]
    if grasp_width is not None and len(gripper_ids) == dual_count:
        _apply_finger_width_overrides(mj_model, gripper_ids, open_cmd, close_cmd, grasp_width)
    close_len = gripper_close_command.shape[0]
    for i, idx in enumerate(gripper_ids):
        if i >= close_len:
            break
        close_cmd[idx] = float(gripper_close_command[i])
    return open_cmd, close_cmd


def _model_gripper_commands(
    mj_model: mujoco.MjModel,
    gripper_model: dict[str, object],
    gripper_close_command: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build open/close commands from the gripper model when no finger actuators exist."""
    raw_open = make_open_command(gripper_model).astype(np.float64)
    raw_close = make_close_command(gripper_model).astype(np.float64)
    nu_robot = int(mj_model.nu)
    if nu_robot < raw_open.shape[0]:
        open_cmd = raw_open[:nu_robot]
        close_cmd = raw_close[:nu_robot]
    elif nu_robot > raw_open.shape[0]:
        open_cmd = np.zeros(nu_robot, dtype=np.float64)
        close_cmd = np.zeros(nu_robot, dtype=np.float64)
        open_cmd[: raw_open.shape[0]] = raw_open
        close_cmd[: raw_close.shape[0]] = raw_close
    else:
        open_cmd = raw_open
        close_cmd = raw_close
    close_len = gripper_close_command.shape[0]
    if close_len > 0:
        overlay_len = min(close_len, nu_robot)
        close_cmd[:overlay_len] = gripper_close_command[:overlay_len]
    return open_cmd, close_cmd


def _evaluate_sim_state(  # noqa: PLR0913, PLR0917  # evaluation consumes the configured outcome thresholds
    scene: MuJoCoScene,
    object_id: str,
    initial_height: float,
    lift_height_threshold: float,
    max_linear_velocity: float,
    max_angular_velocity: float,
) -> tuple[float, np.ndarray, float, bool]:
    """Compute final height, velocity, contact count, and success from the sim state."""
    mj_model = scene.model
    mj_data = scene.data
    final_pose = scene.body_pose(object_id)
    final_height = float(final_pose[2, 3])

    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, object_id)
    if body_id == -1:
        msg = f"Body '{object_id}' not found in simulation model"
        raise ValueError(msg)
    object_velocity = np.array(mj_data.cvel[body_id], copy=True)

    contact_count = float(len(collect_contacts(scene.contacts, {object_id})))

    lift_judge = build_lift_outcome_judge(lift_height_threshold)
    stability_judge = build_stability_judge(max_linear_velocity, max_angular_velocity)
    lifted = evaluate_lift_success(lift_judge, initial_height, final_height)
    stable = evaluate_stability(stability_judge, object_velocity)
    success = bool(contact_count >= 1 and lifted and stable)
    return final_height, object_velocity, contact_count, success


def simulate_grasp(  # noqa: PLR0913, PLR0915, PLR0917  # public simulation API
    grasp_pose: np.ndarray,
    object_id: str,
    ycb_root: Path,
    robot_xml_path: Path,
    table_xml_path: Path | None,
    num_simulation_steps: int,
    gripper_close_command: np.ndarray,
    **options: Any,
) -> dict[str, np.ndarray | bool | float]:
    """Execute a single grasp in a MuJoCo simulation and report its outcome.

    Args:
        grasp_pose: Contact-frame grasp pose in the world frame as a ``(4, 4)``
            transformation (origin at the antipodal contact midpoint). The pose is
            converted to the Panda hand frame before inverse kinematics.
        object_id: Logical YCB object identifier to load.
        ycb_root: Root directory of the YCB object set.
        robot_xml_path: Path to the robot MJCF description.
        table_xml_path: Optional path to a workbench/table MJCF description.
        num_simulation_steps: Number of physics steps to execute.
        gripper_close_command: Gripper command used to close the gripper.
        grasp_width: Optional finger opening width in meters. When supplied and
            the model exposes two finger joint actuators, close targets are
            derived via ``panda_width_to_finger_joints``.
        quiet: When ``True``, suppress IK fallback diagnostic prints.
        lift_height_threshold: Minimum world-frame height gain required to
            count the grasp as a successful lift.
        max_linear_velocity: Maximum acceptable linear velocity of the object.
        max_angular_velocity: Maximum acceptable angular velocity of the object.
        options: Optional simulation overrides such as lift thresholds and grasp width.

    Returns:
        A dictionary describing the simulation outcome, including the success
        flag and any recorded contact, velocity, or trajectory information.

    Raises:
        ValueError: If inputs or simulation state are invalid.
        FileNotFoundError: If required robot or YCB assets are missing.
    """
    lift_height_threshold = float(options.pop("lift_height_threshold", LIFT_HEIGHT_THRESHOLD))
    max_linear_velocity = float(options.pop("max_linear_velocity", MAX_LINEAR_VELOCITY))
    max_angular_velocity = float(options.pop("max_angular_velocity", MAX_ANGULAR_VELOCITY))
    grasp_width = options.pop("grasp_width", None)
    quiet = bool(options.pop("quiet", False))
    if options:
        unexpected = ", ".join(sorted(options))
        raise TypeError(f"Unexpected grasp simulation options: {unexpected}")

    _validate_simulate_grasp_args(
        grasp_pose,
        robot_xml_path,
        ycb_root,
        num_simulation_steps,
        lift_height_threshold,
        max_linear_velocity,
        max_angular_velocity,
        grasp_width,
    )

    hand_to_contact = panda_hand_to_contact_transform()
    hand_pose = transform_grasp_pose(grasp_pose, invert_transform(hand_to_contact))

    object_dir = resolve_ycb_object_directory(ycb_root, object_id)
    object_xml_path = find_ycb_mjcf(object_dir)

    scene = MuJoCoScene(
        robot_xml_path,
        object_xml_path,
        table_xml_path,
        object_name=object_id,
    )
    mj_model = scene.model
    mj_data = scene.data
    object_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, object_id)
    if object_body_id == -1:
        msg = f"Body '{object_id}' not found in simulation model"
        raise ValueError(msg)

    q_target, ik_failed, fk_solver = _solve_grasp_ik(robot_xml_path, hand_pose, quiet=quiet)

    scene.reset()
    if mj_data.qpos.shape[0] >= q_target.shape[0]:
        mj_data.qpos[: q_target.shape[0]] = q_target
    mujoco.mj_forward(mj_model, mj_data)
    if ik_failed:
        ee_pose = fk_solver(q_target)
        placed = _place_object_at_gripper(scene, object_id, ee_pose, hand_pose, quiet=quiet)
        if not placed:
            return {
                "success": False,
                "initial_height": 0.0,
                "final_height": 0.0,
                "contact_count": 0.0,
                "object_velocity": np.zeros(6),
                "grasp_pose": grasp_pose,
                "fk_position_error": float("inf"),
            }

    initial_pose = scene.body_pose(object_id)
    initial_height = float(initial_pose[2, 3])

    dt = mj_model.opt.timestep
    if dt <= 0 or not np.isfinite(dt):
        dt = FALLBACK_TIMESTEP

    gripper_model = load_gripper_model(str(robot_xml_path))
    gripper_model["model"] = mj_model
    gripper_model["data"] = mj_data

    gripper_ids = gripper_actuator_indices(mj_model)
    if gripper_ids:
        open_cmd, close_cmd = _actuator_gripper_commands(
            mj_model,
            gripper_ids,
            q_target,
            gripper_close_command,
            grasp_width,
        )
    else:
        open_cmd, close_cmd = _model_gripper_commands(mj_model, gripper_model, gripper_close_command)

    pre_grasp_steps = min(
        PRE_GRASP_STEPS,
        max(1, num_simulation_steps // 4),
    )
    close_steps = max(1, num_simulation_steps - pre_grasp_steps)

    step_scene(lambda step_dt: scene.step(open_cmd, step_dt), dt, pre_grasp_steps)
    step_scene(lambda step_dt: scene.step(close_cmd, step_dt), dt, close_steps)

    if ik_failed:
        fk_position_error = float("inf")
    else:
        achieved_pose = fk_solver(q_target)
        fk_position_error = float(np.linalg.norm(achieved_pose[:3, 3] - hand_pose[:3, 3]))

    final_height, object_velocity, contact_count, success = _evaluate_sim_state(
        scene,
        object_id,
        initial_height,
        lift_height_threshold,
        max_linear_velocity,
        max_angular_velocity,
    )

    return {
        "success": success,
        "initial_height": initial_height,
        "final_height": final_height,
        "contact_count": contact_count,
        "object_velocity": object_velocity,
        "grasp_pose": grasp_pose,
        "fk_position_error": fk_position_error,
    }


def run_simulation_sweep(  # noqa: PLR0913  # public sweep API
    grasp_poses: np.ndarray,
    *,
    object_id: str,
    ycb_root: Path,
    robot_xml_path: Path,
    table_xml_path: Path | None,
    num_simulation_steps: int,
    gripper_close_command: np.ndarray,
    **options: Any,
) -> list[dict[str, np.ndarray | bool | float]]:
    """Execute a batch of grasps and aggregate the simulation outcomes.

    Args:
        grasp_poses: Grasp poses with shape ``(K, 4, 4)``.
        object_id: Logical YCB object identifier to load.
        ycb_root: Root directory of the YCB object set.
        robot_xml_path: Path to the robot MJCF description.
        table_xml_path: Optional path to a workbench/table MJCF description.
        num_simulation_steps: Number of physics steps to execute per grasp.
        gripper_close_command: Gripper command used to close the gripper.
        lift_height_threshold: Minimum world-frame height gain required to
            count the grasp as a successful lift.
        max_linear_velocity: Maximum acceptable linear velocity of the object.
        max_angular_velocity: Maximum acceptable angular velocity of the object.
        options: Optional simulation overrides keyed by configuration name.

    Returns:
        A list of per-grasp simulation outcomes.

    Raises:
        ValueError: If ``grasp_poses`` shape is invalid.
    """
    lift_height_threshold = float(options.pop("lift_height_threshold", LIFT_HEIGHT_THRESHOLD))
    max_linear_velocity = float(options.pop("max_linear_velocity", MAX_LINEAR_VELOCITY))
    max_angular_velocity = float(options.pop("max_angular_velocity", MAX_ANGULAR_VELOCITY))
    if options:
        unexpected = ", ".join(sorted(options))
        msg = f"Unexpected simulation sweep options: {unexpected}"
        raise TypeError(msg)

    if grasp_poses.ndim == POINT_CLOUD_NDIM:
        if grasp_poses.shape == SE3_MATRIX_SHAPE:
            grasp_poses = grasp_poses.reshape(1, *SE3_MATRIX_SHAPE)
        else:
            msg = "grasp_poses must have shape (K, 4, 4) or (4, 4)"
            raise ValueError(msg)

    if grasp_poses.ndim != GRASP_POSES_NDIM or grasp_poses.shape[1:] != SE3_MATRIX_SHAPE:
        msg = "grasp_poses must have shape (K, 4, 4)"
        raise ValueError(msg)

    outcomes = []
    for i in range(grasp_poses.shape[0]):
        outcome = simulate_grasp(
            grasp_pose=grasp_poses[i],
            object_id=object_id,
            ycb_root=ycb_root,
            robot_xml_path=robot_xml_path,
            table_xml_path=table_xml_path,
            num_simulation_steps=num_simulation_steps,
            gripper_close_command=gripper_close_command,
            lift_height_threshold=lift_height_threshold,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
        )
        outcomes.append(outcome)

    return outcomes
