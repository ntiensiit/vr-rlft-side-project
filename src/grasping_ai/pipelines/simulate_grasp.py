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
    panda_fingertip_object_contacts,
    panda_hand_to_contact_transform,
    panda_has_nonpad_object_collision,
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
    place_freejoint_body_on_surface,
    set_freejoint_body_pose,
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
GRIPPER_MAX_WIDTH = float(FLATTENED_YAML_CONFIG.get("robot.gripper.max_width", 0.08))
LIFT_HEIGHT_THRESHOLD = float(FLATTENED_YAML_CONFIG.get("metrics.lift_height_threshold"))
MAX_LINEAR_VELOCITY = float(FLATTENED_YAML_CONFIG.get("limits.max_linear_velocity"))
MAX_ANGULAR_VELOCITY = float(FLATTENED_YAML_CONFIG.get("limits.max_angular_velocity"))
PRE_GRASP_STEPS = int(FLATTENED_YAML_CONFIG.get("pre_grasp_steps"))
POINT_CLOUD_NDIM = int(FLATTENED_YAML_CONFIG.get("geometry.point_cloud_ndim", 2))
GRASP_POSES_NDIM = int(FLATTENED_YAML_CONFIG.get("grasp.poses_ndim", 3))
VALIDATION_CLOSE_STEPS = 250
VALIDATION_CONTACT_ACQUIRE_STEPS = 500
VALIDATION_CONTACT_SETTLE_STEPS = 100
VALIDATION_LIFT_STEPS = 500
# The Panda tendon length is half the total jaw opening. A 40 mm reduction in
# commanded width reaches the deployed actuator's rated 70 N force cap.
GRIPPER_CONTACT_PRELOAD_WIDTH = 0.040


def _robot_contacts_table(mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> bool:
    """Return whether a robot link intersects the table in the current state."""
    table_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "table")
    robot_root_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "link0")
    if table_id == -1 or robot_root_id == -1:
        return False

    def is_robot_body(body_id: int) -> bool:
        """Check if a given body ID belongs to the robot.

        Args:
            body_id: The MuJoCo body identifier to check.

        Returns:
            True if the body is part of the robot, False otherwise.
        """
        current = body_id
        while current > 0:
            if current == robot_root_id:
                return True
            current = int(mj_model.body_parentid[current])
        return False

    for contact_id in range(int(mj_data.ncon)):
        contact = mj_data.contact[contact_id]
        body_a = int(mj_model.geom_bodyid[contact.geom[0]])
        body_b = int(mj_model.geom_bodyid[contact.geom[1]])
        if (body_a == table_id and is_robot_body(body_b)) or (body_b == table_id and is_robot_body(body_a)):
            return True
    return False

def _validate_simulate_grasp_args(  # noqa: PLR0913, PLR0917  # validation keeps related simulation inputs together
    grasp_pose: np.ndarray,
    robot_xml_path: Path,
    ycb_root: Path,
    num_simulation_steps: int,
    lift_height_threshold: float,
    max_linear_velocity: float,
    max_angular_velocity: float,
    grasp_width: float | None,
    lift_distance: float,
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
    if lift_distance <= 0 or not np.isfinite(lift_distance):
        raise ValueError("lift_distance must be positive and finite")


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


def _finger_object_contacts(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    object_id: str,
) -> tuple[float, bool]:
    """Count object contacts on both opposed Panda fingertip pads."""
    return panda_fingertip_object_contacts(mj_model, mj_data, object_id)


def _evaluate_sim_state(  # noqa: PLR0913, PLR0917  # evaluation consumes the configured outcome thresholds
    scene: MuJoCoScene,
    object_id: str,
    initial_height: float,
    lift_height_threshold: float,
    max_linear_velocity: float,
    max_angular_velocity: float,
) -> tuple[float, np.ndarray, float, bool, bool, bool]:
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

    contact_count, bilateral_contact = _finger_object_contacts(mj_model, mj_data, object_id)

    lift_judge = build_lift_outcome_judge(lift_height_threshold)
    stability_judge = build_stability_judge(max_linear_velocity, max_angular_velocity)
    lifted = evaluate_lift_success(lift_judge, initial_height, final_height)
    stable = evaluate_stability(stability_judge, object_velocity)
    success = bool(bilateral_contact and lifted and stable)
    return final_height, object_velocity, contact_count, bilateral_contact, stable, success


def simulate_grasp(  # noqa: C901, PLR0912, PLR0913, PLR0915, PLR0917  # public simulation API
    grasp_pose: np.ndarray,
    object_id: str,
    ycb_root: Path,
    robot_xml_path: Path,
    table_xml_path: Path | None,
    num_simulation_steps: int,
    gripper_close_command: np.ndarray,
    **options: Any,  # noqa: ANN401
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
    lift_distance = float(options.pop("lift_distance", max(0.1, 2.0 * lift_height_threshold)))
    object_position = options.pop("object_position", None)
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
        lift_distance,
    )
    if object_position is not None:
        object_position = np.asarray(object_position, dtype=np.float64)
        if object_position.shape != (3,) or not np.isfinite(object_position).all():
            msg = "object_position must be a finite array with shape (3,)"
            raise ValueError(msg)

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
    lifted_grasp_pose = np.asarray(grasp_pose, dtype=np.float64).copy()
    lifted_grasp_pose[2, 3] += lift_distance
    lifted_hand_pose = transform_grasp_pose(lifted_grasp_pose, invert_transform(hand_to_contact))
    lift_ik_failed = False
    q_lift = q_target.copy()
    if not ik_failed:
        robot_model = load_robot_model(str(robot_xml_path))
        lift_solver = build_inverse_kinematics(
            robot_model,
            max_iterations=IK_MAX_ITERATIONS,
            tolerance=max(IK_TOLERANCE, 3e-3),
        )
        try:
            q_lift = solve_inverse_kinematics(lift_solver, lifted_hand_pose, q_target)
        except ValueError as exc:
            if not quiet:
                logger.warning("Lift IK failed: {}", exc)
            lift_ik_failed = True

    scene.reset()
    if object_position is not None:
        set_freejoint_body_pose(mj_model, mj_data, object_id, object_position)
    if table_xml_path is not None:
        place_freejoint_body_on_surface(mj_model, mj_data, object_id)
    if mj_data.qpos.shape[0] >= q_target.shape[0]:
        mj_data.qpos[: q_target.shape[0]] = q_target
    mujoco.mj_forward(mj_model, mj_data)
    if ik_failed or lift_ik_failed:
        # Never move the object to manufacture a contact after IK failure.
        # That made invalid grasps look physically successful and produced
        # floating objects in validation and visualization.
        initial_pose = scene.body_pose(object_id)
        return {
            "success": False,
            "ik_converged": not ik_failed,
            "lift_ik_converged": not lift_ik_failed,
            "initial_height": float(initial_pose[2, 3]),
            "final_height": float(initial_pose[2, 3]),
            "contact_count": 0.0,
            "bilateral_contact": False,
            "stable": False,
            "table_collision_free": False,
            "object_velocity": np.zeros(6),
            "grasp_pose": grasp_pose,
            "fk_position_error": float("inf"),
        }

    if _robot_contacts_table(mj_model, mj_data):
        # A pose that intersects the work surface is not a valid candidate.
        # Reject it here, while generating the dataset, so it cannot later be
        # selected by the visualizer simply because its numerical IK converged.
        initial_pose = scene.body_pose(object_id)
        logger.debug("Rejecting grasp for {}: robot/table collision at the IK pose", object_id)
        return {
            "success": False,
            "ik_converged": True,
            "lift_ik_converged": True,
            "initial_height": float(initial_pose[2, 3]),
            "final_height": float(initial_pose[2, 3]),
            "contact_count": 0.0,
            "bilateral_contact": False,
            "stable": False,
            "contact_sustained": False,
            "initial_robot_object_collision_free": True,
            "table_collision_free": False,
            "object_velocity": np.zeros(6),
            "grasp_pose": grasp_pose,
            "fk_position_error": float(np.linalg.norm(fk_solver(q_target)[:3, 3] - hand_pose[:3, 3])),
        }

    if panda_has_nonpad_object_collision(mj_model, mj_data, object_id):
        # The candidate describes fingertip contacts, so the open grasp pose
        # must not begin with the palm or an arm link embedded in the object.
        # Such penetration was the source of the apparent floating/ejection:
        # MuJoCo resolved the overlap before the fingers ever established a
        # physical grasp.
        initial_pose = scene.body_pose(object_id)
        logger.debug("Rejecting grasp for {}: non-pad Panda/object collision", object_id)
        return {
            "success": False,
            "ik_converged": True,
            "lift_ik_converged": True,
            "initial_height": float(initial_pose[2, 3]),
            "final_height": float(initial_pose[2, 3]),
            "contact_count": 0.0,
            "bilateral_contact": False,
            "stable": False,
            "contact_sustained": False,
            "initial_robot_object_collision_free": False,
            "table_collision_free": True,
            "object_velocity": np.zeros(6),
            "grasp_pose": grasp_pose,
            "fk_position_error": float(np.linalg.norm(fk_solver(q_target)[:3, 3] - hand_pose[:3, 3])),
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

    pre_grasp_steps = min(PRE_GRASP_STEPS, max(1, num_simulation_steps // 20))
    fixed_motion_steps = (
        pre_grasp_steps
        + VALIDATION_CLOSE_STEPS
        + VALIDATION_CONTACT_ACQUIRE_STEPS
        + VALIDATION_CONTACT_SETTLE_STEPS
        + VALIDATION_LIFT_STEPS
    )
    if num_simulation_steps >= fixed_motion_steps:
        close_steps = VALIDATION_CLOSE_STEPS
        contact_acquisition_steps = VALIDATION_CONTACT_ACQUIRE_STEPS
        contact_settle_steps = VALIDATION_CONTACT_SETTLE_STEPS
        lift_steps = VALIDATION_LIFT_STEPS
    else:
        available_steps = num_simulation_steps - pre_grasp_steps
        close_steps = max(1, round(available_steps * 0.20))
        contact_acquisition_steps = max(1, round(available_steps * 0.25))
        contact_settle_steps = max(1, round(available_steps * 0.05))
        lift_steps = max(1, round(available_steps * 0.30))
    final_settle_steps = max(
        0,
        num_simulation_steps
        - pre_grasp_steps
        - close_steps
        - contact_acquisition_steps
        - contact_settle_steps
        - lift_steps,
    )

    lift_cmd = close_cmd.copy()
    for actuator_id in range(int(mj_model.nu)):
        if mj_model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        joint_id = int(mj_model.actuator_trnid[actuator_id, 0])
        qpos_address = int(mj_model.jnt_qposadr[joint_id])
        if qpos_address < q_lift.shape[0]:
            lift_cmd[actuator_id] = q_lift[qpos_address]

    table_collision_free = True
    max_object_height = initial_height

    def step_and_check(command: np.ndarray, step_dt: float) -> None:
        """Step the simulation and check for table collisions and height gains.

        Args:
            command: The actuator control command array.
            step_dt: The timestep duration.
        """
        nonlocal max_object_height, table_collision_free
        scene.step(command, step_dt)
        table_collision_free = table_collision_free and not _robot_contacts_table(mj_model, mj_data)
        max_object_height = max(max_object_height, float(scene.body_pose(object_id)[2, 3]))

    step_scene(lambda step_dt: step_and_check(open_cmd, step_dt), dt, pre_grasp_steps)
    # Closing in one control step creates a high-energy impact (especially
    # with the Panda's position-controlled tendon) that can eject the object
    # and masquerade as a successful lift.  Ramp the target just like a real
    # gripper controller.  Once both fingers touch, retain only a small
    # position preload instead of continuing toward zero width and crushing
    # or launching the object.
    contact_hold_cmd: np.ndarray | None = None
    contact_width = float("nan")

    def capture_contact_hold_command(base_command: np.ndarray) -> np.ndarray:
        """Capture the current gripper width and build a command to hold it.

        Args:
            base_command: The underlying close command being executed.

        Returns:
            A new command array configured to hold the current contact width.
        """
        nonlocal contact_width
        held = np.asarray(base_command, dtype=np.float64).copy()
        actual_width = float(np.sum(mj_data.qpos[q_target.shape[0] - 2 : q_target.shape[0]]))
        contact_width = actual_width
        target_width = max(0.0, actual_width - GRIPPER_CONTACT_PRELOAD_WIDTH)
        for actuator_id in gripper_ids:
            lo, hi = mj_model.actuator_ctrlrange[actuator_id]
            held[actuator_id] = float(
                np.clip(lo + (target_width / GRIPPER_MAX_WIDTH) * (hi - lo), lo, hi),
            )
        return held

    for command in np.linspace(open_cmd, close_cmd, close_steps):
        active_command = contact_hold_cmd if contact_hold_cmd is not None else command
        step_and_check(active_command, dt)
        if contact_hold_cmd is None and _finger_object_contacts(mj_model, mj_data, object_id)[1]:
            contact_hold_cmd = capture_contact_hold_command(command)

    # Position-controlled fingers lag their command. Give them a bounded
    # acquisition window at the grasp pose and capture the *measured* opening
    # when bilateral pad contact occurs. Unused acquisition time is retained
    # as a final stability hold so every candidate receives the same number
    # of simulation steps.
    acquisition_steps_used = 0
    for _acquisition_steps_used in range(1, contact_acquisition_steps + 1):
        acquisition_steps_used = _acquisition_steps_used
        active_command = contact_hold_cmd if contact_hold_cmd is not None else close_cmd
        step_and_check(active_command, dt)
        if contact_hold_cmd is None and _finger_object_contacts(mj_model, mj_data, object_id)[1]:
            contact_hold_cmd = capture_contact_hold_command(active_command)
        if contact_hold_cmd is not None:
            break
    unused_acquisition_steps = contact_acquisition_steps - acquisition_steps_used

    contact_sustained = contact_hold_cmd is not None
    contact_settle_fraction = 0.0
    if contact_hold_cmd is not None:
        contact_settle_count = 0
        for _ in range(contact_settle_steps):
            step_and_check(contact_hold_cmd, dt)
            if _finger_object_contacts(mj_model, mj_data, object_id)[1]:
                contact_settle_count += 1
            else:
                contact_sustained = False
                break
        contact_settle_fraction = contact_settle_count / contact_settle_steps

    lift_contact_fraction = 0.0
    final_hold_contact_fraction = 0.0
    if not contact_sustained:
        # A lift without a bilateral grasp only knocks the object around and
        # can create a false height gain.  Keep the arm at the grasp pose and
        # let the object settle; the outcome will be rejected below.
        step_scene(
            lambda step_dt: step_and_check(close_cmd, step_dt),
            dt,
            lift_steps + final_settle_steps + unused_acquisition_steps,
        )
    else:
        if contact_hold_cmd is None:  # pragma: no cover - guarded by contact_sustained
            raise RuntimeError("contact hold command missing after sustained contact")
        close_cmd = contact_hold_cmd
        lift_cmd[gripper_ids] = contact_hold_cmd[gripper_ids]
        lift_contact_sustained = True
        lift_contact_count = 0
        for command in np.linspace(close_cmd, lift_cmd, lift_steps):
            step_and_check(command, dt)
            if _finger_object_contacts(mj_model, mj_data, object_id)[1]:
                lift_contact_count += 1
            else:
                lift_contact_sustained = False
        lift_contact_fraction = lift_contact_count / lift_steps
        contact_sustained = contact_sustained and lift_contact_sustained
        final_hold_steps = final_settle_steps + unused_acquisition_steps
        final_hold_contact_count = 0
        for _ in range(final_hold_steps):
            step_and_check(lift_cmd, dt)
            if _finger_object_contacts(mj_model, mj_data, object_id)[1]:
                final_hold_contact_count += 1
            else:
                contact_sustained = False
        final_hold_contact_fraction = (
            final_hold_contact_count / final_hold_steps if final_hold_steps else 1.0
        )

    if ik_failed:
        fk_position_error = float("inf")
    else:
        achieved_pose = fk_solver(q_target)
        fk_position_error = float(np.linalg.norm(achieved_pose[:3, 3] - hand_pose[:3, 3]))

    final_height, object_velocity, contact_count, bilateral_contact, stable, success = _evaluate_sim_state(
        scene,
        object_id,
        initial_height,
        lift_height_threshold,
        max_linear_velocity,
        max_angular_velocity,
    )
    hand_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    final_hand_height = float(mj_data.xpos[hand_body_id, 2]) if hand_body_id != -1 else float("nan")
    lift_joint_error = float(np.linalg.norm(mj_data.qpos[: q_lift.shape[0]] - q_lift))

    return {
        "success": bool(success and contact_sustained and table_collision_free),
        "ik_converged": not ik_failed,
        "lift_ik_converged": not lift_ik_failed,
        "initial_height": initial_height,
        "final_height": final_height,
        "contact_count": contact_count,
        "bilateral_contact": bilateral_contact,
        "stable": stable,
        "contact_sustained": contact_sustained,
        "initial_robot_object_collision_free": True,
        "contact_width": contact_width,
        "contact_settle_fraction": contact_settle_fraction,
        "lift_contact_fraction": lift_contact_fraction,
        "final_hold_contact_fraction": final_hold_contact_fraction,
        "max_object_height": max_object_height,
        "table_collision_free": table_collision_free,
        "object_velocity": object_velocity,
        "grasp_pose": grasp_pose,
        "fk_position_error": fk_position_error,
        "final_hand_height": final_hand_height,
        "target_hand_height": float(lifted_hand_pose[2, 3]),
        "lift_joint_error": lift_joint_error,
        "final_robot_qpos": np.array(mj_data.qpos[: q_lift.shape[0]], copy=True),
        "target_lift_qpos": np.array(q_lift, copy=True),
        "final_ctrl": np.array(mj_data.ctrl, copy=True),
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
    object_position: np.ndarray | None = None,
    **options: Any,  # noqa: ANN401
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
        object_position: Optional initial body-origin position before table
            surface placement.
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
            object_position=object_position,
        )
        outcomes.append(outcome)

    return outcomes
