"""MuJoCo robot viewer."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco  # type: ignore[import-untyped]
import mujoco.viewer  # type: ignore[import-untyped]
import numpy as np
import pytransform3d.rotations as pr
from loguru import logger
from scipy.optimize import least_squares

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
from grasping_ai.perception.geometry import invert_transform
from grasping_ai.robotics.gripper import (
    MAX_WIDTH,
    panda_fingertip_object_contacts,
    panda_hand_to_contact_transform,
    panda_width_to_finger_joints,
)
from grasping_ai.robotics.kinematics import (
    build_inverse_kinematics,
    load_robot_model,
    robot_model_mj_model,
    robot_model_nq,
    solve_inverse_kinematics,
)
from grasping_ai.robotics.transforms import transform_grasp_pose
from grasping_ai.simulation.scene import MuJoCoScene, place_freejoint_body_on_surface, set_freejoint_body_pose
from grasping_ai.simulation.ycb import (
    find_ycb_mjcf,
    resolve_ycb_object_directory,
)
from grasping_ai.utils.path_validation import (
    require_optional_path,
    require_path,
)

if TYPE_CHECKING:
    from collections.abc import Callable

FALLBACK_TIMESTEP = float(FLATTENED_YAML_CONFIG.get("fallback_timestep", 0.002))
SE3_MATRIX_SHAPE = tuple(int(v) for v in FLATTENED_YAML_CONFIG.get("grasp.se3_matrix_shape", [4, 4]))
IK_MAX_ITERATIONS = int(FLATTENED_YAML_CONFIG.get("robot.ik.max_iterations"))
IK_TOLERANCE = float(FLATTENED_YAML_CONFIG.get("robot.ik.tolerance"))
MIN_TRAJECTORY_STEPS = 2
MIN_CONTACT_LIFT_STEPS = 12
LIFT_IK_TOLERANCE = 3e-3
TOP_DOWN_IK_MAX_RESIDUAL = 5e-3
TOP_DOWN_IK_WARN_RESIDUAL = 2e-4
TOP_DOWN_DEFAULT_GRIPPER_WIDTH = 0.072
GRIPPER_CLOSED_THRESHOLD = 0.01


def load_visualization_scene(  # noqa: C901
    robot_xml_path: Path,
    object_id: str | None = None,
    ycb_root: Path | None = None,
    table_xml_path: Path | None = None,
    object_position: np.ndarray | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load a MuJoCo model and data for interactive viewing.

    Args:
        robot_xml_path: Path to the robot MJCF description.
        object_id: Optional YCB object identifier to include in the scene.
        ycb_root: Root directory of the YCB MJCF set. Required when
            ``object_id`` is set.
        table_xml_path: Optional workbench/table MJCF description.
        object_position: Optional world-frame object position. When omitted,
            objects with a table are placed on its surface.

    Returns:
        ``(mj_model, mj_data)`` ready for ``mujoco.viewer``.
    """
    require_path(robot_xml_path, "robot_xml_path")
    if not robot_xml_path.is_file():
        msg = f"robot_xml_path not found: {robot_xml_path}"
        raise FileNotFoundError(msg)
    require_optional_path(table_xml_path, "table_xml_path")
    if table_xml_path is not None and not table_xml_path.is_file():
        msg = f"table_xml_path not found: {table_xml_path}"
        raise FileNotFoundError(msg)
    if object_id is not None and not isinstance(object_id, str):
        msg = "object_id must be a string or None"
        raise TypeError(msg)
    object_xml_path = None
    if object_id:
        if ycb_root is None or not isinstance(ycb_root, Path):
            msg = "ycb_root is required when object_id is set"
            raise ValueError(msg)
        if not ycb_root.is_dir():
            msg = f"ycb_root not found: {ycb_root}"
            raise FileNotFoundError(msg)

        object_xml_path = find_ycb_mjcf(resolve_ycb_object_directory(ycb_root, object_id))

    scene = MuJoCoScene(
        robot_xml_path,
        object_xml_path,
        table_xml_path,
        object_name=object_id,
    )
    if object_id is not None and object_position is not None:
        set_freejoint_body_pose(scene.model, scene.data, object_id, object_position)
        if table_xml_path is not None and table_xml_path.stem == "floor":
            # A floor-only visualization is a display anchor, not a physical
            # workspace. Keep the free-joint object at its requested pose.
            scene.model.opt.gravity[:] = 0.0
            body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, object_id)
            if body_id != -1:
                scene.model.body_gravcomp[body_id] = 1.0
        elif table_xml_path is not None:
            # The requested position is a body-origin seed, not the final
            # resting pose.  Account for each mesh's baked-in translation and
            # place its compiled geometry on the actual table surface.
            place_freejoint_body_on_surface(scene.model, scene.data, object_id)
    elif object_id is not None and table_xml_path is not None:
        place_freejoint_body_on_surface(scene.model, scene.data, object_id)
    # Keep MuJoCo contacts and gravity enabled.  The viewer must show the
    # physical result of the selected grasp candidate, rather than a visual
    # attachment that can make an ungrasped object appear to float.
    return scene.model, scene.data


def apply_grasp_pose(  # noqa: PLR0913
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    robot_xml_path: Path,
    grasp_pose: np.ndarray,
    *,
    allow_ik_failure: bool = True,
    close_gripper: bool = False,
) -> np.ndarray:
    """Set the robot joints to the configuration for a grasp pose."""
    if grasp_pose.shape != SE3_MATRIX_SHAPE:
        msg = f"grasp_pose must have shape {SE3_MATRIX_SHAPE}, got {grasp_pose.shape}"
        raise ValueError(msg)
    if not np.isfinite(grasp_pose).all():
        raise ValueError("grasp_pose must contain only finite values")

    hand_pose = transform_grasp_pose(
        grasp_pose,
        invert_transform(panda_hand_to_contact_transform()),
    )
    robot_model = load_robot_model(str(robot_xml_path))
    robot_mj_model = robot_model_mj_model(robot_model)
    nq_robot = robot_model_nq(robot_model)
    ik_solver = build_inverse_kinematics(
        robot_model,
        max_iterations=IK_MAX_ITERATIONS,
        tolerance=IK_TOLERANCE,
    )
    if int(robot_mj_model.nkey) > 0:
        initial_joints = np.array(robot_mj_model.key_qpos[0, :nq_robot], dtype=np.float64)
    else:
        initial_joints = np.zeros(nq_robot, dtype=np.float64)

    try:
        q_target = solve_inverse_kinematics(ik_solver, hand_pose, initial_joints)
    except ValueError as exc:
        if not allow_ik_failure:
            raise
        logger.warning(
            "Could not reach grasp pose; opening viewer at the robot home pose. "
            "Use script.allow_ik_failure=false to fail instead: {}",
            exc,
        )
        q_target = initial_joints
    if mj_data.qpos.shape[0] < q_target.shape[0]:
        msg = f"Scene qpos has size {mj_data.qpos.shape[0]}, but grasp requires {q_target.shape[0]}"
        raise ValueError(msg)
    mj_data.qpos[: q_target.shape[0]] = q_target
    _hold_robot_pose(mj_model, mj_data, q_target, close_gripper=close_gripper)
    mujoco.mj_forward(mj_model, mj_data)
    if _robot_contacts_table(mj_model, mj_data):
        msg = "IK solution causes a Panda/table collision"
        if not allow_ik_failure:
            raise ValueError(msg)
        logger.warning("{}; displaying it for visualization", msg)
    logger.info("Loaded grasp pose into the robot configuration")
    return q_target


def build_lift_trajectory(
    robot_xml_path: Path,
    grasp_pose: np.ndarray,
    q_start: np.ndarray,
    lift_height: float,
    steps: int = 80,
) -> np.ndarray:
    """Solve and interpolate a vertical lift from a grasp pose."""
    if lift_height <= 0 or steps < MIN_TRAJECTORY_STEPS:
        raise ValueError("lift_height must be positive and steps must be at least 2")
    lifted_grasp = np.array(grasp_pose, dtype=np.float64, copy=True)
    lifted_grasp[2, 3] += lift_height
    hand_pose = transform_grasp_pose(
        lifted_grasp,
        invert_transform(panda_hand_to_contact_transform()),
    )
    robot_model = load_robot_model(str(robot_xml_path))
    ik_solver = build_inverse_kinematics(
        robot_model,
        max_iterations=IK_MAX_ITERATIONS,
        tolerance=LIFT_IK_TOLERANCE,
    )
    q_lift = solve_inverse_kinematics(ik_solver, hand_pose, np.asarray(q_start, dtype=np.float64))
    return np.linspace(q_start, q_lift, steps)


def build_contact_lift_trajectory(  # noqa: PLR0913
    robot_xml_path: Path,
    grasp_pose: np.ndarray,
    q_grasp: np.ndarray,
    lift_height: float,
    steps: int = 120,
    *,
    gripper_width: float = TOP_DOWN_DEFAULT_GRIPPER_WIDTH,
) -> tuple[np.ndarray, int]:
    """Build a candidate-based close-and-lift path without moving the object.

    The returned lift index is the first frame that raises the arm.  Callers
    can gate that phase on actual finger/object contacts in MuJoCo.
    """
    if lift_height <= 0 or steps < MIN_CONTACT_LIFT_STEPS:
        raise ValueError(f"lift_height must be positive and steps must be at least {MIN_CONTACT_LIFT_STEPS}")
    q_open = np.asarray(q_grasp, dtype=np.float64).copy()
    q_open[-2:] = panda_width_to_finger_joints(gripper_width)
    q_closed = q_open.copy()
    q_closed[-2:] = panda_width_to_finger_joints(0.003)
    robot_model = load_robot_model(str(robot_xml_path))
    ik_solver = build_inverse_kinematics(
        robot_model,
        max_iterations=IK_MAX_ITERATIONS,
        tolerance=LIFT_IK_TOLERANCE,
    )
    # Match physical validation at the default one-second playback: 0.1 s
    # open, 0.4 s closing, 0.35 s lifting, then a 0.15 s stability hold.
    settle_steps = max(2, round(steps * 0.10))
    close_steps = max(2, round(steps * 0.40))
    hold_steps = max(2, round(steps * 0.15))
    lift_steps = steps - settle_steps - close_steps - hold_steps
    lifted_grasp = np.asarray(grasp_pose, dtype=np.float64).copy()
    lifted_grasp[2, 3] += lift_height
    lifted_hand = transform_grasp_pose(
        lifted_grasp,
        invert_transform(panda_hand_to_contact_transform()),
    )
    q_lift = solve_inverse_kinematics(ik_solver, lifted_hand, q_closed)
    q_lift[-2:] = q_closed[-2:]
    trajectory = np.vstack(
        (
            np.repeat(q_open.reshape(1, -1), settle_steps, axis=0),
            np.linspace(q_open, q_closed, close_steps),
            np.linspace(q_closed, q_lift, lift_steps),
            np.repeat(q_lift.reshape(1, -1), hold_steps, axis=0),
        ),
    )
    return trajectory, settle_steps + close_steps


def build_top_down_pick_trajectory(  # noqa: C901, PLR0913, PLR0915, PLR0917
    robot_xml_path: Path,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    object_id: str,
    lift_height: float,
    steps: int = 80,
    grasp_pose: np.ndarray | None = None,
    gripper_width: float = TOP_DOWN_DEFAULT_GRIPPER_WIDTH,
    *,
    animate: bool = True,
) -> np.ndarray:
    """Solve a top-down Panda pick and its vertical lift trajectory.

    The generated grasp files contain side grasps for this object and are not
    suitable for a vertical pick.  This path uses the compiled MuJoCo mesh
    bounds instead: the hand is placed just above the object's top surface,
    with the Panda hand z-axis pointing downward.  IK uses a bounded least-
    squares solve because the legacy damped-Jacobian solver can stall at the
    Panda's straight-down wrist singularity.
    """
    if lift_height <= 0 or steps < MIN_TRAJECTORY_STEPS:
        raise ValueError("lift_height must be positive and steps must be at least 2")

    object_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, object_id)
    if object_body_id == -1:
        raise ValueError(f"Object body '{object_id}' not found in visualization scene")
    object_geom_id = next(
        (geom_id for geom_id in range(int(mj_model.ngeom))
         if int(mj_model.geom_bodyid[geom_id]) == object_body_id),
        -1,
    )
    if object_geom_id == -1:
        raise ValueError(f"Object '{object_id}' has no MuJoCo geometry")

    mujoco.mj_forward(mj_model, mj_data)
    object_center = np.array(mj_data.geom_xpos[object_geom_id], dtype=np.float64)
    object_half_height = float(mj_model.geom_aabb[object_geom_id, 5])

    robot_model = load_robot_model(str(robot_xml_path))
    robot_mj_model = robot_model_mj_model(robot_model)
    nq_robot = robot_model_nq(robot_model)
    hand_id = mujoco.mj_name2id(robot_mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if hand_id == -1:
        raise ValueError("Panda hand body was not found in robot XML")
    home_q = (
        np.array(robot_mj_model.key_qpos[0, :nq_robot], dtype=np.float64)
        if int(robot_mj_model.nkey) > 0
        else np.zeros(nq_robot, dtype=np.float64)
    )
    home_data = mujoco.MjData(robot_mj_model)
    home_data.qpos[:nq_robot] = home_q
    mujoco.mj_forward(robot_mj_model, home_data)
    if grasp_pose is None:
        hand_rotation = np.array(home_data.xmat[hand_id].reshape(3, 3), dtype=np.float64)
    else:
        # Keep the yaw of the selected pose from the user's .npz, but invert
        # its approach axis so the same grasp comes from above the box.
        yaw = float(np.arctan2(grasp_pose[1, 0], grasp_pose[0, 0]))
        c, s = np.cos(yaw), np.sin(yaw)
        contact_rotation = np.array(
            [[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )
        hand_rotation = contact_rotation @ invert_transform(
            panda_hand_to_contact_transform(),
        )[:3, :3]

    # The hand's local +z points toward the fingers.  At the Panda home pose
    # it points downward, so placing the hand just above the box makes the
    # fingers descend around the box instead of lifting it from underneath.
    hand_clearance = 0.012
    grasp_hand = np.eye(4, dtype=np.float64)
    grasp_hand[:3, :3] = hand_rotation
    grasp_hand[:3, 3] = object_center + np.array(
        # The finger bodies start 58.4 mm along the hand's local +z axis.
        # Since that axis points down here, their tips reach the box top.
        [0.0, 0.0, object_half_height + hand_clearance + 0.0584],
        dtype=np.float64,
    )
    lift_hand = grasp_hand.copy()
    lift_hand[2, 3] += lift_height

    joint_lower = np.full(nq_robot, -np.inf, dtype=np.float64)
    joint_upper = np.full(nq_robot, np.inf, dtype=np.float64)
    for joint_id in range(int(robot_mj_model.njnt)):
        if not robot_mj_model.jnt_limited[joint_id]:
            continue
        qadr = int(robot_mj_model.jnt_qposadr[joint_id])
        if robot_mj_model.jnt_type[joint_id] in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            joint_lower[qadr], joint_upper[qadr] = robot_mj_model.jnt_range[joint_id]

    def solve_target(target: np.ndarray, initial: np.ndarray) -> np.ndarray:
        def residual(q: np.ndarray) -> np.ndarray:
            local_data = mujoco.MjData(robot_mj_model)
            local_data.qpos[:nq_robot] = q
            mujoco.mj_forward(robot_mj_model, local_data)
            current = np.eye(4, dtype=np.float64)
            current[:3, :3] = local_data.xmat[hand_id].reshape(3, 3)
            current[:3, 3] = local_data.xpos[hand_id]
            delta = target @ invert_transform(current)
            return np.hstack((delta[:3, 3], pr.compact_axis_angle_from_matrix(delta[:3, :3])))

        result = least_squares(
            residual,
            initial,
            bounds=(joint_lower, joint_upper),
            max_nfev=1200,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        residual_norm = float(np.linalg.norm(result.fun))
        if residual_norm > TOP_DOWN_IK_MAX_RESIDUAL:
            raise ValueError(f"Top-down IK failed with residual {residual_norm:.6f}")
        if residual_norm > TOP_DOWN_IK_WARN_RESIDUAL:
            logger.warning("Top-down IK residual is {:.6f} m/rad", residual_norm)
        return np.asarray(result.x, dtype=np.float64)

    q_grasp = solve_target(grasp_hand, home_q)
    approach_hand = grasp_hand.copy()
    approach_hand[2, 3] += max(0.08, min(0.16, object_half_height + 0.08))
    q_approach = solve_target(approach_hand, home_q)
    q_lift = solve_target(lift_hand, q_grasp)
    finger_q1, finger_q2 = panda_width_to_finger_joints(gripper_width)
    q_approach[-2:] = [finger_q1, finger_q2]
    q_open = q_grasp.copy()
    q_open[-2:] = [finger_q1, finger_q2]
    q_close = q_grasp.copy()
    q_close[-2:] = [panda_width_to_finger_joints(0.003)[0], panda_width_to_finger_joints(0.003)[1]]
    q_lift[-2:] = q_close[-2:]

    if not animate:
        mj_data.qpos[:nq_robot] = q_close
        mujoco.mj_forward(mj_model, mj_data)
        logger.info("Solved static top-down grasp for '{}'", object_id)
        # A one-frame trajectory routes static mode through the same actuator
        # initialization path as animated mode without advancing the pose.
        return q_close.reshape(1, nq_robot)

    # Keep the phases explicit so the viewer can delay object attachment until
    # the fingers have actually closed around it.
    approach_steps = max(8, steps // 5)
    descend_steps = max(8, steps // 5)
    close_steps = max(8, steps // 6)
    lift_steps = max(8, steps - approach_steps - descend_steps - close_steps)
    trajectory = np.vstack(
        (
            np.linspace(home_q, q_approach, approach_steps, endpoint=False),
            np.linspace(q_approach, q_open, descend_steps, endpoint=False),
            np.linspace(q_open, q_close, close_steps, endpoint=False),
            np.linspace(q_close, q_lift, lift_steps),
        ),
    )
    mj_data.qpos[:nq_robot] = q_approach
    mujoco.mj_forward(mj_model, mj_data)
    logger.info(
        "Solved top-down pick for '{}' from selected grasp data at hand z={:.4f}; "
        "gripper width={:.4f} m; lifting by {:.3f} m",
        object_id,
        grasp_hand[2, 3],
        gripper_width,
        lift_height,
    )
    return trajectory


def _robot_contacts_table(mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> bool:
    """Return whether any Panda body is currently touching the table."""
    table_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "table")
    robot_root_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "link0")
    if table_id == -1 or robot_root_id == -1:
        return False

    def is_robot_body(body_id: int) -> bool:
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


def _gripper_contacts_object(mj_model: mujoco.MjModel, mj_data: mujoco.MjData, object_id: str) -> bool:
    """Return whether both opposed Panda fingertip pads touch the object."""
    return panda_fingertip_object_contacts(mj_model, mj_data, object_id)[1]


def _hold_robot_pose(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    q_target: np.ndarray,
    *,
    close_gripper: bool,
) -> None:
    """Set actuator targets so physics holds the displayed robot pose."""
    for actuator_id in range(int(mj_model.nu)):
        if mj_model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        joint_id = int(mj_model.actuator_trnid[actuator_id, 0])
        qpos_address = int(mj_model.jnt_qposadr[joint_id])
        if qpos_address < q_target.shape[0]:
            mj_data.ctrl[actuator_id] = q_target[qpos_address]
    if int(mj_model.nkey) > 0 and mj_model.nu > 0:
        for actuator_id in range(int(mj_model.nu)):
            if mj_model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_TENDON:
                if close_gripper:
                    mj_data.ctrl[actuator_id] = 0.0
                    continue
                # The Panda tendon control range maps linearly from closed to
                # its maximum opening.  Follow the trajectory's actual finger
                # width; the old boolean switch kept the gripper fully open
                # through the whole close phase and snapped it shut at the
                # final frame.
                width = float(np.sum(q_target[-2:]))
                lo, hi = mj_model.actuator_ctrlrange[actuator_id]
                mj_data.ctrl[actuator_id] = float(np.clip(lo + (width / MAX_WIDTH) * (hi - lo), lo, hi))


def convert_grasp_pose_to_world(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    grasp_pose: np.ndarray,
    *,
    object_id: str | None,
    pose_format: str,
) -> np.ndarray:
    """Convert an object-frame grasp pose into the assembled scene frame."""
    if pose_format == "world":
        return grasp_pose
    if pose_format != "object":
        msg = f"Unsupported grasp_pose_format '{pose_format}'; use 'object' or 'world'"
        raise ValueError(msg)
    if object_id is None:
        raise ValueError("object_id is required for object-frame grasp poses")

    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, object_id)
    if body_id == -1:
        raise ValueError(f"Object body '{object_id}' not found in visualization scene")
    object_to_world = np.eye(4, dtype=np.float64)
    object_to_world[:3, :3] = mj_data.xmat[body_id].reshape(3, 3)
    object_to_world[:3, 3] = mj_data.xpos[body_id]
    return object_to_world @ grasp_pose


def run_robot_viewer(  # noqa: C901, PLR0912,PLR0913  # viewer hooks are injected by tests as keywords
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    *,
    launch_passive: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    max_duration_s: float | None = None,
    lift_trajectory: np.ndarray | None = None,
    object_id: str | None = None,
    trajectory_duration_s: float = 8.0,
    lift_start_index: int | None = None,
    require_gripper_contact: bool = False,
) -> None:
    """Open a MuJoCo passive viewer and step physics until the window closes.

    Actuator and simulation controls are handled by MuJoCo's built-in viewer UI.

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.
        launch_passive: Optional viewer factory (defaults to mujoco.viewer).
        sleep: Optional sleep function.
        clock: Optional clock function returning seconds.
        max_duration_s: Optional wall-clock limit, used by tests.
        lift_trajectory: Optional joint trajectory for the top-down lift.
        object_id: Optional freejoint object used only for physical contact checks.
        trajectory_duration_s: Playback duration for the trajectory in seconds.
        lift_start_index: First trajectory frame that raises the arm.
        require_gripper_contact: Hold at the closed pose unless a finger is
            touching ``object_id`` when the lift phase begins.
    """
    if launch_passive is None:
        launch_passive = mujoco.viewer.launch_passive
    if sleep is None:
        sleep = time.sleep
    if clock is None:
        clock = time.time

    dt = float(mj_model.opt.timestep)
    if dt <= 0 or not np.isfinite(dt):
        dt = FALLBACK_TIMESTEP
    if trajectory_duration_s <= 0 or not np.isfinite(trajectory_duration_s):
        raise ValueError("trajectory_duration_s must be positive and finite")

    logger.info("Launching MuJoCo viewer...")
    viewer = launch_passive(mj_model, mj_data)
    start = clock()
    grasp_contact_confirmed = False
    last_contact_index = 0
    try:
        while True:
            is_running_fn = getattr(viewer, "is_running", None)
            if callable(is_running_fn) and not is_running_fn():
                break
            if max_duration_s is not None and clock() - start >= max_duration_s:
                break
            if lift_trajectory is not None:
                progress = (clock() - start) / trajectory_duration_s
                trajectory_index = min(int(progress * len(lift_trajectory)), len(lift_trajectory) - 1)
                if (
                    require_gripper_contact
                    and object_id is not None
                    and lift_start_index is not None
                    and trajectory_index >= lift_start_index
                ):
                    has_contact = _gripper_contacts_object(mj_model, mj_data, object_id)
                    if has_contact:
                        grasp_contact_confirmed = True
                        last_contact_index = trajectory_index
                    elif not grasp_contact_confirmed:
                        trajectory_index = lift_start_index - 1
                    else:
                        # Never continue a lift after physical contact is lost.
                        trajectory_index = last_contact_index
                q_current = lift_trajectory[trajectory_index]
                _hold_robot_pose(mj_model, mj_data, q_current, close_gripper=False)
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
            sleep(dt)
    finally:
        close = getattr(viewer, "close", None)
        if callable(close):
            close()
