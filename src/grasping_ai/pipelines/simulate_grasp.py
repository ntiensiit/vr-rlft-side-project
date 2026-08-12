from pathlib import Path
from typing import cast

import numpy as np

SimulationOutcome = dict[str, np.ndarray | bool | float]


def simulate_grasp(
    grasp_pose: np.ndarray,
    object_id: str,
    ycb_root: Path,
    robot_xml_path: Path,
    table_xml_path: Path | None,
    num_simulation_steps: int,
    gripper_close_command: np.ndarray,
    lift_height_threshold: float = 0.05,
    max_linear_velocity: float = 0.05,
    max_angular_velocity: float = 0.1,
) -> SimulationOutcome:
    """Execute a single grasp in a MuJoCo simulation and report its outcome.

    Args:
        grasp_pose: Grasp pose expressed in the world frame as a ``(4, 4)``
            transformation.
        object_id: Logical YCB object identifier to load.
        ycb_root: Root directory of the YCB object set.
        robot_xml_path: Path to the robot MJCF description.
        table_xml_path: Optional path to a workbench/table MJCF description.
        num_simulation_steps: Number of physics steps to execute.
        gripper_close_command: Gripper command used to close the gripper.
        lift_height_threshold: Minimum world-frame height gain required to
            count the grasp as a successful lift.
        max_linear_velocity: Maximum acceptable linear velocity of the object.
        max_angular_velocity: Maximum acceptable angular velocity of the object.

    Returns:
        A dictionary describing the simulation outcome, including the success
        flag and any recorded contact, velocity, or trajectory information.
    """
    if grasp_pose.shape != (4, 4):
        raise ValueError(f"grasp_pose must have shape (4, 4), got {grasp_pose.shape}")
    if not isinstance(robot_xml_path, Path) or not robot_xml_path.is_file():
        raise FileNotFoundError(f"robot_xml_path not found: {robot_xml_path}")
    if not isinstance(ycb_root, Path) or not ycb_root.is_dir():
        raise FileNotFoundError(f"ycb_root not found: {ycb_root}")
    if num_simulation_steps <= 0:
        raise ValueError("num_simulation_steps must be positive")
    if lift_height_threshold < 0:
        raise ValueError("lift_height_threshold must be non-negative")
    if max_linear_velocity < 0:
        raise ValueError("max_linear_velocity must be non-negative")
    if max_angular_velocity < 0:
        raise ValueError("max_angular_velocity must be non-negative")

    # Resolve YCB object XML
    from grasping_ai.simulation.ycb import find_ycb_mjcf, resolve_ycb_object_directory
    object_dir = resolve_ycb_object_directory(ycb_root, object_id)
    object_xml_path = find_ycb_mjcf(object_dir)

    # Compose the scene and load the model
    from grasping_ai.simulation.scene import MuJoCoScene

    scene = MuJoCoScene(
        robot_xml_path,
        object_xml_path,
        table_xml_path,
        object_name=object_id,
    )
    mj_model = scene.model
    mj_data = scene.data

    # Run inverse kinematics to match grasp_pose
    from grasping_ai.robotics.kinematics import (
        build_inverse_kinematics,
        load_robot_model,
        solve_inverse_kinematics,
    )
    robot_model = load_robot_model(str(robot_xml_path))
    nq_robot = cast(int, robot_model["nq"])
    ik_solver = build_inverse_kinematics(robot_model, max_iterations=200, tolerance=1e-4)
    initial_joints = np.zeros(nq_robot)

    try:
        q_target = solve_inverse_kinematics(ik_solver, grasp_pose, initial_joints)
    except ValueError:
        return {
            "success": False,
            "initial_height": 0.0,
            "final_height": 0.0,
            "contact_count": 0.0,
            "object_velocity": np.zeros(6),
            "grasp_pose": grasp_pose,
            "fk_position_error": float("inf"),
        }

    # Teleport robot to grasp pose in simulation
    scene.reset()

    # Copy IK solution to robot joints in simulation state
    if mj_data.qpos.shape[0] >= nq_robot:
        mj_data.qpos[:nq_robot] = q_target
    import mujoco  # type: ignore[import-untyped]
    mujoco.mj_forward(mj_model, mj_data)

    # Record initial height of the object
    initial_pose = scene.body_pose(object_id)
    initial_height = float(initial_pose[2, 3])

    # Step simulation and apply gripper close command
    dt = mj_model.opt.timestep
    if dt <= 0 or not np.isfinite(dt):
        dt = 0.002

    from grasping_ai.robotics.gripper import (
        load_gripper_model,
        make_close_command,
        make_open_command,
    )
    from grasping_ai.robotics.kinematics import build_forward_kinematics
    from grasping_ai.simulation.scene import step_scene

    gripper_model = load_gripper_model(str(robot_xml_path))
    gripper_model["model"] = mj_model
    gripper_model["data"] = mj_data
    nu_robot = cast(int, mj_model.nu)

    def _fit_actuator_command(raw_cmd: np.ndarray) -> np.ndarray:
        ctrl_cmd = raw_cmd.astype(np.float64)
        if nu_robot < ctrl_cmd.shape[0]:
            return ctrl_cmd[:nu_robot]
        if nu_robot > ctrl_cmd.shape[0]:
            padded = np.zeros(nu_robot, dtype=np.float64)
            padded[: ctrl_cmd.shape[0]] = ctrl_cmd
            return padded
        return ctrl_cmd

    open_cmd = _fit_actuator_command(make_open_command(gripper_model))
    close_cmd = _fit_actuator_command(make_close_command(gripper_model))
    close_len = gripper_close_command.shape[0]
    if close_len > 0:
        overlay_len = min(close_len, nu_robot)
        close_cmd[:overlay_len] = gripper_close_command[:overlay_len]

    pre_grasp_steps = min(10, max(1, num_simulation_steps // 4))
    close_steps = max(1, num_simulation_steps - pre_grasp_steps)

    def _advance_open(step_dt: float) -> None:
        scene.step(open_cmd, step_dt)

    def _advance_close(step_dt: float) -> None:
        scene.step(close_cmd, step_dt)

    step_scene(_advance_open, dt, pre_grasp_steps)
    step_scene(_advance_close, dt, close_steps)

    fk_solver = build_forward_kinematics(robot_model)
    achieved_pose = fk_solver(q_target)
    fk_position_error = float(np.linalg.norm(achieved_pose[:3, 3] - grasp_pose[:3, 3]))

    # Read final height and velocities of the object
    final_pose = scene.body_pose(object_id)
    final_height = float(final_pose[2, 3])

    # Read velocity
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, object_id)
    if body_id == -1:
        raise ValueError(f"Body '{object_id}' not found in simulation model")
    object_velocity = np.array(mj_data.cvel[body_id], copy=True)

    # Read contacts
    from grasping_ai.simulation.scene import collect_contacts
    object_contacts = collect_contacts(scene.contacts, {object_id})
    contact_count = len(object_contacts)

    # Success requires a genuine lift: stable contact plus the object raised
    # above the lift threshold with bounded object velocity.
    from grasping_ai.evaluation.metrics import (
        build_lift_outcome_judge,
        build_stability_judge,
        evaluate_lift_success,
        evaluate_stability,
    )
    lift_judge = build_lift_outcome_judge(lift_height_threshold)
    stability_judge = build_stability_judge(max_linear_velocity, max_angular_velocity)
    lifted = evaluate_lift_success(lift_judge, initial_height, final_height)
    stable = evaluate_stability(stability_judge, object_velocity)
    success = bool(contact_count >= 1 and lifted and stable)

    return {
        "success": success,
        "initial_height": initial_height,
        "final_height": final_height,
        "contact_count": float(contact_count),
        "object_velocity": object_velocity,
        "grasp_pose": grasp_pose,
        "fk_position_error": fk_position_error,
    }


def run_simulation_sweep(
    grasp_poses: np.ndarray,
    object_id: str,
    ycb_root: Path,
    robot_xml_path: Path,
    table_xml_path: Path | None,
    num_simulation_steps: int,
    gripper_close_command: np.ndarray,
    lift_height_threshold: float = 0.05,
    max_linear_velocity: float = 0.05,
    max_angular_velocity: float = 0.1,
) -> list[SimulationOutcome]:
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

    Returns:
        A list of per-grasp simulation outcomes.
    """
    if grasp_poses.ndim == 2:
        if grasp_poses.shape == (4, 4):
            grasp_poses = grasp_poses.reshape(1, 4, 4)
        else:
            raise ValueError("grasp_poses must have shape (K, 4, 4) or (4, 4)")

    if grasp_poses.ndim != 3 or grasp_poses.shape[1:] != (4, 4):
        raise ValueError("grasp_poses must have shape (K, 4, 4)")

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
