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

    Returns:
        A dictionary describing the simulation outcome, including success flag
        and any recorded contact or trajectory information.
    """
    if grasp_pose.shape != (4, 4):
        raise ValueError(f"grasp_pose must have shape (4, 4), got {grasp_pose.shape}")
    if not isinstance(robot_xml_path, Path) or not robot_xml_path.is_file():
        raise FileNotFoundError(f"robot_xml_path not found: {robot_xml_path}")
    if not isinstance(ycb_root, Path) or not ycb_root.is_dir():
        raise FileNotFoundError(f"ycb_root not found: {ycb_root}")
    if num_simulation_steps <= 0:
        raise ValueError("num_simulation_steps must be positive")

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

    # Set gripper command
    nu_robot = cast(int, mj_model.nu)
    ctrl_cmd = np.zeros(nu_robot, dtype=np.float64)
    close_len = gripper_close_command.shape[0]
    if close_len <= nu_robot:
        ctrl_cmd[:close_len] = gripper_close_command
    else:
        ctrl_cmd[:] = gripper_close_command[:nu_robot]

    for _ in range(num_simulation_steps):
        scene.step(ctrl_cmd, dt)

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

    # Success criteria: object hasn't fallen and has contacts
    success = bool(final_height >= (initial_height - 0.05) and contact_count >= 1)

    return {
        "success": success,
        "initial_height": initial_height,
        "final_height": final_height,
        "contact_count": float(contact_count),
        "object_velocity": object_velocity,
        "grasp_pose": grasp_pose,
    }


def run_simulation_sweep(
    grasp_poses: np.ndarray,
    object_id: str,
    ycb_root: Path,
    robot_xml_path: Path,
    table_xml_path: Path | None,
    num_simulation_steps: int,
    gripper_close_command: np.ndarray,
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
        )
        outcomes.append(outcome)

    return outcomes
