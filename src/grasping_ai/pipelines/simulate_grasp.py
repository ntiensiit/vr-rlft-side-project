from pathlib import Path

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
    raise NotImplementedError


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
    raise NotImplementedError
