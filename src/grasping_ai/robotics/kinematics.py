from collections.abc import Callable

import numpy as np

JointConfiguration = np.ndarray
RigidTransform = np.ndarray
ForwardKinematics = Callable[[JointConfiguration], RigidTransform]


def load_robot_model(robot_description_path: str) -> dict[str, object]:
    """Load a robot description from disk.

    Args:
        robot_description_path: Path to a robot description file such as an
            XML/MJCF robot definition.

    Returns:
        A dictionary describing robot kinematic and dynamic parameters.
    """
    raise NotImplementedError


def build_forward_kinematics(robot_model: dict[str, object]) -> ForwardKinematics:
    """Build a callable forward-kinematics function for a robot.

    Args:
        robot_model: Robot model returned by ``load_robot_model``.

    Returns:
        A callable mapping joint configurations to end-effector transforms.
    """
    raise NotImplementedError


def build_inverse_kinematics(
    robot_model: dict[str, object], max_iterations: int, tolerance: float
) -> Callable[..., np.ndarray]:
    """Build a numerical inverse-kinematics solver for a robot.

    Args:
        robot_model: Robot model returned by ``load_robot_model``.
        max_iterations: Maximum number of solver iterations.
        tolerance: Convergence tolerance on the pose error.

    Returns:
        A callable that accepts a target end-effector transform and an
        initial joint configuration and returns a solved joint configuration.
    """
    raise NotImplementedError


def solve_inverse_kinematics(
    ik_solver: Callable[..., np.ndarray],
    target_pose: RigidTransform,
    initial_joints: JointConfiguration,
) -> JointConfiguration:
    """Run inverse kinematics to find a joint configuration for a target pose.

    Args:
        ik_solver: Solver returned by ``build_inverse_kinematics``.
        target_pose: Desired end-effector pose as a ``(4, 4)`` transform.
        initial_joints: Initial joint configuration used to seed the solver.

    Returns:
        A joint configuration that achieves ``target_pose`` to within tolerance.
    """
    raise NotImplementedError
