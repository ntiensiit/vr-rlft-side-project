from collections.abc import Callable

import numpy as np

GripperCommand = np.ndarray


def load_gripper_model(gripper_description_path: str) -> dict[str, object]:
    """Load a gripper description from disk.

    Args:
        gripper_description_path: Path to a gripper description file such as an
            XML/MJCF gripper definition.

    Returns:
        A dictionary describing the gripper kinematic and dynamic parameters.
    """
    raise NotImplementedError


def build_gripper_controller(gripper_model: dict[str, object]) -> Callable[[GripperCommand], None]:
    """Build a callable gripper controller that issues open/close commands.

    Args:
        gripper_model: Gripper model returned by ``load_gripper_model``.

    Returns:
        A callable accepting a gripper command vector and applying it to the
        underlying simulation environment.
    """
    raise NotImplementedError


def make_open_command(gripper_model: dict[str, object]) -> GripperCommand:
    """Build an "open" gripper command for the supplied gripper model.

    Args:
        gripper_model: Gripper model returned by ``load_gripper_model``.

    Returns:
        A gripper command vector that fully opens the gripper.
    """
    raise NotImplementedError


def make_close_command(gripper_model: dict[str, object]) -> GripperCommand:
    """Build a "close" gripper command for the supplied gripper model.

    Args:
        gripper_model: Gripper model returned by ``load_gripper_model``.

    Returns:
        A gripper command vector that closes the gripper.
    """
    raise NotImplementedError
