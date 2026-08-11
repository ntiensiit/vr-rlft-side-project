from collections.abc import Callable
from pathlib import Path

import numpy as np

SimulationStep = Callable[[float], None]
ContactReporter = Callable[[], list[dict[str, np.ndarray]]]


def load_mujoco_model(model_xml_path: Path) -> object:
    """Load a MuJoCo simulation model from an XML file.

    Args:
        model_xml_path: Path to the MuJoCo MJCF XML description.

    Returns:
        An opaque simulation model object usable by other simulation helpers.
    """
    raise NotImplementedError


def create_simulation(model: object) -> tuple[object, SimulationStep, ContactReporter]:
    """Create a stepping interface over a MuJoCo model.

    Args:
        model: MuJoCo model returned by ``load_mujoco_model``.

    Returns:
        A tuple ``(state, step, contacts)`` where ``state`` is an opaque state
        handle, ``step`` advances the simulation by a time increment, and
        ``contacts`` reports the current contact information.
    """
    raise NotImplementedError


def reset_simulation(state: object) -> None:
    """Reset the simulation state to its initial configuration.

    Args:
        state: Opaque state handle returned by ``create_simulation``.
    """
    raise NotImplementedError


def read_joint_positions(state: object) -> np.ndarray:
    """Read the current joint positions from the simulation state.

    Args:
        state: Opaque state handle returned by ``create_simulation``.

    Returns:
        Joint position vector with shape ``(num_joints,)``.
    """
    raise NotImplementedError


def set_joint_positions(state: object, positions: np.ndarray) -> None:
    """Write joint positions into the simulation state.

    Args:
        state: Opaque state handle returned by ``create_simulation``.
        positions: Joint position vector with shape ``(num_joints,)``.
    """
    raise NotImplementedError


def read_body_pose(state: object, body_name: str) -> np.ndarray:
    """Read the world-frame pose of a named body.

    Args:
        state: Opaque state handle returned by ``create_simulation``.
        body_name: Name of the body whose pose should be read.

    Returns:
        A ``(4, 4)`` transformation matrix representing the body pose.
    """
    raise NotImplementedError
