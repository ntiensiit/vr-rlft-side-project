from collections.abc import Callable
from pathlib import Path

from grasping_ai.simulation.mujoco_env import ContactReporter, SimulationStep

SceneCommand = Callable[[], None]


def build_scene_xml(
    robot_xml_path: Path, object_xml_path: Path, table_xml_path: Path | None
) -> Path:
    """Assemble a MuJoCo scene XML file combining robot, object, and table.

    Args:
        robot_xml_path: Path to the robot MJCF description.
        object_xml_path: Path to the object MJCF description.
        table_xml_path: Optional path to a table/workbench MJCF description.

    Returns:
        Path to the assembled scene XML file written to disk.
    """
    raise NotImplementedError


def attach_object_to_scene(state: object, object_xml_path: Path, object_name: str) -> None:
    """Attach a YCB object into an existing simulation scene.

    Args:
        state: Opaque state handle returned by ``create_simulation``.
        object_xml_path: Path to the object MJCF description.
        object_name: Logical name to assign to the attached object.
    """
    raise NotImplementedError


def step_scene(step: SimulationStep, dt: float, num_steps: int) -> None:
    """Advance a scene by a fixed number of simulation steps.

    Args:
        step: Stepping callable returned by ``create_simulation``.
        dt: Simulation time step in seconds.
        num_steps: Number of steps to execute.
    """
    raise NotImplementedError


def collect_contacts(contacts: ContactReporter, body_names: set[str]) -> list[dict[str, object]]:
    """Filter contact reports to only those involving the supplied body names.

    Args:
        contacts: Contact reporter returned by ``create_simulation``.
        body_names: Set of body names whose contacts should be retained.

    Returns:
        A list of filtered contact records.
    """
    raise NotImplementedError
