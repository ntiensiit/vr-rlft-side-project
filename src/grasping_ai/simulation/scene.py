import os
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import mujoco  # type: ignore[import-untyped]
import numpy as np

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
    if not isinstance(robot_xml_path, Path):
        raise TypeError("robot_xml_path must be a pathlib.Path instance")
    if not isinstance(object_xml_path, Path):
        raise TypeError("object_xml_path must be a pathlib.Path instance")
    if table_xml_path is not None and not isinstance(table_xml_path, Path):
        raise TypeError("table_xml_path must be a pathlib.Path instance or None")

    if not robot_xml_path.is_file():
        raise FileNotFoundError(f"Robot XML path '{robot_xml_path}' does not exist")
    if not object_xml_path.is_file():
        raise FileNotFoundError(f"Object XML path '{object_xml_path}' does not exist")
    if table_xml_path is not None and not table_xml_path.is_file():
        raise FileNotFoundError(f"Table XML path '{table_xml_path}' does not exist")

    out_dir = Path("data/interim")
    out_dir.mkdir(parents=True, exist_ok=True)

    fd, path_str = tempfile.mkstemp(suffix="_scene.xml", dir=str(out_dir))
    os.close(fd)
    path = Path(path_str)

    lines = [
        '<mujoco model="assembled_scene">',
        f'    <include file="{robot_xml_path.resolve().as_posix()}"/>',
        f'    <include file="{object_xml_path.resolve().as_posix()}"/>',
    ]
    if table_xml_path is not None:
        lines.append(f'    <include file="{table_xml_path.resolve().as_posix()}"/>')
    lines.append('</mujoco>')

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def attach_object_to_scene(state: object, object_xml_path: Path, object_name: str) -> None:
    """Attach a YCB object into an existing simulation scene.

    Args:
        state: Opaque state handle returned by ``create_simulation``.
        object_xml_path: Path to the object MJCF description.
        object_name: Logical name to assign to the attached object.
    """
    if not isinstance(state, dict) or "model" not in state or "data" not in state:
        raise TypeError("state must be a simulation state dictionary")
    if not isinstance(object_xml_path, Path):
        raise TypeError("object_xml_path must be a pathlib.Path instance")
    if not isinstance(object_name, str):
        raise TypeError("object_name must be a string")

    state_dict = cast(dict[str, Any], state)

    if not object_xml_path.is_file():
        raise FileNotFoundError(f"Object XML file '{object_xml_path}' does not exist")

    try:
        tree = ET.parse(object_xml_path)
        root = tree.getroot()
    except Exception as e:
        raise ValueError(f"Failed to parse object XML file '{object_xml_path}': {e}") from e

    body_renamed = False
    for body in root.findall(".//body"):
        body.set("name", object_name)
        body_renamed = True
        break

    if not body_renamed:
        raise ValueError(f"No body element found in object XML '{object_xml_path}' to rename")

    out_dir = Path("data/interim")
    out_dir.mkdir(parents=True, exist_ok=True)

    fd_obj, obj_path_str = tempfile.mkstemp(suffix=f"_{object_name}.xml", dir=str(out_dir))
    os.close(fd_obj)
    modified_object_xml_path = Path(obj_path_str)
    tree.write(modified_object_xml_path, encoding="utf-8", xml_declaration=True)

    state_dict["attached_xml_paths"].append(modified_object_xml_path)

    fd_scene, scene_path_str = tempfile.mkstemp(suffix="_scene.xml", dir=str(out_dir))
    os.close(fd_scene)
    new_scene_xml_path = Path(scene_path_str)

    lines = ['<mujoco model="assembled_scene">']
    original_xml = state_dict.get("model_xml_path")
    if original_xml is not None:
        lines.append(f'    <include file="{original_xml.resolve().as_posix()}"/>')
    for path in state_dict["attached_xml_paths"]:
        lines.append(f'    <include file="{path.resolve().as_posix()}"/>')
    lines.append('</mujoco>')

    new_scene_xml_path.write_text("\n".join(lines), encoding="utf-8")

    try:
        new_model: Any = mujoco.MjModel.from_xml_path(str(new_scene_xml_path))
        new_data: Any = mujoco.MjData(new_model)
    except Exception as e:
        raise ValueError(f"Failed to reload MuJoCo simulation after attaching object: {e}") from e

    old_model: Any = state_dict["model"]
    old_data: Any = state_dict["data"]

    nq_to_copy = min(old_model.nq, new_model.nq)
    nv_to_copy = min(old_model.nv, new_model.nv)
    nu_to_copy = min(old_model.nu, new_model.nu)

    new_data.qpos[:nq_to_copy] = old_data.qpos[:nq_to_copy]
    new_data.qvel[:nv_to_copy] = old_data.qvel[:nv_to_copy]
    new_data.ctrl[:nu_to_copy] = old_data.ctrl[:nu_to_copy]

    mujoco.mj_forward(new_model, new_data)

    state_dict["model"] = new_model
    state_dict["data"] = new_data


def step_scene(step: SimulationStep, dt: float, num_steps: int) -> None:
    """Advance a scene by a fixed number of simulation steps.

    Args:
        step: Stepping callable returned by ``create_simulation``.
        dt: Simulation time step in seconds.
        num_steps: Number of steps to execute.
    """
    if not callable(step):
        raise TypeError("step must be a callable (SimulationStep)")
    if not isinstance(dt, (int, float, np.floating, np.integer)):
        raise TypeError("dt must be a float or integer")
    if not isinstance(num_steps, (int, np.integer)):
        raise TypeError("num_steps must be an integer")
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("dt must be a positive finite number")
    if num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")

    for _ in range(num_steps):
        step(dt)


def collect_contacts(contacts: ContactReporter, body_names: set[str]) -> list[dict[str, object]]:
    """Filter contact reports to only those involving the supplied body names.

    Args:
        contacts: Contact reporter returned by ``create_simulation``.
        body_names: Set of body names whose contacts should be retained.

    Returns:
        A list of filtered contact records.
    """
    if not callable(contacts):
        raise TypeError("contacts must be a callable (ContactReporter)")
    if not isinstance(body_names, set):
        raise TypeError("body_names must be a set")
    for name in body_names:
        if not isinstance(name, str):
            raise TypeError("All elements in body_names must be strings")

    all_contacts = contacts()
    filtered: list[dict[str, object]] = []
    for c in all_contacts:
        c_body_names = set(c["body_names"])
        if c_body_names.intersection(body_names):
            filtered.append(dict(c))
    return filtered
