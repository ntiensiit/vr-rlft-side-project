import os
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import mujoco  # type: ignore[import-untyped]
import numpy as np

from grasping_ai.simulation.mujoco_env import (
    ContactReporter,
    SimulationStep,
    create_simulation,
    load_mujoco_model,
    read_body_pose,
    set_actuator_controls,
)

SceneCommand = Callable[[], None]


def build_scene_xml(
    robot_xml_path: Path,
    object_xml_path: Path,
    table_xml_path: Path | None,
    object_name: str | None = None,
) -> Path:
    """Assemble a MuJoCo scene XML file combining robot, object, and table.

    Args:
        robot_xml_path: Path to the robot MJCF description.
        object_xml_path: Path to the object MJCF description.
        table_xml_path: Optional path to a table/workbench MJCF description.
        object_name: Optional logical name assigned to the object body. When
            supplied, the first body of the object XML is renamed before
            inclusion; when ``None``, the object XML is included unchanged.

    Returns:
        Path to the assembled scene XML file written to disk.
    """
    if not isinstance(robot_xml_path, Path):
        raise TypeError("robot_xml_path must be a pathlib.Path instance")
    if not isinstance(object_xml_path, Path):
        raise TypeError("object_xml_path must be a pathlib.Path instance")
    if table_xml_path is not None and not isinstance(table_xml_path, Path):
        raise TypeError("table_xml_path must be a pathlib.Path instance or None")
    if object_name is not None and not isinstance(object_name, str):
        raise TypeError("object_name must be a string or None")

    if not robot_xml_path.is_file():
        raise FileNotFoundError(f"Robot XML path '{robot_xml_path}' does not exist")
    if not object_xml_path.is_file():
        raise FileNotFoundError(f"Object XML path '{object_xml_path}' does not exist")
    if table_xml_path is not None and not table_xml_path.is_file():
        raise FileNotFoundError(f"Table XML path '{table_xml_path}' does not exist")

    out_dir = Path("data/interim")
    out_dir.mkdir(parents=True, exist_ok=True)

    included_object_path = object_xml_path
    if object_name is not None:
        included_object_path = _rename_object_body(
            object_xml_path, object_name, out_dir
        )

    fd, path_str = tempfile.mkstemp(suffix="_scene.xml", dir=str(out_dir))
    os.close(fd)
    path = Path(path_str)

    lines = [
        '<mujoco model="assembled_scene">',
        f'    <include file="{robot_xml_path.resolve().as_posix()}"/>',
        f'    <include file="{included_object_path.resolve().as_posix()}"/>',
    ]
    if table_xml_path is not None:
        lines.append(f'    <include file="{table_xml_path.resolve().as_posix()}"/>')
    lines.append('</mujoco>')

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _rename_object_body(
    object_xml_path: Path, object_name: str, out_dir: Path
) -> Path:
    """Write a copy of an object XML with its first body renamed.

    Args:
        object_xml_path: Path to the object MJCF description.
        object_name: Logical name to assign to the first body element.
        out_dir: Directory in which the renamed object XML is written.

    Returns:
        Path to the modified object XML file written to disk.

    Raises:
        ValueError: If the object XML cannot be parsed or contains no body.
    """
    try:
        tree = ET.parse(object_xml_path)
        root = tree.getroot()
    except Exception as e:
        raise ValueError(
            f"Failed to parse object XML file '{object_xml_path}': {e}"
        ) from e

    body_renamed = False
    for body in root.findall(".//body"):
        body.set("name", object_name)
        body_renamed = True
        break

    if not body_renamed:
        raise ValueError(
            f"No body element found in object XML '{object_xml_path}' to rename"
        )

    fd, obj_path_str = tempfile.mkstemp(
        suffix=f"_{object_name}.xml", dir=str(out_dir)
    )
    os.close(fd)
    modified_object_xml_path = Path(obj_path_str)
    tree.write(modified_object_xml_path, encoding="utf-8", xml_declaration=True)
    return modified_object_xml_path


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


class MuJoCoScene:
    """Composed MuJoCo scene with snapshot-based reset.

    The scene is assembled once from robot, optional object, and optional
    table MJCF files. Resets restore a captured state snapshot instead of
    reconstructing the model from XML on every trial, and attached objects
    keep their logical names so body lookups stay stable.

    Args:
        robot_xml_path: Path to the robot MJCF description.
        object_xml_path: Optional path to the object MJCF description.
        table_xml_path: Optional path to a table/workbench MJCF description.
        object_name: Optional logical name assigned to the object body.

    Raises:
        FileNotFoundError: If any supplied MJCF path does not exist.
        ValueError: If the object XML contains no body element.
    """

    def __init__(
        self,
        robot_xml_path: Path,
        object_xml_path: Path | None = None,
        table_xml_path: Path | None = None,
        object_name: str | None = None,
    ) -> None:
        """Initialize the scene by composing the supplied MJCF files."""
        if not isinstance(robot_xml_path, Path):
            raise TypeError("robot_xml_path must be a pathlib.Path instance")
        if object_xml_path is not None and not isinstance(object_xml_path, Path):
            raise TypeError("object_xml_path must be a pathlib.Path instance or None")
        if table_xml_path is not None and not isinstance(table_xml_path, Path):
            raise TypeError("table_xml_path must be a pathlib.Path instance or None")
        if object_name is not None and not isinstance(object_name, str):
            raise TypeError("object_name must be a string or None")

        self.robot_xml_path = robot_xml_path
        self.object_xml_path = object_xml_path
        self.table_xml_path = table_xml_path
        self.object_name = object_name

        scene_path = self._resolve_scene_path()
        self._model_handle = load_mujoco_model(scene_path)
        self._state, self._step_fn, self._contacts_fn = create_simulation(
            self._model_handle
        )
        self._snapshot = self._capture_state()

    def _resolve_scene_path(self) -> Path:
        """Return the assembled scene path or the bare robot XML."""
        if self.object_xml_path is None:
            return self.robot_xml_path
        return build_scene_xml(
            self.robot_xml_path,
            self.object_xml_path,
            self.table_xml_path,
            object_name=self.object_name,
        )

    def _capture_state(self) -> dict[str, np.ndarray]:
        """Snapshot the current simulation state."""
        data = self.data
        return {
            "qpos": np.array(data.qpos, copy=True),
            "qvel": np.array(data.qvel, copy=True),
            "ctrl": np.array(data.ctrl, copy=True),
            "time": np.array([data.time]),
        }

    @property
    def state(self) -> object:
        """Return the opaque simulation state handle."""
        return self._state

    @property
    def model(self) -> Any:
        """Return the underlying MuJoCo model."""
        return cast(dict[str, Any], self._state)["model"]

    @property
    def data(self) -> Any:
        """Return the underlying MuJoCo data."""
        return cast(dict[str, Any], self._state)["data"]

    def reset(self) -> None:
        """Restore the captured initial state without rebuilding the model."""
        snapshot = self._snapshot
        data = self.data
        data.qpos[:] = snapshot["qpos"]
        data.qvel[:] = snapshot["qvel"]
        data.ctrl[:] = snapshot["ctrl"]
        data.time = float(snapshot["time"][0])
        mujoco.mj_forward(self.model, data)

    def step(self, ctrl: np.ndarray, dt: float) -> None:
        """Advance the simulation by one control step.

        Args:
            ctrl: Actuator control vector with shape ``(num_actuators,)``.
            dt: Simulation time step in seconds.
        """
        set_actuator_controls(self._state, ctrl)
        self._step_fn(dt)

    def body_pose(self, body_name: str) -> np.ndarray:
        """Return the world-frame pose of a named body.

        Args:
            body_name: Name of the body whose pose should be read.

        Returns:
            A ``(4, 4)`` transformation matrix representing the body pose.
        """
        return read_body_pose(self._state, body_name)

    def contacts(self) -> list[dict[str, np.ndarray]]:
        """Return the current contact reports of the scene."""
        return self._contacts_fn()

    def attach_object(self, object_xml_path: Path, object_name: str) -> None:
        """Attach an object into the scene and refresh the reset snapshot.

        Args:
            object_xml_path: Path to the object MJCF description.
            object_name: Logical name assigned to the attached object.
        """
        attach_object_to_scene(self._state, object_xml_path, object_name)
        self._snapshot = self._capture_state()
