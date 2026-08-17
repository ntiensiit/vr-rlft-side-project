"""Compose MuJoCo scenes from robot and object assets."""

from __future__ import annotations

import os
import re
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import mujoco  # type: ignore[import-untyped]
import numpy as np
from loguru import logger

from grasping_ai.simulation.mujoco_env import (
    ContactReporter,
    SimulationStep,
    create_simulation,
    load_mujoco_model,
    read_body_pose,
    set_actuator_controls,
)
from grasping_ai.utils.path_validation import (
    require_optional_path,
    require_path,
)

SceneCommand = Callable[[], None]


def set_freejoint_body_pose(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    body_name: str,
    position: np.ndarray,
    quaternion_wxyz: np.ndarray | None = None,
) -> None:
    """Place a freejoint body at a world-frame pose and update kinematics."""
    if position.shape != (3,) or not np.isfinite(position).all():
        msg = "position must be a finite array with shape (3,)"
        raise ValueError(msg)
    if quaternion_wxyz is not None and (
        quaternion_wxyz.shape != (4,) or not np.isfinite(quaternion_wxyz).all()
    ):
        msg = "quaternion_wxyz must be a finite array with shape (4,)"
        raise ValueError(msg)

    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body '{body_name}' not found in MuJoCo model")
    for joint_id in range(int(mj_model.njnt)):
        if (
            mj_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
            and int(mj_model.jnt_bodyid[joint_id]) == body_id
        ):
            qpos_address = int(mj_model.jnt_qposadr[joint_id])
            mj_data.qpos[qpos_address : qpos_address + 3] = position
            if quaternion_wxyz is not None:
                mj_data.qpos[qpos_address + 3 : qpos_address + 7] = quaternion_wxyz
            mujoco.mj_forward(mj_model, mj_data)
            return
    raise ValueError(f"Body '{body_name}' has no freejoint")


def place_freejoint_body_on_surface(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    body_name: str,
    support_geom_name: str = "table_top",
) -> float:
    """Place a freejoint body's lowest compiled geom point on a support geom."""
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    support_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, support_geom_name)
    if body_id == -1:
        raise ValueError(f"Body '{body_name}' not found in MuJoCo model")
    if support_id == -1:
        raise ValueError(f"Support geom '{support_geom_name}' not found in MuJoCo model")

    mujoco.mj_forward(mj_model, mj_data)
    support_rotation = mj_data.geom_xmat[support_id].reshape(3, 3)
    support_half_extents = mj_model.geom_aabb[support_id, 3:6]
    support_top = float(
        mj_data.geom_xpos[support_id, 2]
        + np.abs(support_rotation[2]) @ support_half_extents,
    )
    object_geom_ids = [
        geom_id
        for geom_id in range(int(mj_model.ngeom))
        if int(mj_model.geom_bodyid[geom_id]) == body_id
    ]
    if not object_geom_ids:
        raise ValueError(f"Body '{body_name}' has no geometry")

    def bottom_height() -> float:
        heights = []
        for geom_id in object_geom_ids:
            if mj_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = int(mj_model.geom_dataid[geom_id])
                vertex_start = int(mj_model.mesh_vertadr[mesh_id])
                vertex_count = int(mj_model.mesh_vertnum[mesh_id])
                vertices = mj_model.mesh_vert[vertex_start : vertex_start + vertex_count]
                rotation = mj_data.geom_xmat[geom_id].reshape(3, 3)
                world_vertices = vertices @ rotation.T + mj_data.geom_xpos[geom_id]
                heights.append(float(np.min(world_vertices[:, 2])))
                continue
            rotation = mj_data.geom_xmat[geom_id].reshape(3, 3)
            half_extents = mj_model.geom_aabb[geom_id, 3:6]
            world_half_z = np.abs(rotation[2]) @ half_extents
            heights.append(float(mj_data.geom_xpos[geom_id, 2] - world_half_z))
        return min(heights)

    position = np.array(mj_data.xpos[body_id], dtype=np.float64)
    position[2] += support_top - bottom_height()
    set_freejoint_body_pose(mj_model, mj_data, body_name, position)
    final_bottom = bottom_height()
    logger.info(
        "Placed '{}' on '{}' surface: bottom={:.6f}, support={:.6f}",
        body_name,
        support_geom_name,
        final_bottom,
        support_top,
    )
    return final_bottom


def _resolve_scene_output_dir(output_dir: Path | None) -> Path:
    """Return a writable directory for assembled scene XML artifacts.

    Args:
        output_dir: Optional explicit directory. When ``None``, uses a
            process-local folder under the system temp directory.

    Returns:
        Path to an existing writable output directory.
    """
    if output_dir is not None:
        validated = require_path(output_dir, "output_dir")
        validated.mkdir(parents=True, exist_ok=True)
        return validated

    temp_root = Path(tempfile.gettempdir()) / "grasping_ai_scenes"
    temp_root.mkdir(parents=True, exist_ok=True)
    return temp_root


def _xml_with_absolute_meshdir(xml_path: Path, out_dir: Path) -> Path:
    """Rewrite ``meshdir`` to an absolute path so includes from a temp scene resolve.

    MuJoCo interprets relative ``meshdir`` against the top-level XML, not the
    included file. Assembled scenes live in a temp directory, so robot meshes
    must be re-rooted to the original MJCF location.
    """
    text = xml_path.read_text(encoding="utf-8")

    def _replace(match: re.Match[str]) -> str:
        raw = match.group(1)
        meshdir = Path(raw)
        if not meshdir.is_absolute():
            meshdir = (xml_path.parent / meshdir).resolve()
        return f'meshdir="{meshdir.as_posix()}"'

    rewritten, count = re.subn(r'meshdir="([^"]*)"', _replace, text, count=1)
    if count == 0:
        return xml_path
    fd, path_str = tempfile.mkstemp(suffix="_absmesh.xml", dir=str(out_dir))
    os.close(fd)
    dest = Path(path_str)
    dest.write_text(rewritten, encoding="utf-8")
    return dest


def build_scene_xml(
    robot_xml_path: Path,
    object_xml_path: Path,
    table_xml_path: Path | None,
    object_name: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Assemble a MuJoCo scene XML file combining robot, object, and table.

    Args:
        robot_xml_path: Path to the robot MJCF description.
        object_xml_path: Path to the object MJCF description.
        table_xml_path: Optional path to a table/workbench MJCF description.
        object_name: Optional logical name assigned to the object body. When
            supplied, the first body of the object XML is renamed before
            inclusion; when ``None``, the object XML is included unchanged.
        output_dir: Optional directory for temporary assembled XML files.
            Defaults to a folder under the system temp directory.

    Returns:
        Path to the assembled scene XML file written to disk.
    """
    require_path(robot_xml_path, "robot_xml_path")
    require_path(object_xml_path, "object_xml_path")
    require_optional_path(table_xml_path, "table_xml_path")
    if object_name is not None and not isinstance(object_name, str):
        msg = "object_name must be a string or None"
        raise TypeError(msg)

    if not robot_xml_path.is_file():
        msg = f"Robot XML path '{robot_xml_path}' does not exist"
        raise FileNotFoundError(msg)
    if not object_xml_path.is_file():
        msg = f"Object XML path '{object_xml_path}' does not exist"
        raise FileNotFoundError(msg)
    if table_xml_path is not None and not table_xml_path.is_file():
        msg = f"Table XML path '{table_xml_path}' does not exist"
        raise FileNotFoundError(msg)

    logger.info(
        "Assembling scene XML with robot: {}, object: {} (name={}), table: {}",
        robot_xml_path,
        object_xml_path,
        object_name,
        table_xml_path,
    )
    out_dir = _resolve_scene_output_dir(output_dir)

    included_object_path = object_xml_path
    if object_name is not None:
        included_object_path = _rename_object_body(object_xml_path, object_name, out_dir)

    robot_for_include = _xml_with_absolute_meshdir(robot_xml_path, out_dir)

    fd, path_str = tempfile.mkstemp(suffix="_scene.xml", dir=str(out_dir))
    os.close(fd)
    path = Path(path_str)

    lines = [
        '<mujoco model="assembled_scene">',
        f'    <include file="{robot_for_include.resolve().as_posix()}"/>',
        f'    <include file="{included_object_path.resolve().as_posix()}"/>',
    ]
    if table_xml_path is not None:
        lines.append(f'    <include file="{table_xml_path.resolve().as_posix()}"/>')
    lines.append("</mujoco>")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _rename_object_body(object_xml_path: Path, object_name: str, out_dir: Path) -> Path:
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
        # Only local project MJCF assets are parsed here, not untrusted XML.
        tree = ET.parse(object_xml_path)  # noqa: S314
        root = tree.getroot()
    except Exception as e:
        msg = f"Failed to parse object XML file '{object_xml_path}': {e}"
        raise ValueError(msg) from e

    body_renamed = False
    for body in root.findall(".//body"):
        body.set("name", object_name)
        body_renamed = True
        break

    if not body_renamed:
        msg = f"No body element found in object XML '{object_xml_path}' to rename"
        raise ValueError(msg)

    fd, obj_path_str = tempfile.mkstemp(suffix=f"_{object_name}.xml", dir=str(out_dir))
    os.close(fd)
    modified_object_xml_path = Path(obj_path_str)
    tree.write(modified_object_xml_path, encoding="utf-8", xml_declaration=True)
    return modified_object_xml_path


def attach_object_to_scene(
    state: object,
    object_xml_path: Path,
    object_name: str,
    output_dir: Path | None = None,
) -> None:
    """Attach a YCB object into an existing simulation scene.

    Args:
        state: Opaque state handle returned by ``create_simulation``.
        object_xml_path: Path to the object MJCF description.
        object_name: Logical name to assign to the attached object.
        output_dir: Optional directory for temporary assembled XML files.
    """
    if not isinstance(state, dict) or "model" not in state or "data" not in state:
        msg = "state must be a simulation state dictionary"
        raise TypeError(msg)
    require_path(object_xml_path, "object_xml_path")
    if not isinstance(object_name, str):
        msg = "object_name must be a string"
        raise TypeError(msg)

    state_dict = cast("dict[str, Any]", state)

    if not object_xml_path.is_file():
        msg = f"Object XML file '{object_xml_path}' does not exist"
        raise FileNotFoundError(msg)

    out_dir = _resolve_scene_output_dir(output_dir)
    modified_object_xml_path = _rename_object_body(object_xml_path, object_name, out_dir)
    state_dict["attached_xml_paths"].append(modified_object_xml_path)

    fd_scene, scene_path_str = tempfile.mkstemp(suffix="_scene.xml", dir=str(out_dir))
    os.close(fd_scene)
    new_scene_xml_path = Path(scene_path_str)

    lines = ['<mujoco model="assembled_scene">']
    original_xml = state_dict.get("model_xml_path")
    if original_xml is not None:
        robot_for_include = _xml_with_absolute_meshdir(original_xml, out_dir)
        lines.append(f'    <include file="{robot_for_include.resolve().as_posix()}"/>')
    lines.extend(
        [f'    <include file="{path.resolve().as_posix()}"/>' for path in state_dict["attached_xml_paths"]],
    )
    lines.append("</mujoco>")

    new_scene_xml_path.write_text("\n".join(lines), encoding="utf-8")

    try:
        new_model: Any = mujoco.MjModel.from_xml_path(str(new_scene_xml_path))
        new_data: Any = mujoco.MjData(new_model)
    except Exception as e:
        msg = f"Failed to reload MuJoCo simulation after attaching object: {e}"
        raise ValueError(msg) from e

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
        msg = "step must be a callable (SimulationStep)"
        raise TypeError(msg)
    if not isinstance(dt, (int, float, np.floating, np.integer)):
        msg = "dt must be a float or integer"
        raise TypeError(msg)
    if not isinstance(num_steps, (int, np.integer)):
        msg = "num_steps must be an integer"
        raise TypeError(msg)
    if dt <= 0 or not np.isfinite(dt):
        msg = "dt must be a positive finite number"
        raise ValueError(msg)
    if num_steps <= 0:
        msg = "num_steps must be a positive integer"
        raise ValueError(msg)

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
        msg = "contacts must be a callable (ContactReporter)"
        raise TypeError(msg)
    if not isinstance(body_names, set):
        msg = "body_names must be a set"
        raise TypeError(msg)
    for name in body_names:
        if not isinstance(name, str):
            msg = "All elements in body_names must be strings"
            raise TypeError(msg)

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
        scene_output_dir: Optional directory for temporary assembled XML files.

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
        scene_output_dir: Path | None = None,
    ) -> None:
        """Initialize the scene by composing the supplied MJCF files."""
        require_path(robot_xml_path, "robot_xml_path")
        require_optional_path(object_xml_path, "object_xml_path")
        require_optional_path(table_xml_path, "table_xml_path")
        if object_name is not None and not isinstance(object_name, str):
            msg = "object_name must be a string or None"
            raise TypeError(msg)
        require_optional_path(scene_output_dir, "scene_output_dir")

        self.robot_xml_path = robot_xml_path
        self.object_xml_path = object_xml_path
        self.table_xml_path = table_xml_path
        self.object_name = object_name
        self._scene_output_dir = scene_output_dir

        scene_path = self._resolve_scene_path()
        self._model_handle = load_mujoco_model(scene_path)
        self._state, self._step_fn, self._contacts_fn = create_simulation(self._model_handle)
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
            output_dir=self._scene_output_dir,
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
    def model(self) -> mujoco.MjModel:
        """Return the underlying MuJoCo model."""
        return cast("dict[str, Any]", self._state)["model"]

    @property
    def data(self) -> mujoco.MjData:
        """Return the underlying MuJoCo data."""
        return cast("dict[str, Any]", self._state)["data"]

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
        attach_object_to_scene(
            self._state,
            object_xml_path,
            object_name,
            output_dir=self._scene_output_dir,
        )
        self._snapshot = self._capture_state()
