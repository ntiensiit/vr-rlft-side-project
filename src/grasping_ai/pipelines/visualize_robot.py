"""MuJoCo robot viewer."""

from __future__ import annotations

from grasping_ai.simulation.scene import MuJoCoScene

from grasping_ai.simulation.ycb import (
    find_ycb_mjcf,
    resolve_ycb_object_directory,
)

from grasping_ai.utils.path_validation import (
    require_optional_path,
    require_path,
)

from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

import mujoco  # type: ignore[import-untyped]
import mujoco.viewer  # type: ignore[import-untyped]
import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

def load_visualization_scene(
    robot_xml_path: Path,
    object_id: str | None = None,
    ycb_root: Path | None = None,
    table_xml_path: Path | None = None,
) -> tuple[Any, Any]:
    """Load a MuJoCo model and data for interactive viewing.

    Args:
        robot_xml_path: Path to the robot MJCF description.
        object_id: Optional YCB object identifier to include in the scene.
        ycb_root: Root directory of the YCB MJCF set. Required when
            ``object_id`` is set.
        table_xml_path: Optional workbench/table MJCF description.

    Returns:
        ``(mj_model, mj_data)`` ready for ``mujoco.viewer``.
    """
    require_path(robot_xml_path, "robot_xml_path")
    if not robot_xml_path.is_file():
        raise FileNotFoundError(f"robot_xml_path not found: {robot_xml_path}")
    require_optional_path(table_xml_path, "table_xml_path")
    if table_xml_path is not None and not table_xml_path.is_file():
        raise FileNotFoundError(f"table_xml_path not found: {table_xml_path}")
    if object_id is not None and not isinstance(object_id, str):
        raise TypeError("object_id must be a string or None")
    object_xml_path = None
    if object_id:
        if ycb_root is None or not isinstance(ycb_root, Path):
            raise ValueError("ycb_root is required when object_id is set")
        if not ycb_root.is_dir():
            raise FileNotFoundError(f"ycb_root not found: {ycb_root}")
        
        object_xml_path = find_ycb_mjcf(resolve_ycb_object_directory(ycb_root, object_id))

    
    scene = MuJoCoScene(
        robot_xml_path,
        object_xml_path,
        table_xml_path,
        object_name=object_id,
    )
    apply_home_keyframe(scene.model, scene.data)
    if object_id is not None and table_xml_path is not None:
        _place_object_on_table(scene.model, scene.data, object_id)
    return scene.model, scene.data

def _place_object_on_table(mj_model: Any, mj_data: Any, object_name: str) -> None:
    """Set the object freejoint so it rests on the table top surface.

    Finds the ``table`` body position and the ``table_top`` geom offset to
    compute the table surface height, then moves the object freejoint so
    its body origin sits on that surface.

    Args:
        mj_model: MuJoCo model containing the table and object bodies.
        mj_data: MuJoCo data whose ``qpos`` will be updated.
        object_name: Logical name of the object body to reposition.
    """
    table_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "table")
    if table_body_id == -1:
        return

    table_z = float(mj_model.body_pos[table_body_id][2])

    table_top_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    if table_top_geom_id != -1:
        geom_pos = mj_model.geom_pos[table_top_geom_id]
        geom_size = mj_model.geom_size[table_top_geom_id]
        table_z += float(geom_pos[2]) + float(geom_size[2])

    object_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    if object_body_id == -1:
        return

    for jnt_id in range(int(mj_model.njnt)):
        if mj_model.jnt_type[jnt_id] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        if int(mj_model.jnt_bodyid[jnt_id]) != object_body_id:
            continue
        qadr = int(mj_model.jnt_qposadr[jnt_id])
        mj_data.qpos[qadr + 2] = table_z
        mj_data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        logger.info("Placed object '{}' on table surface at z={:.4f}", object_name, table_z)
        break

    mujoco.mj_forward(mj_model, mj_data)

def apply_home_keyframe(mj_model: Any, mj_data: Any) -> None:
    """Reset robot joints to keyframe 0 without wiping extra scene DOFs.

    Assembled scenes add object freejoints after the robot. Applying the
    robot-only home keyframe with ``mj_resetDataKeyframe`` would zero those
    extra coordinates and drop the object at the origin.
    """
    if int(mj_model.nkey) <= 0:
        mujoco.mj_resetData(mj_model, mj_data)
        mujoco.mj_forward(mj_model, mj_data)
        return

    key_qpos = np.asarray(mj_model.key_qpos[0], dtype=np.float64)
    for joint_id in range(int(mj_model.njnt)):
        if mj_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        qadr = int(mj_model.jnt_qposadr[joint_id])
        width = 4 if mj_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_BALL else 1
        if qadr + width > key_qpos.shape[0] or qadr + width > mj_data.qpos.shape[0]:
            continue
        mj_data.qpos[qadr : qadr + width] = key_qpos[qadr : qadr + width]
    key_ctrl = np.asarray(mj_model.key_ctrl[0], dtype=np.float64)
    nctrl = min(int(key_ctrl.shape[0]), int(mj_data.ctrl.shape[0]))
    mj_data.ctrl[:nctrl] = key_ctrl[:nctrl]
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

def run_robot_viewer(
    mj_model: Any,
    mj_data: Any,
    *,
    launch_passive: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    max_duration_s: float | None = None,
) -> None:
    """Open a MuJoCo passive viewer and step physics until the window closes.

    Actuator and simulation controls are handled by MuJoCo's built-in viewer UI.

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.
        launch_passive: Optional viewer factory (defaults to mujoco.viewer).
        sleep: Optional sleep function.
        clock: Optional clock function returning seconds.
        max_duration_s: Optional wall-clock limit, used by tests.
    """
    if launch_passive is None:
        launch_passive = mujoco.viewer.launch_passive
    if sleep is None:
        sleep = time.sleep
    if clock is None:
        clock = time.time

    dt = float(mj_model.opt.timestep)
    if dt <= 0 or not np.isfinite(dt):
        dt = 0.002

    logger.info("Launching MuJoCo viewer...")
    viewer = launch_passive(mj_model, mj_data)
    start = clock()
    try:
        while True:
            is_running_fn = getattr(viewer, "is_running", None)
            if callable(is_running_fn) and not is_running_fn():
                break
            if max_duration_s is not None and clock() - start >= max_duration_s:
                break
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
            sleep(dt)
    finally:
        close = getattr(viewer, "close", None)
        if callable(close):
            close()
