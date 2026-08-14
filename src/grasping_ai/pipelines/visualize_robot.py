import json
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any
from loguru import logger

import numpy as np


def load_visualization_scene(
    robot_xml_path: Path,
    object_id: str | None = None,
    ycb_root: Path | None = None,
    table_xml_path: Path | None = None,
) -> tuple[Any, Any]:
    """Load a MuJoCo model and data for interactive viewing or teleoperation.

    Args:
        robot_xml_path: Path to the robot MJCF description.
        object_id: Optional YCB object identifier to include in the scene.
        ycb_root: Root directory of the YCB MJCF set. Required when
            ``object_id`` is set.
        table_xml_path: Optional workbench/table MJCF description.

    Returns:
        ``(mj_model, mj_data)`` ready for ``mujoco.viewer``.
    """
    if not isinstance(robot_xml_path, Path):
        raise TypeError("robot_xml_path must be a pathlib.Path instance")
    if not robot_xml_path.is_file():
        raise FileNotFoundError(f"robot_xml_path not found: {robot_xml_path}")
    if table_xml_path is not None and not isinstance(table_xml_path, Path):
        raise TypeError("table_xml_path must be a pathlib.Path instance or None")
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
        from grasping_ai.simulation.ycb import find_ycb_mjcf, resolve_ycb_object_directory

        object_xml_path = find_ycb_mjcf(resolve_ycb_object_directory(ycb_root, object_id))

    from grasping_ai.simulation.scene import MuJoCoScene

    scene = MuJoCoScene(
        robot_xml_path,
        object_xml_path,
        table_xml_path,
        object_name=object_id,
    )
    apply_home_keyframe(scene.model, scene.data)
    return scene.model, scene.data


def apply_home_keyframe(mj_model: Any, mj_data: Any) -> None:
    """Reset robot joints to keyframe 0 without wiping extra scene DOFs.

    Assembled scenes add object freejoints after the robot. Applying the
    robot-only home keyframe with ``mj_resetDataKeyframe`` would zero those
    extra coordinates and drop the object at the origin.
    """
    import mujoco  # type: ignore[import-untyped]

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


def read_tui_key() -> int | None:
    """Read one pending terminal key without blocking.

    Returns:
        A GLFW keycode, ``-1`` for quit, or ``None`` when no key is waiting.
    """
    import sys

    key_quit = -1
    if sys.platform == "win32":
        import msvcrt

        if not msvcrt.kbhit():
            return None
        first = msvcrt.getch()
        if first in (b"\x00", b"\xe0"):
            second = msvcrt.getch()
            return {b"K": 263, b"M": 262}.get(second)
        if first in (b"q", b"Q", b"\x1b"):
            return key_quit
        try:
            char = first.decode("ascii")
        except UnicodeDecodeError:
            return None
    else:
        import select

        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        char = sys.stdin.read(1)
        if char == "\x1b":
            extra = ""
            if select.select([sys.stdin], [], [], 0)[0]:
                extra += sys.stdin.read(1)
            if extra == "[" and select.select([sys.stdin], [], [], 0)[0]:
                extra += sys.stdin.read(1)
            return {"[D": 263, "[C": 262}.get(extra, key_quit)
        if char in ("q", "Q"):
            return key_quit

    if len(char) == 1 and "1" <= char <= "9":
        return 49 + (ord(char) - ord("1"))
    return {
        "g": 71,
        "G": 71,
        "h": 72,
        "H": 72,
        " ": 32,
        "-": 45,
        "=": 61,
        "[": 91,
        "]": 93,
    }.get(char)


def run_keyboard_tui(
    *,
    host: str = "127.0.0.1",
    port: int = 5511,
    read_key: Callable[[], int | None] | None = None,
    sleep: Callable[[float], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    publish_key: Callable[[int], None] | None = None,
) -> None:
    """Capture terminal keys and publish them on the ``robot/keyboard`` UDP topic.

    Args:
        host: UDP host of the viewer subscriber.
        port: UDP port of the viewer subscriber.
        read_key: Optional non-blocking key reader (used by tests).
        sleep: Optional sleep function.
        should_stop: Optional cooperative stop flag (used by tests).
        publish_key: Optional key publisher (used by tests).
    """
    import time

    topic = "robot/keyboard"
    if read_key is None:
        read_key = read_tui_key
    if sleep is None:
        sleep = time.sleep
    owns_publisher = publish_key is None
    pub_sock: socket.socket | None = None
    if publish_key is None:
        pub_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        def publish_key(keycode: int) -> None:
            assert pub_sock is not None
            payload = json.dumps({"topic": topic, "keycode": keycode}, separators=(",", ":")).encode("utf-8")
            pub_sock.sendto(payload, (host, port))

    print(
        "Robot keyboard TUI (no MuJoCo window). Publishes to topic robot/keyboard.\n"
        "Start scripts/visualize_robot.py in another terminal first, then use keys:\n"
        "  1-9 select actuator; left/right nudge; g gripper; h home; space pause; q quit"
    )
    logger.info("Publishing {} -> udp://{}:{}", topic, host, port)
    try:
        while True:
            if should_stop is not None and should_stop():
                break
            keycode = read_key()
            if keycode == -1:
                logger.info("TUI quit")
                break
            if keycode is not None:
                publish_key(keycode)
                logger.info("sent {} keycode={}", topic, keycode)
            sleep(0.02)
    finally:
        if owns_publisher and pub_sock is not None:
            pub_sock.close()


def run_robot_viewer(
    mj_model: Any,
    mj_data: Any,
    *,
    listen_keyboard_topic: bool = True,
    topic_host: str | None = None,
    topic_port: int | None = None,
    launch_passive: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    max_duration_s: float | None = None,
) -> None:
    """Open a MuJoCo viewer and apply keycodes received on the keyboard topic.

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.
        listen_keyboard_topic: If True, bind UDP ``robot/keyboard`` and apply
            incoming keycodes to the robot controller.
        topic_host: Optional UDP bind host override.
        topic_port: Optional UDP bind port override.
        launch_passive: Optional viewer factory (defaults to mujoco.viewer).
        sleep: Optional sleep function.
        clock: Optional clock function returning seconds.
        max_duration_s: Optional wall-clock limit, used by tests.
    """
    control_state = init_robot_control(mj_model, mj_data)
    topic_sock: socket.socket | None = None
    key_source: Callable[[], list[int]] | None = None
    if listen_keyboard_topic:
        host = "127.0.0.1" if topic_host is None else topic_host
        port = 5511 if topic_port is None else topic_port
        if not isinstance(host, str) or not host:
            raise TypeError("host must be a non-empty string")
        if not isinstance(port, int) or not (0 <= port <= 65535):
            raise ValueError("port must be an integer in 0..65535")
        topic_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        topic_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        topic_sock.bind((host, port))
        topic_sock.setblocking(False)
        port = int(topic_sock.getsockname()[1])
        topic = "robot/keyboard"

        def key_source() -> list[int]:
            keycodes: list[int] = []
            while True:
                try:
                    payload, _addr = topic_sock.recvfrom(4096)
                except BlockingIOError:
                    break
                except OSError:
                    break
                try:
                    message = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(message, dict):
                    continue
                if message.get("topic") != topic:
                    continue
                keycode = message.get("keycode")
                if isinstance(keycode, int):
                    keycodes.append(keycode)
            return keycodes

        logger.info("Listening for robot/keyboard on udp://{}:{}", host, port)
    try:
        run_robot_control_loop(
            mj_model,
            mj_data,
            control_state,
            key_source=key_source,
            launch_passive=launch_passive,
            sleep=sleep,
            clock=clock,
            max_duration_s=max_duration_s,
        )
    finally:
        if topic_sock is not None:
            topic_sock.close()


def init_robot_control(mj_model: Any, mj_data: Any) -> dict[str, Any]:
    """Initialize keyboard teleoperation state for actuator targets.

    Args:
        mj_model: MuJoCo model with at least one actuator.
        mj_data: MuJoCo data whose ``ctrl`` vector will be driven.

    Returns:
        Mutable control state consumed by ``handle_robot_control_key``.

    Raises:
        ValueError: If the model has no actuators.
    """
    import mujoco  # type: ignore[import-untyped]

    from grasping_ai.robotics.gripper import gripper_actuator_indices

    if int(mj_model.nu) <= 0:
        raise ValueError("model has no actuators to control")
    if int(mj_model.nkey) > 0 and mj_model.key_ctrl.shape[1] == mj_model.nu:
        ctrl = np.array(mj_model.key_ctrl[0], dtype=np.float64, copy=True)
    else:
        ctrl = np.array(mj_data.ctrl, dtype=np.float64, copy=True)
    mj_data.ctrl[:] = ctrl
    mujoco.mj_forward(mj_model, mj_data)
    return {
        "model": mj_model,
        "data": mj_data,
        "selected": 0,
        "paused": False,
        "gripper_open": True,
        "gripper_ids": gripper_actuator_indices(mj_model),
        "ctrl": ctrl,
    }


def handle_robot_control_key(control_state: dict[str, Any], keycode: int) -> None:
    """Apply a GLFW keycode to robot teleoperation state.

    Args:
        control_state: State returned by ``init_robot_control``.
        keycode: GLFW key code from the viewer or keyboard topic.
    """
    import mujoco  # type: ignore[import-untyped]

    if not isinstance(keycode, (int, np.integer)):
        raise TypeError("keycode must be an integer")
    code = int(keycode)
    mj_model = control_state["model"]
    mj_data = control_state["data"]
    ctrl = control_state["ctrl"]
    selected = control_state["selected"]
    if 49 <= code <= 57:
        index = code - 49
        if index < int(mj_model.nu):
            control_state["selected"] = index
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or f"actuator_{index}"
            logger.info("Selected actuator {}: {}", index, name)
        return
    if code in (263, 45, 91):
        if mj_model.actuator_ctrllimited[selected]:
            lo, hi = mj_model.actuator_ctrlrange[selected]
            lo, hi = float(lo), float(hi)
        else:
            lo, hi = -3.14, 3.14
        span = hi - lo
        step = 15.0 if span > 10.0 else 0.05
        value = float(np.clip(ctrl[selected] - step, lo, hi))
        ctrl[selected] = value
        mj_data.ctrl[selected] = value
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, selected) or f"actuator_{selected}"
        logger.info("actuator {} ({}) = {:.4f}", selected, name, value)
        return
    if code in (262, 61, 93):
        if mj_model.actuator_ctrllimited[selected]:
            lo, hi = mj_model.actuator_ctrlrange[selected]
            lo, hi = float(lo), float(hi)
        else:
            lo, hi = -3.14, 3.14
        span = hi - lo
        step = 15.0 if span > 10.0 else 0.05
        value = float(np.clip(ctrl[selected] + step, lo, hi))
        ctrl[selected] = value
        mj_data.ctrl[selected] = value
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, selected) or f"actuator_{selected}"
        logger.info("actuator {} ({}) = {:.4f}", selected, name, value)
        return
    if code == 71:
        gripper_ids = control_state["gripper_ids"]
        if not gripper_ids:
            logger.warning("No gripper actuator found")
            return
        control_state["gripper_open"] = not control_state["gripper_open"]
        for idx in gripper_ids:
            if mj_model.actuator_ctrllimited[idx]:
                lo, hi = mj_model.actuator_ctrlrange[idx]
                lo, hi = float(lo), float(hi)
            else:
                lo, hi = -3.14, 3.14
            ctrl[idx] = hi if control_state["gripper_open"] else lo
            mj_data.ctrl[idx] = ctrl[idx]
        logger.info("Gripper open" if control_state["gripper_open"] else "Gripper closed")
        return
    if code == 72:
        apply_home_keyframe(mj_model, mj_data)
        if int(mj_model.nkey) > 0 and mj_model.key_ctrl.shape[1] == mj_model.nu:
            control_state["ctrl"] = np.array(mj_model.key_ctrl[0], dtype=np.float64, copy=True)
        else:
            control_state["ctrl"] = np.array(mj_data.ctrl, dtype=np.float64, copy=True)
        mj_data.ctrl[:] = control_state["ctrl"]
        logger.info("Reset to home keyframe")
        return
    if code == 32:
        control_state["paused"] = not control_state["paused"]
        logger.info("Paused" if control_state["paused"] else "Running")


def apply_robot_control(control_state: dict[str, Any]) -> None:
    """Write the current command vector into ``data.ctrl``.

    Args:
        control_state: State returned by ``init_robot_control``.
    """
    control_state["data"].ctrl[:] = control_state["ctrl"]


def run_robot_control_loop(
    mj_model: Any,
    mj_data: Any,
    control_state: dict[str, Any] | None = None,
    *,
    key_source: Callable[[], list[int]] | None = None,
    max_duration_s: float | None = None,
    launch_passive: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
) -> None:
    """Step physics in a passive MuJoCo viewer while handling keyboard control.

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.
        control_state: Optional pre-built control state. Created if omitted.
        key_source: Optional poller returning GLFW keycodes from the keyboard
            topic (or tests).
        max_duration_s: Optional wall-clock limit, used by tests.
        launch_passive: Optional viewer factory (defaults to mujoco.viewer).
        sleep: Optional sleep function.
        clock: Optional clock function returning seconds.
    """
    import time

    import mujoco  # type: ignore[import-untyped]

    if control_state is None:
        control_state = init_robot_control(mj_model, mj_data)
    if launch_passive is None:
        import mujoco.viewer  # type: ignore[import-untyped]

        launch_passive = mujoco.viewer.launch_passive
    if sleep is None:
        sleep = time.sleep
    if clock is None:
        clock = time.time

    dt = float(mj_model.opt.timestep)
    if dt <= 0 or not np.isfinite(dt):
        dt = 0.002

    print(
        "Robot control keys come from run_keyboard_tui (UDP topic robot/keyboard):\n"
        "  1-9          select actuator\n"
        "  left/right   decrease/increase selected actuator\n"
        "  [ / ]        same as left/right\n"
        "  g            toggle gripper open/close\n"
        "  h            reset to home keyframe\n"
        "  space        pause/resume physics\n"
        "  close window to exit"
    )
    logger.info("Launching MuJoCo control viewer...")
    viewer = launch_passive(
        mj_model,
        mj_data,
        key_callback=lambda keycode: handle_robot_control_key(control_state, keycode),
    )
    start = clock()
    try:
        while True:
            is_running_fn = getattr(viewer, "is_running", None)
            if callable(is_running_fn) and not is_running_fn():
                break
            if max_duration_s is not None and clock() - start >= max_duration_s:
                break
            if key_source is not None:
                for keycode in key_source():
                    handle_robot_control_key(control_state, keycode)
            if not control_state["paused"]:
                apply_robot_control(control_state)
                mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
            sleep(dt)
    finally:
        close = getattr(viewer, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--keyboard-tui":
        run_keyboard_tui()
    else:
        raise SystemExit("Use scripts/visualize_robot.py for the MuJoCo viewer, or pass --keyboard-tui.")
