from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

from grasping_ai.pipelines.visualize_robot import (
    apply_home_keyframe,
    handle_robot_control_key,
    init_robot_control,
    load_visualization_scene,
    run_robot_control_loop,
    run_robot_viewer,
)


def test_load_visualization_scene_requires_robot_file(tmp_path: Path) -> None:
    """Verify that load_visualization_scene fails with FileNotFoundError if the robot file is missing."""
    with pytest.raises(FileNotFoundError, match="robot_xml_path"):
        load_visualization_scene(tmp_path / "missing.xml")


def test_load_visualization_scene_requires_ycb_root_with_object(
    panda_robot_xml: Path,
) -> None:
    """Verify that load_visualization_scene fails if YCB root directory is missing when object is loaded."""
    with pytest.raises(ValueError, match="ycb_root"):
        load_visualization_scene(panda_robot_xml, object_id="cracker")


def test_load_visualization_scene_applies_home_keyframe(panda_robot_xml: Path) -> None:
    """Verify that load_visualization_scene initializes joint positions to the home keyframe values."""
    _model, data = load_visualization_scene(panda_robot_xml)
    assert np.allclose(
        data.qpos[:9],
        [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04],
        atol=1e-4,
    )


def test_apply_home_keyframe_without_keys(tmp_path: Path) -> None:
    """Verify that apply_home_keyframe handles MuJoCo models that do not contain keyframes gracefully."""
    xml = tmp_path / "nokey.xml"
    xml.write_text(
        """<mujoco model="nokey">
        <worldbody><body name="b" pos="0 0 0">
            <joint name="j" type="slide" axis="0 0 1"/>
            <geom type="box" size="0.05 0.05 0.05"/>
        </body></worldbody>
        <actuator><position joint="j"/></actuator>
        </mujoco>""",
        encoding="utf-8",
    )
    model, data = load_visualization_scene(xml)
    data.qpos[0] = 0.3
    apply_home_keyframe(model, data)
    assert data.qpos[0] == pytest.approx(0.0)


def test_robot_control_selects_and_nudges_actuator(panda_robot_xml: Path) -> None:
    """Verify that keyboard control handles pausing, home keys, selecting and nudging actuators."""
    model, data = load_visualization_scene(panda_robot_xml)
    control_state = init_robot_control(model, data)
    handle_robot_control_key(control_state, 49)
    before = float(control_state["ctrl"][0])
    handle_robot_control_key(control_state, 262)
    assert control_state["ctrl"][0] > before
    handle_robot_control_key(control_state, 32)
    assert control_state["paused"] is True
    handle_robot_control_key(control_state, 71)
    handle_robot_control_key(control_state, 72)
    assert np.allclose(
        data.qpos[:9],
        [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04],
        atol=1e-4,
    )


def test_run_robot_viewer_listens_and_steps(
    panda_robot_xml: Path,
) -> None:
    """Verify that run_robot_viewer runs the simulation passive viewer until closed."""
    model, data = load_visualization_scene(panda_robot_xml)

    class FakeViewer:
        def __init__(self) -> None:
            self.remaining = 2
            self.closed = False

        def is_running(self) -> bool:
            self.remaining -= 1
            return self.remaining >= 0

        def sync(self) -> None:
            return

        def close(self) -> None:
            self.closed = True

    viewer = FakeViewer()
    run_robot_viewer(
        model,
        data,
        listen_keyboard_topic=False,
        launch_passive=lambda *args, **kwargs: viewer,
        sleep=lambda _dt: None,
        clock=lambda: 0.0,
    )
    assert viewer.closed is True


def test_run_robot_control_loop_steps_until_viewer_stops(
    panda_robot_xml: Path,
) -> None:
    """Verify that run_robot_control_loop syncs and steps the MuJoCo simulation until passive viewer stops."""
    model, data = load_visualization_scene(panda_robot_xml)
    control_state = init_robot_control(model, data)

    class FakeViewer:
        def __init__(self) -> None:
            self.remaining = 2
            self.syncs = 0
            self.closed = False

        def is_running(self) -> bool:
            self.remaining -= 1
            return self.remaining >= 0

        def sync(self) -> None:
            self.syncs += 1

        def close(self) -> None:
            self.closed = True

    viewer = FakeViewer()
    run_robot_control_loop(
        model,
        data,
        control_state,
        launch_passive=lambda *args, **kwargs: viewer,
        sleep=lambda _dt: None,
        clock=lambda: 0.0,
    )
    assert viewer.syncs >= 1
    assert viewer.closed is True


def test_run_robot_control_loop_applies_topic_keys(panda_robot_xml: Path) -> None:
    """Verify that run_robot_control_loop registers and handles keypresses from UDP keyboard topics."""
    model, data = load_visualization_scene(panda_robot_xml)
    control_state = init_robot_control(model, data)
    queued = [262]

    class FakeViewer:
        def __init__(self) -> None:
            self.remaining = 2

        def is_running(self) -> bool:
            self.remaining -= 1
            return self.remaining >= 0

        def sync(self) -> None:
            return

        def close(self) -> None:
            return

    def poll_keys() -> list[int]:
        keys = list(queued)
        queued.clear()
        return keys

    before = float(control_state["ctrl"][0])
    run_robot_control_loop(
        model,
        data,
        control_state,
        key_source=poll_keys,
        launch_passive=lambda *args, **kwargs: FakeViewer(),
        sleep=lambda _dt: None,
        clock=lambda: 0.0,
    )
    assert control_state["ctrl"][0] > before
