"""Tests for robot visualization."""

from __future__ import annotations

from grasping_ai.pipelines.visualize_robot import (
    apply_home_keyframe,
    load_visualization_scene,
    run_robot_viewer,
)

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

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

def test_run_robot_viewer_steps_until_viewer_stops(
    panda_robot_xml: Path,
) -> None:
    """Verify that run_robot_viewer runs the simulation passive viewer until closed."""
    model, data = load_visualization_scene(panda_robot_xml)

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
    run_robot_viewer(
        model,
        data,
        launch_passive=lambda *args, **kwargs: viewer,
        sleep=lambda _dt: None,
        clock=lambda: 0.0,
    )
    assert viewer.syncs >= 1
    assert viewer.closed is True
