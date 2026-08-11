from pathlib import Path

import pytest
from gymnasium.utils.env_checker import check_env

from grasping_ai.simulation.mujoco_env import MuJoCoGraspingEnv

MINIMAL_ACTUATED_ROBOT_XML = """\
<mujoco model="minimal_actuated_robot">
    <compiler angle="radian"/>
    <worldbody>
        <body name="base" pos="0 0 0">
            <geom name="base_geom" type="box" size="0.1 0.1 0.1"/>
            <body name="link1" pos="0 0 0.2">
                <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14" limited="true"/>
                <geom name="link1_geom" type="cylinder" size="0.05 0.1"/>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor name="motor1" joint="joint1" gear="1" ctrlrange="-1.5 1.5"/>
    </actuator>
</mujoco>
"""

@pytest.fixture
def minimal_robot_xml(tmp_path: Path) -> Path:
    path = tmp_path / "robot.xml"
    path.write_text(MINIMAL_ACTUATED_ROBOT_XML, encoding="utf-8")
    return path

@pytest.mark.filterwarnings("ignore")
def test_gymnasium_env_compliance(minimal_robot_xml):
    env = MuJoCoGraspingEnv(minimal_robot_xml)
    check_env(env)
