from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

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

MINIMAL_UNACTUATED_ROBOT_XML = """\
<mujoco model="minimal_unactuated_robot">
    <compiler angle="radian"/>
    <worldbody>
        <body name="base" pos="0 0 0">
            <geom name="base_geom" type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>
"""

@pytest.fixture
def minimal_robot_xml(tmp_path: Path) -> Path:
    path = tmp_path / "robot.xml"
    path.write_text(MINIMAL_ACTUATED_ROBOT_XML, encoding="utf-8")
    return path

@pytest.fixture
def minimal_unactuated_robot_xml(tmp_path: Path) -> Path:
    path = tmp_path / "unactuated_robot.xml"
    path.write_text(MINIMAL_UNACTUATED_ROBOT_XML, encoding="utf-8")
    return path

def test_env_initialization(minimal_robot_xml):
    env = MuJoCoGraspingEnv(minimal_robot_xml)

    # Check observation and action space shapes/dtypes
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (2,)  # qpos (1) + qvel (1)
    assert env.observation_space.dtype == np.float32

    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.action_space.shape == (1,)
    assert env.action_space.dtype == np.float32

    # Actuator limits validation
    assert np.allclose(env.action_space.low, np.array([-1.5], dtype=np.float32))
    assert np.allclose(env.action_space.high, np.array([1.5], dtype=np.float32))

def test_env_initialization_no_actuators(minimal_unactuated_robot_xml):
    with pytest.raises(ValueError, match="zero actuators"):
        MuJoCoGraspingEnv(minimal_unactuated_robot_xml)

def test_env_reset(minimal_robot_xml):
    env = MuJoCoGraspingEnv(minimal_robot_xml)
    obs, info = env.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)

def test_env_reset_determinism(minimal_robot_xml):
    env = MuJoCoGraspingEnv(minimal_robot_xml)
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    assert np.allclose(obs1, obs2)

def test_env_step_valid(minimal_robot_xml):
    env = MuJoCoGraspingEnv(minimal_robot_xml)
    env.reset(seed=42)

    action = np.array([0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_env_step_non_finite_rejection(minimal_robot_xml):
    env = MuJoCoGraspingEnv(minimal_robot_xml)
    env.reset(seed=42)

    with pytest.raises(ValueError, match="finite values"):
        env.step(np.array([np.nan], dtype=np.float32))

    with pytest.raises(ValueError, match="finite values"):
        env.step(np.array([np.inf], dtype=np.float32))

def test_env_step_padding_truncation(minimal_robot_xml):
    env = MuJoCoGraspingEnv(minimal_robot_xml)
    env.reset(seed=42)

    # Over-sized action should be truncated
    obs, _reward, _terminated, _truncated, _info = env.step(np.array([0.5, 1.0], dtype=np.float32))
    assert np.isfinite(obs).all()

    # Under-sized action should be padded
    obs, _reward, _terminated, _truncated, _info = env.step(np.array([], dtype=np.float32))
    assert np.isfinite(obs).all()
