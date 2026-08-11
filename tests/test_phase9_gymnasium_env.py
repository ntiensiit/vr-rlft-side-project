from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from grasping_ai.inference.policy_runner import (
    build_rl_policy_runner,
    load_rl_policy_checkpoint,
)
from grasping_ai.pipelines.train_rl import run_rl_training_pipeline
from grasping_ai.simulation.mujoco_env import MuJoCoGraspingEnv, RewardConfig
from grasping_ai.simulation.scene import build_scene_xml

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


OBJECT_XML = """\
<mujoco model="object">
    <worldbody>
        <body name="object" pos="0 0 0.5">
            <freejoint/>
            <geom name="object_geom" type="sphere" size="0.05"/>
        </body>
    </worldbody>
</mujoco>
"""


@pytest.fixture
def scene_xml(tmp_path: Path) -> Path:
    robot_path = tmp_path / "robot.xml"
    robot_path.write_text(MINIMAL_ACTUATED_ROBOT_XML, encoding="utf-8")
    object_path = tmp_path / "object.xml"
    object_path.write_text(OBJECT_XML, encoding="utf-8")
    return build_scene_xml(robot_path, object_path, None)


SB3_ROBOT_XML = """\
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
        <motor name="motor1" joint="joint1" gear="1" ctrlrange="-1.0 1.0"/>
    </actuator>
</mujoco>
"""


@pytest.fixture
def robot_xml(tmp_path: Path) -> Path:
    path = tmp_path / "robot.xml"
    path.write_text(SB3_ROBOT_XML, encoding="utf-8")
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


def test_env_step_terminal_returns_terminal_observation_without_reset(
    minimal_robot_xml, monkeypatch
):
    """Verify a non-finite transition returns the terminal observation unchanged."""
    import grasping_ai.simulation.mujoco_env as mujoco_env_module

    env = MuJoCoGraspingEnv(minimal_robot_xml)
    env.reset(seed=42)

    terminal_obs = np.full(2, np.nan, dtype=np.float32)
    monkeypatch.setattr(env, "_get_observation", lambda: terminal_obs)

    reset_calls: list[object] = []
    original_reset = mujoco_env_module.reset_simulation

    def spy_reset(state):
        reset_calls.append(state)
        return original_reset(state)

    monkeypatch.setattr(mujoco_env_module, "reset_simulation", spy_reset)

    obs, _reward, terminated, truncated, _info = env.step(np.array([0.5], dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert not np.isfinite(obs).all()
    assert reset_calls == []


def test_env_step_padding_truncation(minimal_robot_xml):
    env = MuJoCoGraspingEnv(minimal_robot_xml)
    env.reset(seed=42)

    # Over-sized action should be truncated
    obs, _reward, _terminated, _truncated, _info = env.step(np.array([0.5, 1.0], dtype=np.float32))
    assert np.isfinite(obs).all()

    # Under-sized action should be padded
    obs, _reward, _terminated, _truncated, _info = env.step(np.array([], dtype=np.float32))
    assert np.isfinite(obs).all()


def test_default_reward_preserves_legacy_behavior(minimal_robot_xml):
    env = MuJoCoGraspingEnv(minimal_robot_xml)
    env.reset(seed=42)
    _obs, reward, terminated, truncated, _info = env.step(
        np.array([0.5], dtype=np.float32)
    )
    assert reward == pytest.approx(-0.01 * 0.25 + 1.0)
    assert terminated is False
    assert truncated is False


def test_contact_reward_term(scene_xml):
    env = MuJoCoGraspingEnv(
        scene_xml, object_name="object", reward_config=RewardConfig(contact_reward=0.5)
    )
    env.reset(seed=42)
    env._has_object_contact = lambda: True  # type: ignore[method-assign]
    _obs, reward, terminated, _truncated, _info = env.step(
        np.array([0.0], dtype=np.float32)
    )
    assert reward == pytest.approx(1.0 + 0.5)
    assert terminated is False


def test_lift_and_grasp_success_rewards(scene_xml):
    env = MuJoCoGraspingEnv(
        scene_xml,
        object_name="object",
        reward_config=RewardConfig(
            lift_reward_weight=2.0,
            grasp_success_bonus=5.0,
            lift_height_threshold=0.05,
        ),
    )
    env.reset(seed=42)
    initial = env._initial_object_height

    env._get_object_height = lambda: initial + 0.03  # type: ignore[method-assign]
    _obs, reward1, _term, _trunc, _info = env.step(np.array([0.0], dtype=np.float32))
    assert reward1 == pytest.approx(1.0 + 2.0 * 0.03)

    env._get_object_height = lambda: initial + 0.08  # type: ignore[method-assign]
    _obs, reward2, _term, _trunc, _info = env.step(np.array([0.0], dtype=np.float32))
    assert reward2 == pytest.approx(1.0 + 2.0 * 0.08 + 5.0)

    _obs, reward3, _term, _trunc, _info = env.step(np.array([0.0], dtype=np.float32))
    assert reward3 == pytest.approx(1.0 + 2.0 * 0.08)


def test_drop_below_threshold_terminates(scene_xml):
    env = MuJoCoGraspingEnv(
        scene_xml, object_name="object", reward_config=RewardConfig(drop_height_threshold=0.1)
    )
    env.reset(seed=42)
    initial = env._initial_object_height
    env._get_object_height = lambda: initial - 0.2  # type: ignore[method-assign]
    _obs, _reward, terminated, truncated, _info = env.step(
        np.array([0.0], dtype=np.float32)
    )
    assert terminated is True
    assert truncated is False


def test_env_validates_reward_configuration(scene_xml):
    with pytest.raises(TypeError, match="reward_config"):
        MuJoCoGraspingEnv(scene_xml, object_name="object", reward_config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="object_name"):
        MuJoCoGraspingEnv(scene_xml, object_name=123)  # type: ignore[arg-type]


def test_non_finite_terminal_uses_config(minimal_robot_xml, monkeypatch):
    env = MuJoCoGraspingEnv(
        minimal_robot_xml,
        reward_config=RewardConfig(terminate_on_non_finite=False),
    )
    env.reset(seed=42)
    monkeypatch.setattr(env, "_get_observation", lambda: np.full(2, np.nan, dtype=np.float32))
    _obs, _reward, terminated, _truncated, _info = env.step(
        np.array([0.0], dtype=np.float32)
    )
    assert terminated is False


def test_env_step_routes_through_shared_command_path(minimal_robot_xml, monkeypatch):
    import grasping_ai.simulation.mujoco_env as mujoco_env_module
    from grasping_ai.simulation.mujoco_env import set_actuator_controls

    calls: list[np.ndarray] = []

    def spy_set_actuator_controls(state, ctrl):
        calls.append(np.asarray(ctrl, copy=True))
        return set_actuator_controls(state, ctrl)

    monkeypatch.setattr(
        mujoco_env_module, "set_actuator_controls", spy_set_actuator_controls
    )

    env = MuJoCoGraspingEnv(minimal_robot_xml)
    env.reset(seed=42)
    env.step(np.array([0.5], dtype=np.float32))

    assert len(calls) == 1
    assert np.allclose(calls[0], [0.5])


@pytest.mark.filterwarnings("ignore")
def test_gymnasium_env_compliance(minimal_robot_xml):
    env = MuJoCoGraspingEnv(minimal_robot_xml)
    check_env(env)


def test_sb3_training_and_inference_compatibility(robot_xml, tmp_path):
    checkpoint_path = tmp_path / "policy.pt"
    log_dir = tmp_path / "tb_logs"

    # Run the pipeline for a tiny number of updates (e.g. 1 update = 64 steps)
    run_rl_training_pipeline(
        robot_xml_path=robot_xml,
        ycb_root=tmp_path,  # unused since object_ids is empty
        object_ids=[],
        policy_checkpoint_path=checkpoint_path,
        observation_dim=2,  # qpos (1) + qvel (1)
        action_dim=1,
        hidden_dim=16,
        learning_rate=1e-3,
        num_updates=1,
        gamma=0.99,
        device="cpu",
        seed=42,
        experiment_log_dir=log_dir,
    )

    assert checkpoint_path.is_file()

    # Load and build inference policy runner
    checkpoint = load_rl_policy_checkpoint(checkpoint_path, "cpu")
    assert "model_state_dict" in checkpoint

    runner = build_rl_policy_runner(
        checkpoint=checkpoint,
        observation_dim=2,
        action_dim=1,
        device="cpu",
    )

    obs = np.array([0.5, -0.2], dtype=np.float32)
    action = runner(obs)

    assert isinstance(action, np.ndarray)
    assert action.shape == (1,)
    assert np.isfinite(action).all()
