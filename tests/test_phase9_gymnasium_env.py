"""Phase 9 Gymnasium environment tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import grasping_ai.simulation.mujoco_env as mujoco_env_module
from grasping_ai.inference.policy_runner import (
    build_rl_policy_runner,
    load_rl_policy_checkpoint,
)
from grasping_ai.pipelines.train_rl import run_rl_training_pipeline
from grasping_ai.simulation.mujoco_env import (
    MuJoCoGraspingEnv,
    RewardConfig,
    create_simulation,
    load_mujoco_model,
    set_actuator_controls,
)
from grasping_ai.simulation.scene import build_scene_xml

if TYPE_CHECKING:
    from pathlib import Path

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
def minimal_unactuated_robot_xml(tmp_path: Path) -> Path:
    """Fixture to write and provide a path to a minimal MuJoCo robot XML without actuators."""
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
def scene_xml(tmp_path: Path, panda_robot_xml: Path) -> Path:
    """Fixture to construct a scene XML merging a Franka Panda robot and a sphere object."""
    object_path = tmp_path / "object.xml"
    object_path.write_text(OBJECT_XML, encoding="utf-8")
    return build_scene_xml(panda_robot_xml, object_path, None)


def test_env_initialization(panda_robot_xml: Path) -> None:
    """Verify that the gymnasium environment initializes with correct action and observation space dimensions."""
    env = MuJoCoGraspingEnv(panda_robot_xml)

    if not (isinstance(env.observation_space, gym.spaces.Box)):
        raise TypeError
    if not (env.observation_space.shape == (18,)):
        raise AssertionError
    if not (env.observation_space.dtype == np.float32):
        raise AssertionError

    if not (isinstance(env.action_space, gym.spaces.Box)):
        raise TypeError
    if not (env.action_space.shape == (8,)):
        raise AssertionError
    if not (env.action_space.dtype == np.float32):
        raise AssertionError
    if not (env.action_space.low.shape == (8,)):
        raise AssertionError
    if not (env.action_space.high.shape == (8,)):
        raise AssertionError


def test_env_initialization_no_actuators(minimal_unactuated_robot_xml: Path) -> None:
    """Verify that environment initialization raises a ValueError if the MuJoCo model has zero actuators."""
    with pytest.raises(ValueError, match="zero actuators"):
        MuJoCoGraspingEnv(minimal_unactuated_robot_xml)


def test_env_reset(panda_robot_xml: Path) -> None:
    """Verify that resetting the environment returns the initial observation vector and info dictionary."""
    env = MuJoCoGraspingEnv(panda_robot_xml)
    obs, info = env.reset(seed=42)

    if not (isinstance(obs, np.ndarray)):
        raise TypeError
    if not (obs.shape == (18,)):
        raise AssertionError
    if not (obs.dtype == np.float32):
        raise AssertionError
    if not (isinstance(info, dict)):
        raise TypeError


def test_env_reset_determinism(panda_robot_xml: Path) -> None:
    """Verify that env reset is deterministic when using the same random seed."""
    env = MuJoCoGraspingEnv(panda_robot_xml)
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    if not (np.allclose(obs1, obs2)):
        raise AssertionError


def test_env_step_valid(panda_robot_xml: Path) -> None:
    """Verify that taking a step in the environment with a valid action updates the state and returns rewards."""
    env = MuJoCoGraspingEnv(panda_robot_xml)
    env.reset(seed=42)

    action = np.array([0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    if not (isinstance(obs, np.ndarray)):
        raise TypeError
    if not (obs.shape == (18,)):
        raise AssertionError
    if not (isinstance(reward, float)):
        raise TypeError
    if not (isinstance(terminated, bool)):
        raise TypeError
    if not (isinstance(truncated, bool)):
        raise TypeError
    if not (isinstance(info, dict)):
        raise TypeError


def test_env_step_non_finite_rejection(panda_robot_xml: Path) -> None:
    """Verify that non-finite step values (NaN/inf) are explicitly rejected with a ValueError."""
    env = MuJoCoGraspingEnv(panda_robot_xml)
    env.reset(seed=42)

    with pytest.raises(ValueError, match="finite values"):
        env.step(np.array([np.nan], dtype=np.float32))

    with pytest.raises(ValueError, match="finite values"):
        env.step(np.array([np.inf], dtype=np.float32))


def test_env_step_terminal_returns_terminal_observation_without_reset(
    panda_robot_xml: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a non-finite transition returns the terminal observation unchanged."""
    env = MuJoCoGraspingEnv(panda_robot_xml)
    env.reset(seed=42)

    terminal_obs = np.full(18, np.nan, dtype=np.float32)
    monkeypatch.setattr(env, "_get_observation", lambda: terminal_obs)

    reset_calls: list[object] = []
    original_reset = mujoco_env_module.reset_simulation

    def spy_reset(state: object) -> None:
        reset_calls.append(state)
        return original_reset(state)

    monkeypatch.setattr(mujoco_env_module, "reset_simulation", spy_reset)

    obs, _reward, terminated, truncated, _info = env.step(np.array([0.5], dtype=np.float32))

    if terminated is not True:
        raise AssertionError
    if truncated is not False:
        raise AssertionError
    if np.isfinite(obs).all():
        raise AssertionError
    if not (reset_calls == []):
        raise AssertionError


def test_env_step_padding_truncation(panda_robot_xml: Path) -> None:
    """Verify that action arrays are automatically padded or truncated to match action space limits."""
    env = MuJoCoGraspingEnv(panda_robot_xml)
    env.reset(seed=42)

    # Over-sized action should be truncated
    obs, _reward, _terminated, _truncated, _info = env.step(np.array([0.5, 1.0], dtype=np.float32))
    if not (np.isfinite(obs).all()):
        raise AssertionError

    # Under-sized action should be padded
    obs, _reward, _terminated, _truncated, _info = env.step(np.array([], dtype=np.float32))
    if not (np.isfinite(obs).all()):
        raise AssertionError


def test_default_reward_preserves_legacy_behavior(panda_robot_xml: Path) -> None:
    """Verify that the default reward computation matches legacy scaling (action cost + survival bonus)."""
    env = MuJoCoGraspingEnv(panda_robot_xml)
    env.reset(seed=42)
    _obs, reward, terminated, truncated, _info = env.step(np.array([0.5], dtype=np.float32))
    if not (reward == pytest.approx(-0.01 * 0.25 + 1.0)):
        raise AssertionError
    if terminated is not False:
        raise AssertionError
    if truncated is not False:
        raise AssertionError


def test_contact_reward_term(scene_xml: Path) -> None:
    """Verify that the reward correctly adds contact bonuses when tracked objects are touched."""
    env = MuJoCoGraspingEnv(scene_xml, object_name="object", reward_config=RewardConfig(contact_reward=0.5))
    env.reset(seed=42)
    env._has_object_contact = lambda: True  # type: ignore[method-assign]  # noqa: SLF001  # stub internal contact probe
    _obs, reward, terminated, _truncated, _info = env.step(np.array([0.0], dtype=np.float32))
    if not (reward == pytest.approx(1.0 + 0.5)):
        raise AssertionError
    if terminated is not False:
        raise AssertionError


def test_lift_and_grasp_success_rewards(scene_xml: Path) -> None:
    """Verify that lifting the object above thresholds triggers lift reward weights and grasp success bonuses."""
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
    initial = env._initial_object_height  # noqa: SLF001  # read internal baseline height

    env._get_object_height = lambda: initial + 0.03  # type: ignore[method-assign]  # noqa: SLF001  # stub internal height probe
    _obs, reward1, _term, _trunc, _info = env.step(np.array([0.0], dtype=np.float32))
    if not (reward1 == pytest.approx(1.0 + 2.0 * 0.03)):
        raise AssertionError

    env._get_object_height = lambda: initial + 0.08  # type: ignore[method-assign]  # noqa: SLF001  # stub internal height probe
    _obs, reward2, _term, _trunc, _info = env.step(np.array([0.0], dtype=np.float32))
    if not (reward2 == pytest.approx(1.0 + 2.0 * 0.08 + 5.0)):
        raise AssertionError

    _obs, reward3, _term, _trunc, _info = env.step(np.array([0.0], dtype=np.float32))
    if not (reward3 == pytest.approx(1.0 + 2.0 * 0.08)):
        raise AssertionError


def test_drop_below_threshold_terminates(scene_xml: Path) -> None:
    """Verify that dropping the object too far below its initial height triggers terminal flags."""
    env = MuJoCoGraspingEnv(
        scene_xml,
        object_name="object",
        reward_config=RewardConfig(drop_height_threshold=0.1),
    )
    env.reset(seed=42)
    initial = env._initial_object_height  # noqa: SLF001  # read internal baseline height
    env._get_object_height = lambda: initial - 0.2  # type: ignore[method-assign]  # noqa: SLF001  # stub internal height probe
    _obs, _reward, terminated, truncated, _info = env.step(np.array([0.0], dtype=np.float32))
    if terminated is not True:
        raise AssertionError
    if truncated is not False:
        raise AssertionError


def test_env_validates_reward_configuration(scene_xml: Path) -> None:
    """Verify that MuJoCoGraspingEnv validates custom RewardConfig and object name argument types."""
    with pytest.raises(TypeError, match="reward_config"):
        MuJoCoGraspingEnv(scene_xml, object_name="object", reward_config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="object_name"):
        MuJoCoGraspingEnv(scene_xml, object_name=123)  # type: ignore[arg-type]


def test_non_finite_terminal_uses_config(panda_robot_xml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that non-finite observations do not trigger termination if config disables it."""
    env = MuJoCoGraspingEnv(
        panda_robot_xml,
        reward_config=RewardConfig(terminate_on_non_finite=False),
    )
    env.reset(seed=42)
    monkeypatch.setattr(env, "_get_observation", lambda: np.full(18, np.nan, dtype=np.float32))
    _obs, _reward, terminated, _truncated, _info = env.step(np.array([0.0], dtype=np.float32))
    if terminated is not False:
        raise AssertionError


def test_env_step_routes_through_shared_command_path(panda_robot_xml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that step actions route control commands directly to MuJoCo actuators."""
    calls: list[np.ndarray] = []

    def spy_set_actuator_controls(state: object, ctrl: np.ndarray) -> None:
        calls.append(np.asarray(ctrl, copy=True))
        return set_actuator_controls(state, ctrl)

    monkeypatch.setattr(mujoco_env_module, "set_actuator_controls", spy_set_actuator_controls)

    env = MuJoCoGraspingEnv(panda_robot_xml)
    env.reset(seed=42)
    env.step(np.array([0.5], dtype=np.float32))

    if not (len(calls) == 1):
        raise AssertionError
    expected = np.zeros(8, dtype=np.float32)
    expected[0] = 0.5
    if not (np.allclose(calls[0], expected)):
        raise AssertionError


@pytest.mark.filterwarnings("ignore")
def test_gymnasium_env_compliance(panda_robot_xml: Path) -> None:
    """Verify gymnasium compliance checks pass on the MuJoCo grasping environment."""
    env = MuJoCoGraspingEnv(panda_robot_xml)
    check_env(env)


def test_sb3_training_and_inference_compatibility(panda_robot_xml: Path, tmp_path: Path) -> None:
    """Train a tiny Panda RL policy and run a one-step inference call.

    Args:
        panda_robot_xml: Path to ``deploy/robot.xml``.
        tmp_path: Temporary directory for the checkpoint and logs.
    """
    checkpoint_path = tmp_path / "policy.pt"
    log_dir = tmp_path / "tb_logs"

    run_rl_training_pipeline(
        robot_xml_path=panda_robot_xml,
        ycb_root=tmp_path,
        object_ids=[],
        policy_checkpoint_path=checkpoint_path,
        observation_dim=18,
        action_dim=8,
        hidden_dim=16,
        learning_rate=1e-3,
        num_updates=1,
        gamma=0.99,
        device="cpu",
        seed=42,
        experiment_log_dir=log_dir,
    )

    if not (checkpoint_path.is_file()):
        raise AssertionError

    checkpoint = load_rl_policy_checkpoint(checkpoint_path, "cpu")
    if "model_state_dict" not in checkpoint:
        raise AssertionError

    runner = build_rl_policy_runner(
        checkpoint=checkpoint,
        observation_dim=18,
        action_dim=8,
        device="cpu",
    )

    obs = np.zeros(18, dtype=np.float32)
    action = runner(obs)

    if not (isinstance(action, np.ndarray)):
        raise TypeError
    if not (action.shape == (8,)):
        raise AssertionError
    if not (np.isfinite(action).all()):
        raise AssertionError


def test_mujoco_env_additional_coverage(tmp_path: Path) -> None:
    """Verify model loading failures, XML errors, and custom control ranges in MuJoCo grasping environment."""
    import mujoco  # noqa: PLC0415  # deferred heavy import

    corrupt_xml = tmp_path / "corrupt.xml"
    corrupt_xml.write_text("<invalid_xml_tag", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to load MuJoCo model from XML"):
        load_mujoco_model(corrupt_xml)

    valid_xml = tmp_path / "valid.xml"
    valid_xml.write_text(
        '<mujoco model="test"><worldbody><body name="b"><joint name="j1" type="slide"/>'
        '<geom type="box" size="0.1 0.1 0.1"/></body></worldbody>'
        '<actuator><position joint="j1"/></actuator></mujoco>',
        encoding="utf-8",
    )
    mj_model = mujoco.MjModel.from_xml_path(str(valid_xml))
    _state, _step_fn, contacts_fn = create_simulation(mj_model)
    if not (contacts_fn() == []):
        raise AssertionError

    with pytest.raises(TypeError, match=r"robot_xml_path must be a pathlib\.Path"):
        MuJoCoGraspingEnv("not_a_path")  # type: ignore[arg-type]

    no_act_xml = tmp_path / "no_act.xml"
    no_act_xml.write_text(
        '<mujoco model="no_act"><worldbody><body name="b">'
        '<geom type="box" size="0.1 0.1 0.1"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="MuJoCo model has zero actuators"):
        MuJoCoGraspingEnv(no_act_xml)

    ctrl_xml = tmp_path / "ctrl.xml"
    ctrl_xml.write_text(
        '<mujoco model="ctrl"><worldbody><body name="b"><joint name="j1" type="slide"/>'
        '<geom type="box" size="0.1 0.1 0.1"/></body></worldbody>'
        '<actuator><position joint="j1" ctrlrange="-0.5 0.5"/></actuator></mujoco>',
        encoding="utf-8",
    )
    env = MuJoCoGraspingEnv(ctrl_xml)
    if not (env.action_space.low[0] == pytest.approx(-0.5)):
        raise AssertionError
    if not (env.action_space.high[0] == pytest.approx(0.5)):
        raise AssertionError
