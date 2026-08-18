"""Phase 9 Gymnasium environment tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import grasping_ai.simulation.mujoco_env as mujoco_env_module
from grasping_ai.simulation.mujoco_env import (
    MuJoCoGraspingEnv,
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


def test_default_reward_uses_configured_action_cost_and_survival_bonus(panda_robot_xml: Path) -> None:
    """Verify configured action cost and survival terms without hard-coded legacy reward values."""
    env = MuJoCoGraspingEnv(panda_robot_xml)
    env.reset(seed=42)
    _obs, reward, terminated, truncated, _info = env.step(np.array([0.5], dtype=np.float32))
    expected = mujoco_env_module.SURVIVAL_BONUS - mujoco_env_module.ACTION_COST_WEIGHT * 0.25
    if not (reward == pytest.approx(expected)):
        raise AssertionError
    if terminated is not False:
        raise AssertionError
    if truncated is not False:
        raise AssertionError


def test_contact_reward_term(scene_xml: Path) -> None:
    """Verify that contact bonuses are granted once on contact acquisition."""
    env = MuJoCoGraspingEnv(scene_xml, object_name="object")
    env.reset(seed=42)
    env._has_object_contact = lambda: True  # type: ignore[method-assign]  # noqa: SLF001  # stub internal contact probe
    env._fingertip_object_contacts = lambda: (0, False)  # type: ignore[method-assign]  # noqa: SLF001
    env._hand_object_distance = lambda: 0.0  # type: ignore[method-assign]  # noqa: SLF001
    env._previous_hand_object_distance = 0.0  # noqa: SLF001
    env._max_fingertip_object_distance = lambda: 0.0  # type: ignore[method-assign]  # noqa: SLF001
    env._previous_fingertip_object_distance = 0.0  # noqa: SLF001
    env._get_object_height = lambda: env._initial_object_height  # type: ignore[method-assign]  # noqa: SLF001
    _obs, reward, terminated, _truncated, _info = env.step(np.array([0.0], dtype=np.float32))
    if not (reward == pytest.approx(mujoco_env_module.SURVIVAL_BONUS + mujoco_env_module.CONTACT_REWARD)):
        raise AssertionError
    if terminated is not False:
        raise AssertionError
    _obs, held_reward, _terminated, _truncated, _info = env.step(np.array([0.0], dtype=np.float32))
    if not (held_reward == pytest.approx(mujoco_env_module.SURVIVAL_BONUS)):
        raise AssertionError


def test_lift_and_grasp_success_rewards(scene_xml: Path) -> None:
    """Verify that lifting the object above thresholds triggers lift reward weights and grasp success bonuses."""
    env = MuJoCoGraspingEnv(
        scene_xml,
        object_name="object",
    )
    env.reset(seed=42)
    initial = env._initial_object_height  # noqa: SLF001  # read internal baseline height
    env._has_object_contact = lambda: False  # type: ignore[method-assign]  # noqa: SLF001  # isolate lift rewards
    env._fingertip_object_contacts = lambda: (0.0, False)  # type: ignore[method-assign]  # noqa: SLF001
    env._hand_object_distance = lambda: 0.0  # type: ignore[method-assign]  # noqa: SLF001
    env._previous_hand_object_distance = 0.0  # noqa: SLF001
    env._max_fingertip_object_distance = lambda: 0.0  # type: ignore[method-assign]  # noqa: SLF001
    env._previous_fingertip_object_distance = 0.0  # noqa: SLF001

    env._get_object_height = lambda: initial + 0.03  # type: ignore[method-assign]  # noqa: SLF001  # stub internal height probe
    _obs, reward1, _term, _trunc, _info = env.step(np.array([0.0], dtype=np.float32))
    if not (reward1 == pytest.approx(mujoco_env_module.SURVIVAL_BONUS)):
        raise AssertionError

    env._fingertip_object_contacts = lambda: (2.0, True)  # type: ignore[method-assign]  # noqa: SLF001
    env._get_object_height = lambda: initial + 0.08  # type: ignore[method-assign]  # noqa: SLF001  # stub internal height probe
    _obs, reward2, _term, _trunc, _info = env.step(np.array([0.0], dtype=np.float32))
    expected_reward2 = (
        mujoco_env_module.SURVIVAL_BONUS
        + mujoco_env_module.BILATERAL_CONTACT_REWARD
        + mujoco_env_module.LIFT_REWARD_WEIGHT * 0.05
        + mujoco_env_module.GRASP_SUCCESS_BONUS
    )
    expected_reward2 = float(
        np.clip(expected_reward2, mujoco_env_module.REWARD_CLIP_MIN, mujoco_env_module.REWARD_CLIP_MAX),
    )
    if not (reward2 == pytest.approx(expected_reward2)):
        raise AssertionError

    _obs, reward3, _term, _trunc, _info = env.step(np.array([0.0], dtype=np.float32))
    if not (
        reward3
        == pytest.approx(
            mujoco_env_module.SURVIVAL_BONUS
            + mujoco_env_module.BILATERAL_HOLD_REWARD,
        )
    ):
        raise AssertionError


def test_drop_below_threshold_terminates(scene_xml: Path) -> None:
    """Verify that dropping the object too far below its initial height triggers terminal flags."""
    env = MuJoCoGraspingEnv(
        scene_xml,
        object_name="object",
    )
    env.reset(seed=42)
    initial = env._initial_object_height  # noqa: SLF001  # read internal baseline height
    env._get_object_height = lambda: initial - 0.2  # type: ignore[method-assign]  # noqa: SLF001  # stub internal height probe
    _obs, _reward, terminated, truncated, _info = env.step(np.array([0.0], dtype=np.float32))
    if terminated is not True:
        raise AssertionError
    if truncated is not False:
        raise AssertionError


def test_env_validates_object_name(scene_xml: Path) -> None:
    """Verify that MuJoCoGraspingEnv validates the object name argument type."""
    with pytest.raises(TypeError, match="object_name"):
        MuJoCoGraspingEnv(scene_xml, object_name=123)  # type: ignore[arg-type]


def test_non_finite_observations_terminate(panda_robot_xml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that non-finite observations trigger termination."""
    env = MuJoCoGraspingEnv(panda_robot_xml)
    env.reset(seed=42)
    monkeypatch.setattr(env, "_get_observation", lambda: np.full(18, np.nan, dtype=np.float32))
    _obs, _reward, terminated, _truncated, _info = env.step(np.array([0.0], dtype=np.float32))
    if terminated is not True:
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


def test_normalized_delta_control_holds_pose_and_scales_commands(panda_robot_xml: Path) -> None:
    """Verify zero holds the current pose and normalized actions increment actuator targets."""
    env = MuJoCoGraspingEnv(panda_robot_xml, control_mode="normalized_delta")
    env.reset(seed=42)
    initial_target = np.array(env._control_target, copy=True)  # noqa: SLF001

    if not np.allclose(env.action_space.low, -1.0) or not np.allclose(env.action_space.high, 1.0):
        raise AssertionError

    _obs, _reward, _terminated, _truncated, info = env.step(np.zeros(8, dtype=np.float32))
    if not np.allclose(info["actuator_control"], initial_target):
        raise AssertionError

    action = np.zeros(8, dtype=np.float32)
    action[0] = 1.0
    action[7] = -1.0
    _obs, _reward, _terminated, _truncated, info = env.step(action)
    expected = np.array(initial_target, copy=True)
    expected[0] += mujoco_env_module.ARM_DELTA_SCALE
    expected[7] -= mujoco_env_module.GRIPPER_DELTA_SCALE
    ctrl_range = env._state["model"].actuator_ctrlrange  # noqa: SLF001
    expected = np.clip(expected, ctrl_range[:, 0], ctrl_range[:, 1])
    if not np.allclose(info["actuator_control"], expected):
        raise AssertionError


def test_task_observations_append_grasp_geometry(scene_xml: Path) -> None:
    """Verify task observations expose finite hand/finger/object geometry."""
    env = MuJoCoGraspingEnv(scene_xml, object_name="object", task_observations=True)
    obs, _info = env.reset(seed=42)
    if obs.shape != (39,):
        raise AssertionError
    if not np.isfinite(obs[-mujoco_env_module.TASK_OBSERVATION_SIZE :]).all():
        raise AssertionError


@pytest.mark.filterwarnings("ignore")
def test_gymnasium_env_compliance(panda_robot_xml: Path) -> None:
    """Verify gymnasium compliance checks pass on the MuJoCo grasping environment."""
    env = MuJoCoGraspingEnv(panda_robot_xml)
    check_env(env)


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
