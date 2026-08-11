"""Phase 5 — Reinforcement Learning Policy tests."""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import grasping_ai
from grasping_ai.inference.policy_runner import (
    build_rl_policy_runner,
    load_rl_policy_checkpoint,
    run_policy_step,
)
from grasping_ai.models.rl_policy import (
    build_policy_network,
    build_value_network,
    select_action,
)
from grasping_ai.pipelines.train_rl import (
    build_rl_environment,
    collect_rl_rollout,
    run_rl_training_pipeline,
)
from grasping_ai.training.rl_trainer import (
    build_rl_training_step,
    compute_discounted_returns,
    compute_gae_advantages,
    run_rl_training_loop,
)

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
        <motor name="motor1" joint="joint1" gear="1"/>
    </actuator>
</mujoco>
"""


def _write_robot_xml(tmp_path: Path) -> Path:
    path = tmp_path / "robot.xml"
    path.write_text(MINIMAL_ACTUATED_ROBOT_XML, encoding="utf-8")
    return path


def test_phase1_package_import_remains_stable():
    """Verify that grasping_ai is importable."""
    assert grasping_ai.__name__ == "grasping_ai"


def test_stable_baselines_dependency_available():
    """Verify that Stable-Baselines3 can be imported."""
    import stable_baselines3
    assert stable_baselines3.__version__ is not None


def test_training_config_file_exists():
    """Verify that configs/training.yaml exists."""
    config_path = os.path.join("configs", "training.yaml")
    assert os.path.isfile(config_path)


def test_policy_network_forward_shape():
    """Verify policy network output has correct shape."""
    policy = build_policy_network(4, 2, 16, 2)
    obs = torch.randn(3, 4)
    out = policy(obs)
    assert out.shape == (3, 2)


def test_value_network_forward_shape():
    """Verify value network output has correct shape."""
    value_net = build_value_network(4, 16, 2)
    obs = torch.randn(3, 4)
    out = value_net(obs)
    assert out.shape == (3, 1)


def test_policy_network_rejects_invalid_dims():
    """Verify policy network raises on invalid dimensions."""
    with pytest.raises(ValueError):
        build_policy_network(0, 2, 16, 2)
    with pytest.raises(ValueError):
        build_policy_network(4, -1, 16, 2)


def test_select_action_shape():
    """Verify select_action produces correct output shape."""
    policy = build_policy_network(4, 2, 16, 2)
    obs = torch.randn(3, 4)
    rng = torch.Generator()
    rng.manual_seed(42)
    action = select_action(policy, obs, rng)
    assert action.shape == (3, 2)


def test_select_action_rejects_invalid_observation():
    """Verify select_action raises on bad observation shape."""
    policy = build_policy_network(4, 2, 16, 2)
    obs_1d = torch.randn(4)
    rng = torch.Generator()
    with pytest.raises(ValueError, match="observation must have shape"):
        select_action(policy, obs_1d, rng)


def test_compute_discounted_returns_basic():
    """Verify discounted returns computation."""
    transitions = [
        (np.zeros(2), np.zeros(1), 1.0, np.zeros(2), False),
        (np.zeros(2), np.zeros(1), 1.0, np.zeros(2), False),
        (np.zeros(2), np.zeros(1), 1.0, np.zeros(2), True),
    ]
    returns = compute_discounted_returns(transitions, gamma=0.99)
    assert returns.shape == (3,)
    assert np.isfinite(returns).all()
    assert returns[0] > returns[2]


def test_compute_discounted_returns_rejects_invalid_gamma():
    """Verify discounted returns raises on bad gamma."""
    transitions = [(np.zeros(2), np.zeros(1), 1.0, np.zeros(2), False)]
    with pytest.raises(ValueError):
        compute_discounted_returns(transitions, gamma=-0.1)
    with pytest.raises(ValueError):
        compute_discounted_returns(transitions, gamma=1.5)


def test_compute_gae_advantages_basic():
    """Verify GAE computation produces finite results."""
    transitions = [
        (np.zeros(2), np.zeros(1), 1.0, np.zeros(2), False),
        (np.zeros(2), np.zeros(1), 1.0, np.zeros(2), True),
    ]
    advantages, returns = compute_gae_advantages(
        transitions, value_fn=lambda obs: 0.0, gamma=0.99, gae_lambda=0.95,
    )
    assert advantages.shape == (2,)
    assert returns.shape == (2,)
    assert np.isfinite(advantages).all()
    assert np.isfinite(returns).all()


def test_build_rl_training_step_runs():
    """Verify RL training step executes without error."""
    policy = build_policy_network(4, 2, 8, 1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)  # type: ignore[union-attr]
    step_fn = build_rl_training_step(policy, optimizer, 0.2, 0.0, "cpu")

    transitions = [
        (np.random.randn(4).astype(np.float32), np.random.randn(2).astype(np.float32), 1.0, np.random.randn(4).astype(np.float32), False)
        for _ in range(10)
    ]
    metrics = step_fn(transitions)
    assert "loss" in metrics
    assert np.isfinite(metrics["loss"])


def test_run_rl_training_loop_saves_checkpoint():
    """Verify RL training loop saves a checkpoint."""
    policy = build_policy_network(4, 2, 8, 1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)  # type: ignore[union-attr]
    step_fn = build_rl_training_step(policy, optimizer, 0.2, 0.0, "cpu")

    def rollout_gen():
        while True:
            yield [
                (np.random.randn(4).astype(np.float32), np.random.randn(2).astype(np.float32), 1.0, np.random.randn(4).astype(np.float32), False)
                for _ in range(5)
            ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "rl_policy.pt"
        run_rl_training_loop(step_fn, iter(rollout_gen()), 2, ckpt_path, 1)
        assert ckpt_path.exists()

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        assert "model_state_dict" in checkpoint
        assert "epoch" in checkpoint


def test_run_rl_pipeline_rejects_missing_robot_xml():
    """Verify pipeline raises on missing robot XML."""
    with tempfile.TemporaryDirectory() as tmp_dir, pytest.raises(FileNotFoundError):
        run_rl_training_pipeline(
            robot_xml_path=Path(tmp_dir) / "nonexistent.xml",
            ycb_root=Path(tmp_dir),
            object_ids=[],
            policy_checkpoint_path=Path(tmp_dir) / "policy.pt",
            observation_dim=4,
            action_dim=2,
            hidden_dim=8,
            learning_rate=1e-3,
            num_updates=1,
            gamma=0.99,
            device="cpu",
        )


def test_run_rl_pipeline_rejects_missing_ycb_root_when_objects_requested():
    """Verify pipeline raises when YCB root is missing but objects requested."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        robot_xml = _write_robot_xml(Path(tmp_dir))
        with pytest.raises(FileNotFoundError):
            run_rl_training_pipeline(
                robot_xml_path=robot_xml,
                ycb_root=Path(tmp_dir) / "missing_ycb",
                object_ids=["obj001"],
                policy_checkpoint_path=Path(tmp_dir) / "policy.pt",
                observation_dim=2,
                action_dim=1,
                hidden_dim=8,
                learning_rate=1e-3,
                num_updates=1,
                gamma=0.99,
                device="cpu",
            )


def test_run_rl_pipeline_initializes_environment():
    """Verify pipeline can initialize the simulation environment."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        robot_xml = _write_robot_xml(Path(tmp_dir))
        env_state, sim_step, contacts = build_rl_environment(
            robot_xml, Path(tmp_dir), "default", 2, 1,
        )
        assert isinstance(env_state, dict)
        assert callable(sim_step)
        assert callable(contacts)


def test_run_rl_pipeline_validates_observation_dim():
    """Verify pipeline raises on observation dim mismatch."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        robot_xml = _write_robot_xml(Path(tmp_dir))
        with pytest.raises(ValueError, match="observation_dim"):
            run_rl_training_pipeline(
                robot_xml_path=robot_xml,
                ycb_root=Path(tmp_dir),
                object_ids=[],
                policy_checkpoint_path=Path(tmp_dir) / "policy.pt",
                observation_dim=999,
                action_dim=1,
                hidden_dim=8,
                learning_rate=1e-3,
                num_updates=1,
                gamma=0.99,
                device="cpu",
            )


def test_run_rl_pipeline_validates_action_dim():
    """Verify pipeline raises on action dim mismatch."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        robot_xml = _write_robot_xml(Path(tmp_dir))
        with pytest.raises(ValueError, match="action_dim"):
            run_rl_training_pipeline(
                robot_xml_path=robot_xml,
                ycb_root=Path(tmp_dir),
                object_ids=[],
                policy_checkpoint_path=Path(tmp_dir) / "policy.pt",
                observation_dim=2,
                action_dim=999,
                hidden_dim=8,
                learning_rate=1e-3,
                num_updates=1,
                gamma=0.99,
                device="cpu",
            )


def test_env_step_reward_is_finite():
    """Verify that environment rollout produces finite rewards."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        robot_xml = _write_robot_xml(Path(tmp_dir))
        env_state, _, _ = build_rl_environment(
            robot_xml, Path(tmp_dir), "default", 2, 1,
        )
        transitions = collect_rl_rollout(
            env_state,
            lambda obs: np.zeros(1, dtype=np.float32),
            num_steps=5,
        )
        assert len(transitions) == 5
        for _obs, _act, reward, _next_obs, _done in transitions:
            assert np.isfinite(reward)


def test_run_rl_pipeline_performs_minimal_training_and_saves_checkpoint():
    """Verify full RL pipeline trains and saves checkpoint."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        robot_xml = _write_robot_xml(Path(tmp_dir))
        ckpt_path = Path(tmp_dir) / "policy.pt"

        run_rl_training_pipeline(
            robot_xml_path=robot_xml,
            ycb_root=Path(tmp_dir),
            object_ids=[],
            policy_checkpoint_path=ckpt_path,
            observation_dim=2,
            action_dim=1,
            hidden_dim=8,
            learning_rate=1e-3,
            num_updates=1,
            gamma=0.99,
            device="cpu",
        )
        assert ckpt_path.exists()


def test_rl_checkpoint_is_loadable_or_discoverable():
    """Verify saved checkpoint can be loaded back."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        robot_xml = _write_robot_xml(Path(tmp_dir))
        ckpt_path = Path(tmp_dir) / "policy.pt"

        run_rl_training_pipeline(
            robot_xml_path=robot_xml,
            ycb_root=Path(tmp_dir),
            object_ids=[],
            policy_checkpoint_path=ckpt_path,
            observation_dim=2,
            action_dim=1,
            hidden_dim=8,
            learning_rate=1e-3,
            num_updates=1,
            gamma=0.99,
            device="cpu",
        )

        checkpoint = load_rl_policy_checkpoint(ckpt_path, "cpu")
        assert isinstance(checkpoint, dict)
        assert "model_state_dict" in checkpoint

        runner = build_rl_policy_runner(checkpoint, 2, 1, "cpu")
        obs = np.random.randn(2).astype(np.float32)
        action = run_policy_step(runner, obs)
        assert action.shape == (1,)
        assert np.isfinite(action).all()


def test_rl_training_does_not_leak_global_state():
    """Verify that two pipeline initializations are independent."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        robot_xml = _write_robot_xml(Path(tmp_dir))

        env1, _, _ = build_rl_environment(
            robot_xml, Path(tmp_dir), "default", 2, 1,
        )
        env2, _, _ = build_rl_environment(
            robot_xml, Path(tmp_dir), "default", 2, 1,
        )

        assert env1 is not env2
        assert env1["data"] is not env2["data"]  # type: ignore[index]
