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
    RL_CHECKPOINT_FORMAT_VERSION,
    RL_POLICY_ARCHITECTURE,
    build_policy_network,
    build_value_network,
    read_rl_policy_metadata,
    save_rl_policy_checkpoint,
    select_action,
)
from grasping_ai.pipelines.train_rl import (
    run_rl_training_pipeline,
)


def _write_ycb_object_xml(tmp_path: Path) -> Path:
    """Helper function to write a minimal YCB object MJCF XML file for testing."""
    object_dir = tmp_path / "ycb" / "006_mustard_bottle"
    object_dir.mkdir(parents=True)
    xml_content = """\
<mujoco model="mustard_bottle">
    <worldbody>
        <body name="mustard_bottle_body" pos="0 0 0.5">
            <freejoint/>
            <geom name="mustard_bottle_geom" type="sphere" size="0.05"/>
        </body>
    </worldbody>
</mujoco>
"""
    path = object_dir / "mustard_bottle.xml"
    path.write_text(xml_content, encoding="utf-8")
    return path


def test_phase1_package_import_remains_stable():
    """Verify that grasping_ai is importable."""
    assert grasping_ai.__name__ == "grasping_ai"


def test_stable_baselines_dependency_available():
    """Verify that Stable-Baselines3 can be imported."""
    import stable_baselines3

    assert stable_baselines3.__version__ is not None


def test_training_config_files_exist():
    """Verify training config default alias and method variants exist."""
    assert os.path.isfile(os.path.join("configs", "training", "default.yaml"))
    assert os.path.isfile(os.path.join("configs", "training", "diffusion.yaml"))
    assert os.path.isfile(os.path.join("configs", "training", "flow.yaml"))


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


def test_run_rl_pipeline_rejects_missing_ycb_root_when_objects_requested(
    panda_robot_xml: Path,
):
    """Verify pipeline raises when YCB root is missing but objects requested.

    Args:
        panda_robot_xml: Path to ``deploy/robot.xml``.
    """
    with tempfile.TemporaryDirectory() as tmp_dir, pytest.raises(FileNotFoundError):
        run_rl_training_pipeline(
            robot_xml_path=panda_robot_xml,
            ycb_root=Path(tmp_dir) / "missing_ycb",
            object_ids=["obj001"],
            policy_checkpoint_path=Path(tmp_dir) / "policy.pt",
            observation_dim=31,
            action_dim=8,
            hidden_dim=8,
            learning_rate=1e-3,
            num_updates=1,
            gamma=0.99,
            device="cpu",
        )


def test_run_rl_pipeline_validates_observation_dim(panda_robot_xml: Path):
    """Verify pipeline raises on observation dim mismatch.

    Args:
        panda_robot_xml: Path to ``deploy/robot.xml``.
    """
    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        pytest.raises(ValueError, match="observation_dim"),
    ):
        run_rl_training_pipeline(
            robot_xml_path=panda_robot_xml,
            ycb_root=Path(tmp_dir),
            object_ids=[],
            policy_checkpoint_path=Path(tmp_dir) / "policy.pt",
            observation_dim=999,
            action_dim=8,
            hidden_dim=8,
            learning_rate=1e-3,
            num_updates=1,
            gamma=0.99,
            device="cpu",
        )


def test_run_rl_pipeline_validates_action_dim(panda_robot_xml: Path):
    """Verify pipeline raises on action dim mismatch.

    Args:
        panda_robot_xml: Path to ``deploy/robot.xml``.
    """
    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        pytest.raises(ValueError, match="action_dim"),
    ):
        run_rl_training_pipeline(
            robot_xml_path=panda_robot_xml,
            ycb_root=Path(tmp_dir),
            object_ids=[],
            policy_checkpoint_path=Path(tmp_dir) / "policy.pt",
            observation_dim=18,
            action_dim=999,
            hidden_dim=8,
            learning_rate=1e-3,
            num_updates=1,
            gamma=0.99,
            device="cpu",
        )


def test_run_rl_pipeline_performs_minimal_training_and_saves_checkpoint(
    panda_robot_xml: Path,
):
    """Verify full RL pipeline trains and saves checkpoint."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "policy.pt"

        run_rl_training_pipeline(
            robot_xml_path=panda_robot_xml,
            ycb_root=Path(tmp_dir),
            object_ids=[],
            policy_checkpoint_path=ckpt_path,
            observation_dim=18,
            action_dim=8,
            hidden_dim=8,
            learning_rate=1e-3,
            num_updates=1,
            gamma=0.99,
            device="cpu",
        )
        assert ckpt_path.exists()


def test_rl_checkpoint_is_loadable_or_discoverable(panda_robot_xml: Path):
    """Verify saved checkpoint can be loaded back."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "policy.pt"

        run_rl_training_pipeline(
            robot_xml_path=panda_robot_xml,
            ycb_root=Path(tmp_dir),
            object_ids=[],
            policy_checkpoint_path=ckpt_path,
            observation_dim=18,
            action_dim=8,
            hidden_dim=8,
            learning_rate=1e-3,
            num_updates=1,
            gamma=0.99,
            device="cpu",
        )

        checkpoint = load_rl_policy_checkpoint(ckpt_path, "cpu")
        assert isinstance(checkpoint, dict)
        assert "model_state_dict" in checkpoint

        runner = build_rl_policy_runner(checkpoint, 18, 8, "cpu")
        obs = np.random.randn(18).astype(np.float32)
        action = run_policy_step(runner, obs)
        assert action.shape == (8,)
        assert np.isfinite(action).all()


def test_run_rl_pipeline_loads_requested_object(panda_robot_xml: Path):
    """Verify the pipeline loads the requested object into the environment."""
    from grasping_ai.simulation.mujoco_env import MuJoCoGraspingEnv
    from grasping_ai.simulation.scene import build_scene_xml
    from grasping_ai.simulation.ycb import resolve_ycb_object_directory

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        object_xml = _write_ycb_object_xml(tmp_path)
        ycb_root = tmp_path / "ycb"

        object_dir = resolve_ycb_object_directory(ycb_root, "mustard_bottle")
        assert object_dir.is_dir()

        scene_xml = build_scene_xml(panda_robot_xml, object_xml, None)
        env = MuJoCoGraspingEnv(scene_xml)

        state_dict = env._state  # type: ignore[attr-defined]
        mj_model = state_dict["model"]
        import mujoco  # type: ignore[import-untyped]

        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "mustard_bottle_body")
        assert body_id != -1

        obs_dim = env.observation_space.shape[0]
        assert obs_dim == 31
        act_dim = env.action_space.shape[0]
        assert act_dim == 8

        ckpt_path = tmp_path / "policy.pt"
        run_rl_training_pipeline(
            robot_xml_path=panda_robot_xml,
            ycb_root=ycb_root,
            object_ids=["mustard_bottle"],
            policy_checkpoint_path=ckpt_path,
            observation_dim=obs_dim,
            action_dim=act_dim,
            hidden_dim=8,
            learning_rate=1e-3,
            num_updates=1,
            gamma=0.99,
            device="cpu",
        )
        assert ckpt_path.exists()


def test_rl_pipeline_tracks_object_and_enables_grasp_rewards(monkeypatch, panda_robot_xml: Path):
    """Verify the RL pipeline propagates the object name and enables grasp rewards."""
    from grasping_ai.simulation.mujoco_env import MuJoCoGraspingEnv, RewardConfig
    from grasping_ai.simulation.ycb import resolve_ycb_object_directory

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        _write_ycb_object_xml(tmp_path)
        ycb_root = tmp_path / "ycb"
        resolve_ycb_object_directory(ycb_root, "mustard_bottle")

        captured = {}

        def recording_env(*args, **kwargs):
            captured["object_name"] = kwargs.get("object_name")
            captured["reward_config"] = kwargs.get("reward_config")
            return MuJoCoGraspingEnv(*args, **kwargs)

        monkeypatch.setattr(
            "grasping_ai.simulation.mujoco_env.MuJoCoGraspingEnv",
            recording_env,
        )

        run_rl_training_pipeline(
            robot_xml_path=panda_robot_xml,
            ycb_root=ycb_root,
            object_ids=["mustard_bottle"],
            policy_checkpoint_path=tmp_path / "policy.pt",
            observation_dim=31,
            action_dim=8,
            hidden_dim=8,
            learning_rate=1e-3,
            num_updates=1,
            gamma=0.99,
            device="cpu",
        )

        assert captured["object_name"] == "mustard_bottle"
        reward_config = captured["reward_config"]
        assert isinstance(reward_config, RewardConfig)
        assert reward_config.contact_reward > 0.0
        assert reward_config.lift_reward_weight > 0.0
        assert reward_config.grasp_success_bonus > 0.0


def test_save_rl_policy_checkpoint_writes_metadata():
    """Verify that save_rl_policy_checkpoint writes valid metadata and formats state dict properly."""
    policy = build_policy_network(4, 2, 16, 2)
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "policy.pt"
        save_rl_policy_checkpoint(
            policy,
            checkpoint_path,
            epoch=5,
            observation_dim=4,
            action_dim=2,
            hidden_dim=16,
            num_layers=2,
            seed=42,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert checkpoint["format_version"] == RL_CHECKPOINT_FORMAT_VERSION
        assert checkpoint["architecture"] == RL_POLICY_ARCHITECTURE
        assert checkpoint["observation_dim"] == 4
        assert checkpoint["action_dim"] == 2
        assert checkpoint["hidden_dim"] == 16
        assert checkpoint["num_layers"] == 2
        assert checkpoint["epoch"] == 5
        assert checkpoint["seed"] == 42
        assert "model_state_dict" in checkpoint

        from grasping_ai.training.checkpoint_io import read_model_checkpoint_metadata

        metadata = read_model_checkpoint_metadata(checkpoint_path, "cpu")
        assert metadata["kind"] == "rl_policy"
        assert metadata["observation_dim"] == 4
        assert metadata["action_dim"] == 2
        assert metadata["hidden_dim"] == 16
        assert metadata["num_layers"] == 2
        assert metadata["epoch"] == 5
        assert metadata["seed"] == 42


def test_read_rl_policy_metadata():
    """Verify that read_rl_policy_metadata successfully parses dimensions and layers from standard checkpoints."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "policy.pt"
        policy = build_policy_network(3, 1, 8, 2)
        save_rl_policy_checkpoint(
            policy,
            checkpoint_path,
            1,
            3,
            1,
            8,
            2,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        metadata = read_rl_policy_metadata(checkpoint)
        assert metadata == (3, 1, 8, 2)


def test_read_rl_policy_metadata_legacy_checkpoint_returns_none():
    """Verify that read_rl_policy_metadata returns None for legacy checkpoints lacking explicit metadata headers."""
    assert read_rl_policy_metadata({}) is None
    assert read_rl_policy_metadata({"model_state_dict": {}}) is None
    assert read_rl_policy_metadata("not-a-dict") is None  # type: ignore[arg-type]


def test_runner_loads_standard_checkpoint():
    """Verify that the policy runner can load and execute inference on standard checkpoints with metadata."""
    policy = build_policy_network(4, 2, 16, 2)
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "policy.pt"
        save_rl_policy_checkpoint(
            policy,
            checkpoint_path,
            1,
            4,
            2,
            16,
            2,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        runner = build_rl_policy_runner(checkpoint, 4, 2, "cpu")
        obs = np.random.randn(4).astype(np.float32)
        action = runner(obs)
        assert action.shape == (2,)
        assert np.isfinite(action).all()


def test_runner_rejects_dimension_mismatch():
    """Verify that the policy runner raises ValueError if observation or action dimensions mismatch the checkpoint metadata."""
    policy = build_policy_network(4, 2, 16, 2)
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "policy.pt"
        save_rl_policy_checkpoint(
            policy,
            checkpoint_path,
            1,
            4,
            2,
            16,
            2,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        with pytest.raises(ValueError, match="observation_dim"):
            build_rl_policy_runner(checkpoint, 5, 2, "cpu")
        with pytest.raises(ValueError, match="action_dim"):
            build_rl_policy_runner(checkpoint, 4, 3, "cpu")


def test_runner_loads_legacy_checkpoint_without_metadata():
    """Verify that policy runner can load legacy checkpoints that lack explicit metadata using inferred sizes."""
    # A legacy checkpoint (model_state_dict only) is still loadable by
    # inferring architecture from parameter names.
    policy = build_policy_network(4, 2, 8, 2)
    legacy = {"epoch": 1, "model_state_dict": policy.state_dict()}
    runner = build_rl_policy_runner(legacy, 4, 2, "cpu")
    obs = np.random.randn(4).astype(np.float32)
    action = runner(obs)
    assert action.shape == (2,)


def test_save_rl_policy_checkpoint_validation():
    """Verify that saving checkpoints validates path types, observation, and action dimensions strictly."""
    policy = build_policy_network(4, 2, 16, 2)
    with pytest.raises(ValueError, match="observation_dim"):
        save_rl_policy_checkpoint(policy, Path("p.pt"), 1, 0, 2, 16, 2)
    with pytest.raises(ValueError, match="action_dim"):
        save_rl_policy_checkpoint(policy, Path("p.pt"), 1, 4, -1, 16, 2)
    with pytest.raises(TypeError, match="policy_checkpoint_path"):
        save_rl_policy_checkpoint(policy, "p.pt", 1, 4, 2, 16, 2)  # type: ignore[arg-type]


def test_select_action_noise_scale_controls_exploration():
    """Verify that select_action noise scale controls stochastic exploration vs deterministic policy execution."""
    policy = build_policy_network(4, 2, 16, 2)
    obs = torch.randn(3, 4)

    with_noise = select_action(policy, obs, torch.Generator().manual_seed(1), noise_scale=0.1)
    with_more_noise = select_action(policy, obs, torch.Generator().manual_seed(1), noise_scale=0.5)
    assert not torch.allclose(with_noise, with_more_noise)

    deterministic = select_action(policy, obs, torch.Generator().manual_seed(1), noise_scale=0.0)
    assert torch.allclose(deterministic, policy(obs))


def test_select_action_rejects_negative_noise_scale():
    """Verify that negative noise scale values raise a ValueError in select_action."""
    policy = build_policy_network(4, 2, 16, 2)
    obs = torch.randn(2, 4)
    rng = torch.Generator()
    with pytest.raises(ValueError, match="non-negative"):
        select_action(policy, obs, rng, noise_scale=-0.1)


def test_build_rl_policy_runner_additional_branches(tmp_path: Path) -> None:
    """Verify that build_rl_policy_runner handles stochastic execution, action bounds, and shape validations correctly."""
    policy = build_policy_network(4, 2, 16, 2)
    checkpoint_path = tmp_path / "policy.pt"
    save_rl_policy_checkpoint(policy, checkpoint_path, 1, 4, 2, 16, 2)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    with pytest.raises(ValueError, match="action_low must have shape"):
        build_rl_policy_runner(checkpoint, 4, 2, "cpu", action_low=[0.0])

    with pytest.raises(ValueError, match="action_high must have shape"):
        build_rl_policy_runner(checkpoint, 4, 2, "cpu", action_high=[1.0, 1.0, 1.0])

    stochastic_runner = build_rl_policy_runner(
        checkpoint,
        4,
        2,
        "cpu",
        stochastic=True,
        exploration_noise=0.1,
        seed=42,
        action_low=[-0.5, -0.5],
        action_high=[0.5, 0.5],
    )
    obs = np.random.randn(4).astype(np.float32)
    action = stochastic_runner(obs)
    assert action.shape == (2,)
    assert np.all(action >= -0.5)
    assert np.all(action <= 0.5)

    with pytest.raises(TypeError, match="observation must be a numpy array"):
        stochastic_runner("invalid_obs")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="observation must have shape"):
        stochastic_runner(np.zeros(3))


def test_load_rl_policy_checkpoint_validations(tmp_path: Path) -> None:
    """Verify that load_rl_policy_checkpoint raises appropriate errors for invalid checkpoint paths or missing files."""
    with pytest.raises(TypeError, match="checkpoint_path must be"):
        load_rl_policy_checkpoint("invalid_path", "cpu")  # type: ignore[arg-type]

    non_existent = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        load_rl_policy_checkpoint(non_existent, "cpu")


def test_run_policy_step_execution() -> None:
    """Verify that run_policy_step correctly invokes policy runners on numpy observation vectors."""

    def dummy_runner(obs: np.ndarray) -> np.ndarray:
        return np.array([0.1, -0.1], dtype=np.float32)

    obs = np.zeros(4, dtype=np.float32)
    action = run_policy_step(dummy_runner, obs)
    assert action.shape == (2,)
    assert np.allclose(action, [0.1, -0.1])


def test_rl_policy_additional_validations(tmp_path: Path) -> None:
    """Verify metadata reading, value network construction, and parameter boundary validations in RL policy components."""
    policy = build_policy_network(4, 2, 16, 2)
    checkpoint_path = tmp_path / "policy.pt"

    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        save_rl_policy_checkpoint(policy, checkpoint_path, 1, 4, 2, 0, 2)

    with pytest.raises(ValueError, match="num_layers must be positive"):
        save_rl_policy_checkpoint(policy, checkpoint_path, 1, 4, 2, 16, 0)

    tensor_ckpt = {
        "observation_dim": torch.tensor(4),
        "action_dim": torch.tensor(2),
        "hidden_dim": torch.tensor(16),
        "num_layers": torch.tensor(2),
    }
    assert read_rl_policy_metadata(tensor_ckpt) == (4, 2, 16, 2)

    corrupted_ckpt = {
        "observation_dim": "invalid",
        "action_dim": 2,
        "hidden_dim": 16,
        "num_layers": 2,
    }
    assert read_rl_policy_metadata(corrupted_ckpt) is None

    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        build_policy_network(4, 2, 0, 2)

    with pytest.raises(ValueError, match="num_layers must be positive"):
        build_policy_network(4, 2, 16, 0)

    with pytest.raises(ValueError, match="observation_dim must be positive"):
        build_value_network(0, 16, 2)

    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        build_value_network(4, 0, 2)

    with pytest.raises(ValueError, match="num_layers must be positive"):
        build_value_network(4, 16, 0)

    obs = torch.randn(2, 4)
    with pytest.raises(TypeError, match=r"rng must be a torch\.Generator"):
        select_action(policy, obs, "not_a_generator")  # type: ignore[arg-type]


def test_run_rl_training_pipeline_integration(tmp_path: Path, panda_robot_xml: Path) -> None:
    """Verify end-to-end execution of the RL training pipeline and subsequent checkpoint metadata persistence."""
    ycb_dir = tmp_path / "ycb"
    ycb_dir.mkdir()
    policy_ckpt = tmp_path / "trained_policy.pt"

    from grasping_ai.simulation.mujoco_env import MuJoCoGraspingEnv

    env = MuJoCoGraspingEnv(panda_robot_xml)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env.close()

    run_rl_training_pipeline(
        robot_xml_path=panda_robot_xml,
        ycb_root=ycb_dir,
        object_ids=[],
        policy_checkpoint_path=policy_ckpt,
        observation_dim=obs_dim,
        action_dim=act_dim,
        hidden_dim=16,
        learning_rate=1e-3,
        num_updates=1,
        gamma=0.99,
        device="cpu",
        seed=42,
    )

    assert policy_ckpt.is_file()
    ckpt = torch.load(policy_ckpt, map_location="cpu")
    assert ckpt["observation_dim"] == obs_dim
    assert ckpt["action_dim"] == act_dim


def test_run_rl_training_pipeline_validation_errors(tmp_path: Path, panda_robot_xml: Path) -> None:
    """Verify validation checks on inputs (invalid paths, out of range values) for the RL training pipeline parameters."""
    valid_robot = panda_robot_xml
    valid_ycb = tmp_path / "ycb"
    valid_ycb.mkdir()

    with pytest.raises(TypeError, match="robot_xml_path"):
        run_rl_training_pipeline("invalid", valid_ycb, [], tmp_path / "c.pt", 18, 8, 16, 1e-3, 1, 0.99, "cpu")  # type: ignore[arg-type]

    with pytest.raises(FileNotFoundError, match="Robot XML file not found"):
        run_rl_training_pipeline(
            tmp_path / "missing.xml",
            valid_ycb,
            [],
            tmp_path / "c.pt",
            18,
            8,
            16,
            1e-3,
            1,
            0.99,
            "cpu",
        )

    with pytest.raises(TypeError, match="ycb_root"):
        run_rl_training_pipeline(
            valid_robot,
            "invalid",
            ["obj1"],
            tmp_path / "c.pt",
            18,
            8,
            16,
            1e-3,
            1,
            0.99,
            "cpu",
        )  # type: ignore[arg-type]

    with pytest.raises(FileNotFoundError, match="YCB root directory not found"):
        run_rl_training_pipeline(
            valid_robot,
            tmp_path / "missing_ycb",
            ["obj1"],
            tmp_path / "c.pt",
            18,
            8,
            16,
            1e-3,
            1,
            0.99,
            "cpu",
        )

    with pytest.raises(ValueError, match="observation_dim"):
        run_rl_training_pipeline(
            valid_robot,
            valid_ycb,
            [],
            tmp_path / "c.pt",
            0,
            8,
            16,
            1e-3,
            1,
            0.99,
            "cpu",
        )

    with pytest.raises(ValueError, match="action_dim"):
        run_rl_training_pipeline(
            valid_robot,
            valid_ycb,
            [],
            tmp_path / "c.pt",
            18,
            0,
            16,
            1e-3,
            1,
            0.99,
            "cpu",
        )

    with pytest.raises(ValueError, match="hidden_dim"):
        run_rl_training_pipeline(
            valid_robot,
            valid_ycb,
            [],
            tmp_path / "c.pt",
            18,
            8,
            0,
            1e-3,
            1,
            0.99,
            "cpu",
        )

    with pytest.raises(ValueError, match="learning_rate"):
        run_rl_training_pipeline(
            valid_robot,
            valid_ycb,
            [],
            tmp_path / "c.pt",
            18,
            8,
            16,
            0.0,
            1,
            0.99,
            "cpu",
        )

    with pytest.raises(ValueError, match="num_updates"):
        run_rl_training_pipeline(
            valid_robot,
            valid_ycb,
            [],
            tmp_path / "c.pt",
            18,
            8,
            16,
            1e-3,
            0,
            0.99,
            "cpu",
        )

    with pytest.raises(ValueError, match="gamma"):
        run_rl_training_pipeline(
            valid_robot,
            valid_ycb,
            [],
            tmp_path / "c.pt",
            18,
            8,
            16,
            1e-3,
            1,
            1.5,
            "cpu",
        )

    with pytest.raises(ValueError, match="object_ids must contain at most one object"):
        run_rl_training_pipeline(
            valid_robot,
            valid_ycb,
            ["obj1", "obj2"],
            tmp_path / "c.pt",
            18,
            8,
            16,
            1e-3,
            1,
            0.99,
            "cpu",
        )
