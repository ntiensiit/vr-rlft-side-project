"""Phase 5 RL policy tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import grasping_ai
from grasping_ai.inference.policy_runner import (
    build_rl_policy_runner,
    run_policy_step,
)
from grasping_ai.models.rl_policy import (
    RL_CHECKPOINT_FORMAT_VERSION,
    RL_POLICY_ARCHITECTURE,
    build_policy_network,
    build_sb3_net_arch,
    copy_sb3_policy_weights,
    read_rl_policy_metadata,
    save_rl_policy_checkpoint,
    select_action,
)
from grasping_ai.training.checkpoint_io import load_torch_checkpoint, read_model_checkpoint_metadata

CHECKPOINT_OBSERVATION_DIM = 4
CHECKPOINT_ACTION_DIM = 2
CHECKPOINT_HIDDEN_DIM = 16
CHECKPOINT_NUM_LAYERS = 2
CHECKPOINT_EPOCH = 5
CHECKPOINT_SEED = 42
PANDA_OBSERVATION_DIM = 31
PANDA_ACTION_DIM = 8
ACTION_BOUND = 0.5


def test_phase1_package_import_remains_stable() -> None:
    """Verify that grasping_ai is importable."""
    if not (grasping_ai.__name__ == "grasping_ai"):
        raise AssertionError


def test_stable_baselines_dependency_available() -> None:
    """Verify that Stable-Baselines3 can be imported."""
    import stable_baselines3  # noqa: PLC0415  # deferred: test verifies the optional dependency import

    if not (stable_baselines3.__version__ is not None):
        raise AssertionError


def test_training_config_files_exist() -> None:
    """Verify training config default alias and method variants exist."""
    if not (Path("configs") / "training" / "default.yaml").is_file():
        raise AssertionError
    if not (Path("configs") / "training" / "diffusion.yaml").is_file():
        raise AssertionError
    if not (Path("configs") / "training" / "flow.yaml").is_file():
        raise AssertionError


def test_policy_network_forward_shape() -> None:
    """Verify policy network output has correct shape."""
    policy = build_policy_network(4, 2, 16, 2)
    obs = torch.randn(3, 4)
    out = policy(obs)
    if not (out.shape == (3, 2)):
        raise AssertionError


def test_policy_network_rejects_invalid_dims() -> None:
    """Verify policy network raises on invalid dimensions."""
    with pytest.raises(ValueError, match="observation_dim must be positive"):
        build_policy_network(0, 2, 16, 2)
    with pytest.raises(ValueError, match="action_dim must be positive"):
        build_policy_network(4, -1, 16, 2)


def test_select_action_shape() -> None:
    """Verify select_action produces correct output shape."""
    policy = build_policy_network(4, 2, 16, 2)
    obs = torch.randn(3, 4)
    rng = torch.Generator()
    rng.manual_seed(42)
    action = select_action(policy, obs, rng)
    if not (action.shape == (3, 2)):
        raise AssertionError


def test_select_action_rejects_invalid_observation() -> None:
    """Verify select_action raises on bad observation shape."""
    policy = build_policy_network(4, 2, 16, 2)
    obs_1d = torch.randn(4)
    rng = torch.Generator()
    with pytest.raises(ValueError, match="observation must have shape"):
        select_action(policy, obs_1d, rng)


















def _assert_common_metadata(payload: dict[str, object]) -> None:
    if not (payload["observation_dim"] == CHECKPOINT_OBSERVATION_DIM):
        raise AssertionError
    if not (payload["action_dim"] == CHECKPOINT_ACTION_DIM):
        raise AssertionError
    if not (payload["hidden_dim"] == CHECKPOINT_HIDDEN_DIM):
        raise AssertionError
    if not (payload["num_layers"] == CHECKPOINT_NUM_LAYERS):
        raise AssertionError
    if not (payload["epoch"] == CHECKPOINT_EPOCH):
        raise AssertionError
    if not (payload["seed"] == CHECKPOINT_SEED):
        raise AssertionError


def _assert_checkpoint_payload(checkpoint: dict[str, object]) -> None:
    if not (checkpoint["format_version"] == RL_CHECKPOINT_FORMAT_VERSION):
        raise AssertionError
    if not (checkpoint["architecture"] == RL_POLICY_ARCHITECTURE):
        raise AssertionError
    _assert_common_metadata(checkpoint)
    if "model_state_dict" not in checkpoint:
        raise AssertionError


def test_save_rl_policy_checkpoint_writes_metadata() -> None:
    """Verify that save_rl_policy_checkpoint writes valid metadata and formats state dict properly."""
    policy = build_policy_network(4, 2, 16, 2)
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "policy.pt"
        save_rl_policy_checkpoint(
            policy,
            checkpoint_path,
            epoch=CHECKPOINT_EPOCH,
            observation_dim=CHECKPOINT_OBSERVATION_DIM,
            action_dim=CHECKPOINT_ACTION_DIM,
            hidden_dim=CHECKPOINT_HIDDEN_DIM,
            num_layers=CHECKPOINT_NUM_LAYERS,
            seed=CHECKPOINT_SEED,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        _assert_checkpoint_payload(checkpoint)

        metadata = read_model_checkpoint_metadata(checkpoint_path, "cpu")
        if not (metadata["kind"] == "rl_policy"):
            raise AssertionError
        _assert_common_metadata(metadata)


def test_read_rl_policy_metadata() -> None:
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
        if not (metadata == (3, 1, 8, 2)):
            raise AssertionError


class _FakeSB3Policy(torch.nn.Module):
    """Minimal SB3 MlpPolicy stand-in for weight-copy tests."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        policy_layers: list[torch.nn.Module] = []
        in_dim = observation_dim
        for _ in range(num_layers):
            policy_layers.append(torch.nn.Linear(in_dim, hidden_dim))
            policy_layers.append(torch.nn.Tanh())
            in_dim = hidden_dim
        self.mlp_extractor = torch.nn.Module()
        self.mlp_extractor.policy_net = torch.nn.Sequential(*policy_layers)
        self.action_net = torch.nn.Linear(hidden_dim, action_dim)


def test_copy_sb3_policy_weights_copies_all_hidden_layers() -> None:
    """Verify dynamic SB3-to-legacy weight copy for variable hidden depth."""
    for num_layers in (1, 2, 3):
        sb3_policy = _FakeSB3Policy(4, 2, 16, num_layers)
        legacy_policy = build_policy_network(4, 2, 16, num_layers)
        copy_sb3_policy_weights(sb3_policy, legacy_policy)

        sb3_hidden = [m for m in sb3_policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
        legacy_hidden = [m for m in legacy_policy if isinstance(m, torch.nn.Linear)][:-1]
        for sb3_layer, legacy_layer in zip(sb3_hidden, legacy_hidden, strict=True):
            if not (torch.allclose(sb3_layer.weight, legacy_layer.weight)):
                raise AssertionError
            if not (torch.allclose(sb3_layer.bias, legacy_layer.bias)):
                raise AssertionError

        legacy_output = [m for m in legacy_policy if isinstance(m, torch.nn.Linear)][-1]
        if not (torch.allclose(sb3_policy.action_net.weight, legacy_output.weight)):
            raise AssertionError
        if not (torch.allclose(sb3_policy.action_net.bias, legacy_output.bias)):
            raise AssertionError


def test_build_sb3_net_arch_matches_policy_depth() -> None:
    """Verify SB3 net_arch depth tracks policy_num_layers."""
    if not (build_sb3_net_arch(64, 3) == {"pi": [64, 64, 64], "vf": [64, 64, 64]}):
        raise AssertionError


def test_read_rl_policy_metadata_legacy_checkpoint_returns_none() -> None:
    """Verify that read_rl_policy_metadata returns None for legacy checkpoints lacking explicit metadata headers."""
    if read_rl_policy_metadata({}) is not None:
        raise AssertionError
    if read_rl_policy_metadata({"model_state_dict": {}}) is not None:
        raise AssertionError
    if read_rl_policy_metadata("not-a-dict") is not None:
        raise AssertionError  # type: ignore[arg-type]


def test_runner_loads_standard_checkpoint() -> None:
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
        rng = np.random.default_rng()
        obs = rng.standard_normal(4).astype(np.float32)
        action = runner(obs)
        if not (action.shape == (2,)):
            raise AssertionError
        if not (np.isfinite(action).all()):
            raise AssertionError


def test_runner_rejects_dimension_mismatch() -> None:
    """Verify the runner raises ValueError when observation/action dims mismatch checkpoint metadata."""
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


def test_runner_loads_legacy_checkpoint_without_metadata() -> None:
    """Verify that policy runner can load legacy checkpoints that lack explicit metadata using inferred sizes."""
    # A legacy checkpoint (model_state_dict only) is still loadable by
    # inferring architecture from parameter names.
    policy = build_policy_network(4, 2, 8, 2)
    legacy = {"epoch": 1, "model_state_dict": policy.state_dict()}
    runner = build_rl_policy_runner(legacy, 4, 2, "cpu")
    rng = np.random.default_rng()
    obs = rng.standard_normal(4).astype(np.float32)
    action = runner(obs)
    if not (action.shape == (2,)):
        raise AssertionError


def test_save_rl_policy_checkpoint_validation() -> None:
    """Verify that saving checkpoints validates path types, observation, and action dimensions strictly."""
    policy = build_policy_network(4, 2, 16, 2)
    with pytest.raises(ValueError, match="observation_dim"):
        save_rl_policy_checkpoint(policy, Path("p.pt"), 1, 0, 2, 16, 2)
    with pytest.raises(ValueError, match="action_dim"):
        save_rl_policy_checkpoint(policy, Path("p.pt"), 1, 4, -1, 16, 2)
    with pytest.raises(TypeError, match="policy_checkpoint_path"):
        save_rl_policy_checkpoint(policy, "p.pt", 1, 4, 2, 16, 2)  # type: ignore[arg-type]


def test_select_action_noise_scale_controls_exploration() -> None:
    """Verify that select_action noise scale controls stochastic exploration vs deterministic policy execution."""
    policy = build_policy_network(4, 2, 16, 2)
    obs = torch.randn(3, 4)

    with_noise = select_action(policy, obs, torch.Generator().manual_seed(1), noise_scale=0.1)
    with_more_noise = select_action(policy, obs, torch.Generator().manual_seed(1), noise_scale=0.5)
    if torch.allclose(with_noise, with_more_noise):
        raise AssertionError

    deterministic = select_action(policy, obs, torch.Generator().manual_seed(1), noise_scale=0.0)
    if not (torch.allclose(deterministic, policy(obs))):
        raise AssertionError


def test_select_action_rejects_negative_noise_scale() -> None:
    """Verify that negative noise scale values raise a ValueError in select_action."""
    policy = build_policy_network(4, 2, 16, 2)
    obs = torch.randn(2, 4)
    rng = torch.Generator()
    with pytest.raises(ValueError, match="non-negative"):
        select_action(policy, obs, rng, noise_scale=-0.1)


def test_build_rl_policy_runner_additional_branches(tmp_path: Path) -> None:
    """Verify build_rl_policy_runner stochastic execution, action bounds, and shape validations."""
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
    rng = np.random.default_rng()
    obs = rng.standard_normal(4).astype(np.float32)
    action = stochastic_runner(obs)
    if not (action.shape == (2,)):
        raise AssertionError
    if not (np.all(action >= -ACTION_BOUND)):
        raise AssertionError
    if not (np.all(action <= ACTION_BOUND)):
        raise AssertionError

    with pytest.raises(TypeError, match="observation must be a numpy array"):
        stochastic_runner("invalid_obs")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="observation must have shape"):
        stochastic_runner(np.zeros(3))


def test_checkpoint_loader_validations(tmp_path: Path) -> None:
    """Verify that the shared checkpoint loader rejects invalid paths."""
    with pytest.raises(TypeError, match="checkpoint_path must be"):
        load_torch_checkpoint("invalid_path", "cpu")  # type: ignore[arg-type]

    non_existent = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        load_torch_checkpoint(non_existent, "cpu")


def test_run_policy_step_execution() -> None:
    """Verify that run_policy_step correctly invokes policy runners on numpy observation vectors."""

    def dummy_runner(_obs: np.ndarray) -> np.ndarray:
        return np.array([0.1, -0.1], dtype=np.float32)

    obs = np.zeros(4, dtype=np.float32)
    action = run_policy_step(dummy_runner, obs)
    if not (action.shape == (2,)):
        raise AssertionError
    if not (np.allclose(action, [0.1, -0.1])):
        raise AssertionError


def test_rl_policy_additional_validations(tmp_path: Path) -> None:
    """Verify metadata reading, value network construction, and parameter boundary validations."""
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
    if not (read_rl_policy_metadata(tensor_ckpt) == (4, 2, 16, 2)):
        raise AssertionError

    corrupted_ckpt = {
        "observation_dim": "invalid",
        "action_dim": 2,
        "hidden_dim": 16,
        "num_layers": 2,
    }
    if read_rl_policy_metadata(corrupted_ckpt) is not None:
        raise AssertionError

    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        build_policy_network(4, 2, 0, 2)

    with pytest.raises(ValueError, match="num_layers must be positive"):
        build_policy_network(4, 2, 16, 0)

    obs = torch.randn(2, 4)
    with pytest.raises(TypeError, match=r"rng must be a torch\.Generator"):
        select_action(policy, obs, "not_a_generator")  # type: ignore[arg-type]
