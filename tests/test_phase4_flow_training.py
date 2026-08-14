import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _make_dataset(tmp_path: Path, *, n_grasps: int, seed: int) -> Path:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    rng = np.random.default_rng(seed)
    pc = rng.standard_normal((64, 3)).astype(np.float32)
    grasps = np.tile(np.eye(4, dtype=np.float32)[None], (n_grasps, 1, 1))
    grasps[:, :3, 3] = rng.standard_normal((n_grasps, 3)).astype(np.float32) * 0.1
    np.save(
        dataset_root / "flow_obj.npy",
        {
            "point_cloud": pc,
            "grasp_poses": grasps,
            "scores": None,
            "object_id": "flow_obj",
        },
        allow_pickle=True,
    )
    return dataset_root


def test_flow_model_forward_delegates_to_flow_field() -> None:
    """Verify ``FlowGeneratorModel.forward`` returns flow-field predictions.

    Returns:
        None. Asserts output shape matches the input grasp batch.
    """
    from grasping_ai.models.flow import FlowGeneratorModel

    model = FlowGeneratorModel(feature_dim=8, hidden_dim=8, num_layers=1)
    x = torch.zeros(2, 9)
    cond = torch.zeros(2, 8)
    out = model.forward(x, cond)
    assert out.shape == (2, 9)


def test_flow_checkpoint_persists_encoder_and_flow_field(tmp_path):
    """The flow checkpoint must contain both encoder and flow_field state.

    Regression for the train/inference model contract: the encoder used at
    training time must be saved as part of the flow checkpoint so inference
    can reproduce the same conditioning signal.
    """
    from grasping_ai.pipelines.train_flow import run_flow_training_pipeline

    dataset_root = _make_dataset(tmp_path, n_grasps=4, seed=0)
    checkpoint = tmp_path / "flow_model.pt"
    run_flow_training_pipeline(
        dataset_root=dataset_root,
        checkpoint_path=checkpoint,
        feature_dim=16,
        hidden_dim=16,
        num_layers=2,
        learning_rate=0.001,
        num_epochs=1,
        batch_size=2,
        device="cpu",
        seed=0,
    )
    state_dict = torch.load(checkpoint, map_location="cpu")
    keys = list(state_dict["model_state_dict"].keys())
    assert any(k.startswith("encoder.") for k in keys), (
        f"flow checkpoint missing encoder keys: {keys}"
    )
    assert any(k.startswith("flow_field.") for k in keys), (
        f"flow checkpoint missing flow_field keys: {keys}"
    )


def test_load_flow_model_checkpoint_reproduces_trained_state(tmp_path):
    """``load_flow_model_checkpoint`` restores the encoder used at training."""
    from grasping_ai.pipelines.train_flow import (
        load_flow_model_checkpoint,
        run_flow_training_pipeline,
    )

    dataset_root = _make_dataset(tmp_path, n_grasps=4, seed=1)
    checkpoint = tmp_path / "flow_repro.pt"
    run_flow_training_pipeline(
        dataset_root=dataset_root,
        checkpoint_path=checkpoint,
        feature_dim=16,
        hidden_dim=16,
        num_layers=2,
        learning_rate=0.001,
        num_epochs=1,
        batch_size=2,
        device="cpu",
        seed=1,
    )

    model = load_flow_model_checkpoint(
        checkpoint, feature_dim=16, hidden_dim=16, num_layers=2, device="cpu"
    )
    expected = torch.load(checkpoint, map_location="cpu")["model_state_dict"]
    for key in expected:
        assert torch.allclose(
            model.state_dict()[key], expected[key]
        ), f"mismatch at {key}"


def test_flow_training_optimizes_encoder_and_flow_field(tmp_path):
    """Both the encoder and flow field parameters change after a training step."""
    from grasping_ai.pipelines.train_flow import build_flow_training_components

    components = build_flow_training_components(
        feature_dim=8, hidden_dim=8, num_layers=2, learning_rate=0.01, device="cpu"
    )
    model = components["model"]
    optimizer = components["optimizer"]

    initial_encoder_norm = sum(
        p.norm().item() for p in model.encoder.parameters()
    )
    initial_flow_norm = sum(
        p.norm().item() for p in model.flow_field.parameters()
    )

    pcs = torch.randn(4, 16, 3)
    targets = torch.randn(4, 9)
    cond = model.condition(pcs)
    pred = model.flow_field(targets, cond)
    loss = pred.pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    updated_encoder_norm = sum(
        p.norm().item() for p in model.encoder.parameters()
    )
    updated_flow_norm = sum(
        p.norm().item() for p in model.flow_field.parameters()
    )

    assert updated_encoder_norm != pytest.approx(initial_encoder_norm)
    assert updated_flow_norm != pytest.approx(initial_flow_norm)


def test_run_flow_training_pipeline_produces_checkpoint(tmp_path):
    """End-to-end flow training writes a checkpoint."""
    from grasping_ai.pipelines.train_flow import run_flow_training_pipeline

    dataset_root = _make_dataset(tmp_path, n_grasps=4, seed=0)
    checkpoint = tmp_path / "flow_model.pt"
    run_flow_training_pipeline(
        dataset_root=dataset_root,
        checkpoint_path=checkpoint,
        feature_dim=16,
        hidden_dim=16,
        num_layers=2,
        learning_rate=0.001,
        num_epochs=2,
        batch_size=2,
        device="cpu",
        seed=0,
    )
    assert checkpoint.is_file()


def test_run_flow_training_pipeline_rejects_missing_dataset_root(tmp_path):
    """Missing dataset root raises FileNotFoundError."""
    from grasping_ai.pipelines.train_flow import run_flow_training_pipeline

    with pytest.raises(FileNotFoundError):
        run_flow_training_pipeline(
            dataset_root=tmp_path / "missing",
            checkpoint_path=tmp_path / "flow.pt",
            feature_dim=8,
            hidden_dim=8,
            num_layers=2,
            learning_rate=0.001,
            num_epochs=1,
            batch_size=1,
            device="cpu",
        )


def test_flow_training_cli(tmp_path):
    """``scripts/train_flow.py`` runs end-to-end via subprocess."""
    dataset_root = _make_dataset(tmp_path, n_grasps=2, seed=1)
    checkpoint = tmp_path / "flow_cli.pt"
    cmd = [
        sys.executable,
        "scripts/train_flow.py",
        "--dataset-root", str(dataset_root),
        "--checkpoint", str(checkpoint),
        "--feature-dim", "8",
        "--hidden-dim", "8",
        "--num-layers", "2",
        "--learning-rate", "0.001",
        "--num-epochs", "1",
        "--batch-size", "1",
        "--device", "cpu",
        "--seed", "0",
    ]
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}

    subprocess.run(
        cmd, env=env, capture_output=True, text=True, check=True
    )
    assert checkpoint.is_file()


def test_run_flow_training_pipeline_validations_and_resume(tmp_path: Path) -> None:
    from grasping_ai.pipelines.train_flow import run_flow_training_pipeline

    with pytest.raises(TypeError, match="dataset_root"):
        run_flow_training_pipeline(
            dataset_root="not_a_path",  # type: ignore[arg-type]
            checkpoint_path=tmp_path / "flow.pt",
            feature_dim=8,
            hidden_dim=8,
            num_layers=2,
            learning_rate=0.001,
            num_epochs=1,
            batch_size=1,
            device="cpu",
        )

    dataset_root = _make_dataset(tmp_path, n_grasps=2, seed=42)
    checkpoint_1 = tmp_path / "flow_ckpt1.pt"
    run_flow_training_pipeline(
        dataset_root=dataset_root,
        checkpoint_path=checkpoint_1,
        feature_dim=8,
        hidden_dim=8,
        num_layers=2,
        learning_rate=0.001,
        num_epochs=1,
        batch_size=1,
        device="cpu",
        seed=42,
    )

    checkpoint_2 = tmp_path / "flow_ckpt2.pt"
    run_flow_training_pipeline(
        dataset_root=dataset_root,
        checkpoint_path=checkpoint_2,
        feature_dim=8,
        hidden_dim=8,
        num_layers=2,
        learning_rate=0.001,
        num_epochs=2,
        batch_size=1,
        device="cpu",
        seed=42,
        pretrained_encoder_path=checkpoint_1,
        resume_checkpoint_path=checkpoint_1,
    )
    assert checkpoint_2.is_file()


def test_flow_network_builder_and_sampler_additional_coverage() -> None:
    from grasping_ai.models.flow import (
        FlowFieldNet,
        load_flow_model_from_state,
    )

    net = FlowFieldNet(8, 16, 2)
    assert isinstance(net, FlowFieldNet)

    with pytest.raises(TypeError, match=r"checkpoint\['model_state_dict'\] must be a dictionary"):
        load_flow_model_from_state({"model_state_dict": "not_a_dict"}, 8, 16, 2, "cpu")
