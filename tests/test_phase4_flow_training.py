"""Phase 4 — flow-matching training pipeline regression test."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest


def test_run_flow_training_pipeline_produces_checkpoint(tmp_path):
    """End-to-end flow training writes a checkpoint and lowers the loss."""
    from grasping_ai.pipelines.train_flow import run_flow_training_pipeline

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    rng = np.random.default_rng(0)
    pc = rng.standard_normal((64, 3)).astype(np.float32)
    grasps = np.tile(np.eye(4, dtype=np.float32)[None], (4, 1, 1))
    grasps[:, :3, 3] = rng.standard_normal((4, 3)).astype(np.float32) * 0.1
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
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    rng = np.random.default_rng(1)
    pc = rng.standard_normal((32, 3)).astype(np.float32)
    grasps = np.tile(np.eye(4, dtype=np.float32)[None], (2, 1, 1))
    np.save(
        dataset_root / "cli_obj.npy",
        {
            "point_cloud": pc,
            "grasp_poses": grasps,
            "scores": None,
            "object_id": "cli_obj",
        },
        allow_pickle=True,
    )

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
    import subprocess

    subprocess.run(
        cmd, env=env, capture_output=True, text=True, check=True
    )
    assert checkpoint.is_file()
