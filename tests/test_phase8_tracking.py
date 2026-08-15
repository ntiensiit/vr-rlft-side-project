from __future__ import annotations
import copy

import numpy as np
import torch

from grasping_ai.pipelines.evaluate import write_evaluation_report
from grasping_ai.pipelines.train_diffusion import run_diffusion_training_pipeline
from grasping_ai.training.trainer import (
    build_training_step,
    run_training_loop,
)


class DummyModel(torch.nn.Module):
    """A simple, un-parameterized dummy model to simulate neural net structures during testing."""

    def __init__(self):
        """Initialize DummyModel with standard fully connected layers and dimensions."""
        super().__init__()
        self.fc = torch.nn.Linear(9, 9)
        self.feature_dim = 128
        self.hidden_dim = 256
        self.num_layers = 4

    def forward(self, x, t, cond):
        """Perform a dummy forward pass returning the linear mapping of the input."""
        return self.fc(x)


def test_build_training_step_seeding():
    """Verify that the built training step functions produce identical outputs given the same seed."""
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    inputs = torch.randn(2, 128)
    targets = torch.randn(2, 9)

    model2 = copy.deepcopy(model)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

    step1 = build_training_step(model, loss_fn, optimizer, "cpu", seed=42)
    step2 = build_training_step(model2, loss_fn, optimizer2, "cpu", seed=42)

    res1 = step1(inputs, targets)
    res2 = step2(inputs, targets)
    assert np.allclose(res1["loss"], res2["loss"])


def test_training_loop_tracking(tmp_path):
    """Verify that training loop processes logs to Tensorboard and checkpoints parameters under correct keys."""
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    step = build_training_step(model, loss_fn, optimizer, "cpu", seed=42)

    dataloader = [(torch.randn(2, 128), torch.randn(2, 9))]
    checkpoint_path = tmp_path / "checkpoint.pt"
    log_dir = tmp_path / "tb_logs"

    metadata = {"lr": 1e-3, "batch_size": 2}

    from unittest.mock import patch
    with patch("mlflow.active_run", return_value=True), \
         patch("mlflow.log_param") as mock_log_param, \
         patch("mlflow.log_metric") as mock_log_metric, \
         patch("mlflow.log_artifact") as mock_log_art:
        run_training_loop(
            step,
            dataloader,
            num_epochs=1,
            checkpoint_path=checkpoint_path,
            log_every=1,
            experiment_log_dir=log_dir,
            metadata=metadata,
            seed=42,
        )
        assert mock_log_param.call_count == 2
        assert mock_log_metric.call_count == 1
        assert mock_log_art.call_count == 1

    assert checkpoint_path.is_file()
    assert log_dir.is_dir()
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    assert len(event_files) > 0

    checkpoint = torch.load(checkpoint_path)
    assert checkpoint.get("seed") == 42


def test_evaluation_tracking(tmp_path):
    """Verify that evaluation reports log the expected summary records and push events to Tensorboard."""
    from grasping_ai.pipelines.evaluate import read_jsonl_records

    report_path = tmp_path / "report.jsonl"
    log_dir = tmp_path / "tb_eval_logs"

    results = {
        "success_rate": 0.8,
        "collision_free_rate": 0.9,
        "force_closure_rate": 0.85,
    }

    write_evaluation_report(report_path, results, experiment_log_dir=log_dir)

    assert report_path.is_file()
    loaded = next(record for record in read_jsonl_records(report_path) if record.get("record_type") == "summary")
    assert loaded == {"record_type": "summary", **results}

    assert log_dir.is_dir()
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    assert len(event_files) > 0


def test_supervised_reproducibility(tmp_path):
    """Verify that supervised model training is reproducible with matching seeds and stochastic with different seeds."""
    dataset_root = tmp_path / "mock_dataset"
    dataset_root.mkdir()

    record = {
        "point_cloud": np.random.randn(10, 3).astype(np.float32),
        "grasp_poses": np.array([np.eye(4) for _ in range(2)]),
        "scores": None,
        "object_id": "test_object",
    }
    np.save(dataset_root / "test_object.npy", record)

    checkpoint_path1 = tmp_path / "chk1.pt"
    checkpoint_path2 = tmp_path / "chk2.pt"
    checkpoint_path3 = tmp_path / "chk3.pt"

    run_diffusion_training_pipeline(
        dataset_root=dataset_root,
        checkpoint_path=checkpoint_path1,
        feature_dim=8,
        hidden_dim=16,
        num_layers=1,
        learning_rate=1e-3,
        num_epochs=1,
        batch_size=1,
        device="cpu",
        seed=42,
    )

    run_diffusion_training_pipeline(
        dataset_root=dataset_root,
        checkpoint_path=checkpoint_path2,
        feature_dim=8,
        hidden_dim=16,
        num_layers=1,
        learning_rate=1e-3,
        num_epochs=1,
        batch_size=1,
        device="cpu",
        seed=42,
    )

    run_diffusion_training_pipeline(
        dataset_root=dataset_root,
        checkpoint_path=checkpoint_path3,
        feature_dim=8,
        hidden_dim=16,
        num_layers=1,
        learning_rate=1e-3,
        num_epochs=1,
        batch_size=1,
        device="cpu",
        seed=43,
    )

    chk1 = torch.load(checkpoint_path1)
    chk2 = torch.load(checkpoint_path2)
    chk3 = torch.load(checkpoint_path3)

    for k in chk1["model_state_dict"]:
        assert torch.allclose(chk1["model_state_dict"][k], chk2["model_state_dict"][k])

    diff = False
    for k in chk1["model_state_dict"]:
        if not torch.allclose(chk1["model_state_dict"][k], chk3["model_state_dict"][k]):
            diff = True
            break
    assert diff, "Different seeds should produce different model initialization and noise"

    # Test invalid pretrained_encoder_path type
    import pytest
    with pytest.raises(TypeError):
        run_diffusion_training_pipeline(
            dataset_root=dataset_root,
            checkpoint_path=tmp_path / "fail.pt",
            feature_dim=8,
            hidden_dim=16,
            num_layers=1,
            learning_rate=1e-3,
            num_epochs=1,
            batch_size=1,
            device="cpu",
            pretrained_encoder_path="not_a_path_object",
        )

    # Test valid pretrained_encoder_path loading
    run_diffusion_training_pipeline(
        dataset_root=dataset_root,
        checkpoint_path=tmp_path / "pretrained_loaded.pt",
        feature_dim=8,
        hidden_dim=16,
        num_layers=1,
        learning_rate=1e-3,
        num_epochs=1,
        batch_size=1,
        device="cpu",
        pretrained_encoder_path=checkpoint_path1,
    )


def test_setup_logging():
    """Test setup_logging with console and file logs."""
    from datetime import datetime
    from pathlib import Path

    from grasping_ai.utils.logging_utils import setup_logging

    current_date = datetime.now().strftime("%Y-%m-%d")
    expected_file = Path("logs") / f"{current_date}-test_run.log"

    if expected_file.exists():
        expected_file.unlink()

    setup_logging(module_name="test_run", level="DEBUG")

    from loguru import logger
    logger.debug("test log message")

    assert expected_file.is_file()
    content = expected_file.read_text()
    assert "test log message" in content

    # Clean up
    logger.remove()
    expected_file.unlink()


def test_init_mlflow():
    """Test init_mlflow with different configurations."""
    from grasping_ai.utils.logging_utils import init_mlflow

    # 1. backend != mlflow
    config_none = {"tracking": {"backend": "none"}}
    assert init_mlflow(config_none) is False

    # 2. backend == mlflow
    from unittest.mock import patch
    with patch("mlflow.set_tracking_uri") as mock_set_uri, \
         patch("mlflow.set_experiment") as mock_set_exp:

        config_mlflow = {
            "tracking": {
                "backend": "mlflow",
                "mlflow": {
                    "tracking_uri": "http://localhost:5000",
                    "experiment_name": "test_experiment"
                }
            }
        }
        assert init_mlflow(config_mlflow) is True
        mock_set_uri.assert_called_once_with("http://localhost:5000")
        mock_set_exp.assert_called_once_with("test_experiment")


def test_save_training_checkpoint_errors():
    """Verify that trainer functions check input types and raise errors."""
    import pytest

    from grasping_ai.training.trainer import load_training_checkpoint, save_training_checkpoint

    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.raises(TypeError):
        save_training_checkpoint(model, optimizer, 1, "not_a_path_object")

    with pytest.raises(TypeError):
        load_training_checkpoint("not_a_path_object", model, optimizer, "cpu")


def test_training_loop_dataloader_types(tmp_path):
    """Test trainer.py run_training_loop with different dataloader types."""
    from grasping_ai.training.trainer import build_training_step, run_training_loop

    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    step = build_training_step(model, loss_fn, optimizer, "cpu", seed=42)
    checkpoint_path = tmp_path / "chk.pt"

    batch = (torch.randn(2, 128), torch.randn(2, 9))

    # 1. Callable dataloader
    def callable_dl():
        return [batch]

    run_training_loop(step, callable_dl, num_epochs=1, checkpoint_path=checkpoint_path, log_every=1)

    # 2. Iterator dataloader
    iter_dl = iter([batch])
    run_training_loop(step, iter_dl, num_epochs=1, checkpoint_path=checkpoint_path, log_every=1)


