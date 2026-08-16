"""Phase 8 experiment tracking tests."""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from loguru import logger

from grasping_ai.pipelines.evaluate import (
    read_jsonl_records,
    write_evaluation_report,
)
from grasping_ai.training.trainer import (
    build_training_step,
    load_training_checkpoint,
    run_training_loop,
    save_training_checkpoint,
)
from grasping_ai.utils.logging_utils import (
    init_mlflow,
    setup_logging,
)

EXPECTED_TWO_CALLS = 2
EXPECTED_SEED = 42


class DummyModel(torch.nn.Module):
    """A simple, un-parameterized dummy model to simulate neural net structures during testing."""

    def __init__(self) -> None:
        """Initialize DummyModel with standard fully connected layers and dimensions."""
        super().__init__()
        self.fc = torch.nn.Linear(9, 9)
        self.feature_dim = 128
        self.hidden_dim = 256
        self.num_layers = 4

    def forward(self, x: torch.Tensor, _t: object, _cond: object) -> torch.Tensor:
        """Perform a dummy forward pass returning the linear mapping of the input."""
        return self.fc(x)


def test_build_training_step_seeding() -> None:
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
    if not (np.allclose(res1["loss"], res2["loss"])):
        raise AssertionError


def test_training_loop_tracking(tmp_path: Path) -> None:
    """Verify that training loop processes logs to Tensorboard and checkpoints parameters under correct keys."""
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    step = build_training_step(model, loss_fn, optimizer, "cpu", seed=42)

    dataloader = [(torch.randn(2, 128), torch.randn(2, 9))]
    checkpoint_path = tmp_path / "checkpoint.pt"
    log_dir = tmp_path / "tb_logs"

    metadata = {"lr": 1e-3, "batch_size": 2}

    with (
        patch("mlflow.active_run", return_value=True),
        patch("mlflow.log_param") as mock_log_param,
        patch("mlflow.log_metric") as mock_log_metric,
        patch("mlflow.log_artifact") as mock_log_art,
    ):
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
        if not (mock_log_param.call_count == EXPECTED_TWO_CALLS):
            raise AssertionError
        if not (mock_log_metric.call_count == 1):
            raise AssertionError
        if not (mock_log_art.call_count == 1):
            raise AssertionError

    if not (checkpoint_path.is_file()):
        raise AssertionError
    if not (log_dir.is_dir()):
        raise AssertionError
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    if not (len(event_files) > 0):
        raise AssertionError

    checkpoint = torch.load(checkpoint_path)
    if not (checkpoint.get("seed") == EXPECTED_SEED):
        raise AssertionError


def test_evaluation_tracking(tmp_path: Path) -> None:
    """Verify that evaluation reports log the expected summary records and push events to Tensorboard."""
    report_path = tmp_path / "report.jsonl"
    log_dir = tmp_path / "tb_eval_logs"

    results = {
        "success_rate": 0.8,
        "collision_free_rate": 0.9,
        "force_closure_rate": 0.85,
    }

    write_evaluation_report(report_path, results, experiment_log_dir=log_dir)

    if not (report_path.is_file()):
        raise AssertionError
    loaded = next(record for record in read_jsonl_records(report_path) if record.get("record_type") == "summary")
    if not (loaded == {"record_type": "summary", **results}):
        raise AssertionError

    if not (log_dir.is_dir()):
        raise AssertionError
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    if not (len(event_files) > 0):
        raise AssertionError


def test_setup_logging() -> None:
    """Test setup_logging with console and file logs."""
    current_date = datetime.now(tz=UTC).date().isoformat()
    expected_file = Path("logs") / f"{current_date}-test_run.log"

    if expected_file.exists():
        expected_file.unlink()

    setup_logging(module_name="test_run", level="DEBUG")

    logger.debug("test log message")

    if not (expected_file.is_file()):
        raise AssertionError
    content = expected_file.read_text()
    if "test log message" not in content:
        raise AssertionError

    # Clean up
    logger.remove()
    expected_file.unlink()


def test_init_mlflow() -> None:
    """Test init_mlflow with different configurations."""
    # 1. backend != mlflow
    config_none = {"tracking": {"backend": "none"}}
    if init_mlflow(config_none) is not False:
        raise AssertionError

    # 2. backend == mlflow
    with patch("mlflow.set_tracking_uri") as mock_set_uri, patch("mlflow.set_experiment") as mock_set_exp:
        config_mlflow = {
            "tracking": {
                "backend": "mlflow",
                "mlflow": {
                    "tracking_uri": "http://localhost:5000",
                    "experiment_name": "test_experiment",
                },
            },
        }
        if init_mlflow(config_mlflow) is not True:
            raise AssertionError
        mock_set_uri.assert_called_once_with("http://localhost:5000")
        mock_set_exp.assert_called_once_with("test_experiment")


def test_save_training_checkpoint_errors() -> None:
    """Verify that trainer functions check input types and raise errors."""
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.raises(TypeError):
        save_training_checkpoint(model, optimizer, 1, "not_a_path_object")

    with pytest.raises(TypeError):
        load_training_checkpoint("not_a_path_object", model, optimizer, "cpu")


def test_training_loop_dataloader_types(tmp_path: Path) -> None:
    """Test trainer.py run_training_loop with different dataloader types."""
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    step = build_training_step(model, loss_fn, optimizer, "cpu", seed=42)
    checkpoint_path = tmp_path / "chk.pt"

    batch = (torch.randn(2, 128), torch.randn(2, 9))

    # 1. Callable dataloader
    def callable_dl() -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [batch]

    run_training_loop(step, callable_dl, num_epochs=1, checkpoint_path=checkpoint_path, log_every=1)

    # 2. Iterator dataloader
    iter_dl = iter([batch])
    run_training_loop(step, iter_dl, num_epochs=1, checkpoint_path=checkpoint_path, log_every=1)
