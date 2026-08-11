import copy
import json

import numpy as np
import torch

from grasping_ai.pipelines.evaluate import write_evaluation_report
from grasping_ai.pipelines.train import run_training_pipeline
from grasping_ai.training.trainer import (
    build_training_step,
    run_training_loop,
)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(9, 9)
        self.feature_dim = 128
        self.hidden_dim = 256
        self.num_layers = 4

    def forward(self, x, t, cond):
        return self.fc(x)


def test_build_training_step_seeding():
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
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    step = build_training_step(model, loss_fn, optimizer, "cpu", seed=42)

    dataloader = [(torch.randn(2, 128), torch.randn(2, 9))]
    checkpoint_path = tmp_path / "checkpoint.pt"
    log_dir = tmp_path / "tb_logs"

    metadata = {"lr": 1e-3, "batch_size": 2}
    run_training_loop(
        step, dataloader, num_epochs=1, checkpoint_path=checkpoint_path,
        log_every=1, experiment_log_dir=log_dir, metadata=metadata, seed=42
    )

    assert checkpoint_path.is_file()
    assert log_dir.is_dir()
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    assert len(event_files) > 0

    checkpoint = torch.load(checkpoint_path)
    assert checkpoint.get("seed") == 42


def test_evaluation_tracking(tmp_path):
    report_path = tmp_path / "report.json"
    log_dir = tmp_path / "tb_eval_logs"

    results = {
        "success_rate": 0.8,
        "collision_free_rate": 0.9,
        "force_closure_rate": 0.85,
    }

    write_evaluation_report(report_path, results, experiment_log_dir=log_dir)

    assert report_path.is_file()
    with report_path.open("r") as fp:
        loaded = json.load(fp)
    assert loaded == results

    assert log_dir.is_dir()
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    assert len(event_files) > 0


def test_supervised_reproducibility(tmp_path):
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

    run_training_pipeline(
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

    run_training_pipeline(
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

    run_training_pipeline(
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
