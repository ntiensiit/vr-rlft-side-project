import copy

import numpy as np
import torch

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
