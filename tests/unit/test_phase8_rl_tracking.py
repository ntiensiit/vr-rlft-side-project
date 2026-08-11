import numpy as np
import torch

from grasping_ai.training.rl_trainer import (
    build_rl_training_step,
    compute_discounted_returns,
    run_rl_training_loop,
)


class DummyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(6, 3)

    def forward(self, x):
        return self.fc(x)

def test_rl_training_gamma():
    transitions = [
        (np.zeros(6), np.zeros(3), 1.0, np.zeros(6), False),
        (np.zeros(6), np.zeros(3), 1.0, np.zeros(6), False),
    ]

    ret_g1 = compute_discounted_returns(transitions, 0.9)
    ret_g2 = compute_discounted_returns(transitions, 0.5)
    assert not np.allclose(ret_g1, ret_g2)

def test_rl_training_loop_tracking(tmp_path):
    policy = DummyPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    step = build_rl_training_step(
        policy, optimizer, clip_ratio=0.2, entropy_coefficient=0.0, device="cpu", gamma=0.9
    )

    rollout = [
        (np.zeros(6), np.zeros(3), 1.0, np.zeros(6), False)
    ]

    checkpoint_path = tmp_path / "rl_policy.pt"
    log_dir = tmp_path / "tb_rl_logs"

    metadata = {"gamma": 0.9}

    run_rl_training_loop(
        step, iter([rollout]), num_updates=1, checkpoint_path=checkpoint_path,
        log_every=1, experiment_log_dir=log_dir, metadata=metadata, seed=123
    )

    assert checkpoint_path.is_file()
    assert log_dir.is_dir()
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    assert len(event_files) > 0

    checkpoint = torch.load(checkpoint_path)
    assert checkpoint.get("seed") == 123
