import numpy as np
import torch

from grasping_ai.pipelines.train import run_training_pipeline


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
