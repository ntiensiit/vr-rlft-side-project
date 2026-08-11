import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import grasping_ai
from grasping_ai.inference.grasp_generator import (
    build_diffusion_grasp_generator,
    generate_candidate_grasps,
    load_grasp_model_checkpoint,
)
from grasping_ai.models.diffusion import (
    GraspGeneratorModel,
    build_diffusion_sampler,
    build_score_network,
    sample_grasps_with_diffusion,
)
from grasping_ai.models.equivariant_encoder import (
    build_equivariant_encoder,
)
from grasping_ai.pipelines.generate_grasps import build_generation_pipeline, write_generated_grasps
from grasping_ai.pipelines.train import load_pretrained_encoder, run_training_pipeline
from grasping_ai.training.losses import build_diffusion_score_loss, build_grasp_pose_regression_loss
from grasping_ai.training.trainer import (
    build_adam_optimizer,
    load_training_checkpoint,
    save_training_checkpoint,
)


def test_phase1_package_import_remains_stable():
    """Verify that grasping_ai is importable."""
    assert grasping_ai.__name__ == "grasping_ai"


def test_model_config_file_exists():
    """Verify that configs/model.yaml exists."""
    config_path = os.path.join("configs", "model.yaml")
    assert os.path.isfile(config_path)


def test_generative_model_forward_shape():
    """Verify score network forward shape."""
    feature_dim = 16
    hidden_dim = 32
    num_layers = 2
    score_model = build_score_network(feature_dim, hidden_dim, num_layers)

    batch_size_val = 4
    x = torch.randn(batch_size_val, 9)
    t = torch.randint(0, 100, (batch_size_val,))
    cond = torch.randn(batch_size_val, feature_dim)

    out = score_model(x, t, cond)
    assert out.shape == (batch_size_val, 9)


def test_generative_model_rejects_invalid_point_cloud_shape():
    """Verify sample_grasps_with_diffusion shape checks."""
    sampler = build_diffusion_sampler(10)
    score_model = build_score_network(16, 32, 2)
    conditioning = torch.randn(4, 5, 16)  # Invalid ndim == 3 (expected 2)
    rng = torch.Generator()

    with pytest.raises(ValueError, match="conditioning must have shape"):
        sample_grasps_with_diffusion(sampler, score_model, conditioning, 9, 5, rng)


def test_generative_model_rejects_non_finite_input():
    """Verify sample_grasps_with_diffusion generator input checks."""
    sampler = build_diffusion_sampler(10)
    score_model = build_score_network(16, 32, 2)
    conditioning = torch.randn(4, 16)

    with pytest.raises(TypeError, match=r"rng must be a torch\.Generator"):
        sample_grasps_with_diffusion(sampler, score_model, conditioning, 9, 5, None)  # type: ignore[arg-type]


def test_supervised_training_loss_is_finite():
    """Verify diffusion score loss is finite."""
    score_model = build_score_network(16, 32, 2)
    loss_fn = build_diffusion_score_loss(score_model)

    pred = torch.randn(4, 9)
    target = torch.randn(4, 9)
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_training_creates_checkpoint():
    """Verify that run_training_pipeline successfully runs and creates a checkpoint."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        dataset_root = temp_path / "dataset"
        dataset_root.mkdir()

        # Create synthetic record matching Phase 3
        record_data = {
            "point_cloud": np.random.randn(50, 3).astype(np.float32),
            "grasp_poses": np.array([np.eye(4) for _ in range(3)], dtype=np.float32),
            "object_id": "ycb_master_chef_can",
        }
        np.save(dataset_root / "sample_0.npy", record_data, allow_pickle=True)

        # Write Phase 3 dataset index.json
        index = {
            "records": [
                {
                    "file_path": "sample_0.npy",
                    "object_id": "ycb_master_chef_can",
                }
            ]
        }
        with open(dataset_root / "index.json", "w") as f:
            json.dump(index, f)

        checkpoint_path = temp_path / "model.pt"

        # Run tiny training pipeline
        run_training_pipeline(
            dataset_root=dataset_root,
            checkpoint_path=checkpoint_path,
            feature_dim=8,
            hidden_dim=16,
            num_layers=2,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=2,
            device="cpu",
        )

        assert checkpoint_path.exists()

        # Load and verify checkpoint contents
        checkpoint = load_grasp_model_checkpoint(checkpoint_path, "cpu")
        assert "model_state_dict" in checkpoint
        assert int(checkpoint["feature_dim"]) == 8
        assert int(checkpoint["hidden_dim"]) == 16
        assert int(checkpoint["num_layers"]) == 2


def test_training_rejects_missing_dataset():
    """Verify that run_training_pipeline checks dataset_root existence."""
    with pytest.raises(FileNotFoundError):
        run_training_pipeline(
            dataset_root=Path("non_existent_dir_12345"),
            checkpoint_path=Path("model.pt"),
            feature_dim=8,
            hidden_dim=16,
            num_layers=2,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=2,
            device="cpu",
        )


def test_checkpoint_roundtrip():
    """Verify saving and loading checkpoints."""
    model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
    optimizer = build_adam_optimizer(model.parameters(), 0.01)

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "ckpt.pt"
        save_training_checkpoint(model, optimizer, 5, checkpoint_path)
        assert checkpoint_path.exists()

        new_model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
        new_optimizer = build_adam_optimizer(new_model.parameters(), 0.01)

        epoch = load_training_checkpoint(checkpoint_path, new_model, new_optimizer, "cpu")
        assert epoch == 5

        # Check weights are loaded identically
        for p1, p2 in zip(model.parameters(), new_model.parameters(), strict=True):
            assert torch.allclose(p1, p2)


def test_generate_grasps_output_shape_single_observation():
    """Verify that inference yields correct output shape."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        checkpoint_path = temp_path / "model.pt"

        # Create model and save checkpoint
        model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
        optimizer = build_adam_optimizer(model.parameters(), 0.01)
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        checkpoint = load_grasp_model_checkpoint(checkpoint_path, "cpu")
        generator = build_diffusion_grasp_generator(checkpoint, feature_dim=8, num_diffusion_steps=5, device="cpu")

        pc = np.random.randn(100, 3).astype(np.float32)
        grasps = generate_candidate_grasps(generator, pc, num_grasps=5)

        assert isinstance(grasps, np.ndarray)
        assert grasps.shape == (5, 4, 4)


def test_generate_grasps_rotations_are_valid():
    """Verify rotation matrices generated are valid SO(3) rotations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        checkpoint_path = temp_path / "model.pt"

        model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
        optimizer = build_adam_optimizer(model.parameters(), 0.01)
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        checkpoint = load_grasp_model_checkpoint(checkpoint_path, "cpu")
        generator = build_diffusion_grasp_generator(checkpoint, feature_dim=8, num_diffusion_steps=5, device="cpu")

        pc = np.random.randn(100, 3).astype(np.float32)
        grasps = generate_candidate_grasps(generator, pc, num_grasps=10)

        for t_matrix in grasps:
            r_matrix = t_matrix[:3, :3]
            # Check det(r_matrix) close to 1
            det = np.linalg.det(r_matrix)
            assert np.allclose(det, 1.0, atol=1e-4)

            # Check r_matrix.T * r_matrix close to I
            rtr = r_matrix.T @ r_matrix
            assert np.allclose(rtr, np.eye(3), atol=1e-4)


def test_generate_grasps_rejects_invalid_checkpoint():
    """Verify load_grasp_model_checkpoint error validation."""
    with pytest.raises(FileNotFoundError):
        load_grasp_model_checkpoint(Path("non_existent_ckpt.pt"), "cpu")


def test_generate_grasps_rejects_invalid_observation_shape():
    """Verify generator shape validation checks."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        checkpoint_path = temp_path / "model.pt"

        model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
        optimizer = build_adam_optimizer(model.parameters(), 0.01)
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        checkpoint = load_grasp_model_checkpoint(checkpoint_path, "cpu")
        generator = build_diffusion_grasp_generator(checkpoint, feature_dim=8, num_diffusion_steps=5, device="cpu")

        # Invalid shape: 3D array (expected 2D)
        pc_invalid = np.random.randn(2, 50, 3).astype(np.float32)
        with pytest.raises(ValueError, match="point_cloud must have shape"):
            generate_candidate_grasps(generator, pc_invalid, num_grasps=5)


def test_model_inference_is_repeatable_without_global_state():
    """Verify that multiple inference runs are identical under same seeded generator."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        checkpoint_path = temp_path / "model.pt"

        model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
        optimizer = build_adam_optimizer(model.parameters(), 0.01)
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        checkpoint = load_grasp_model_checkpoint(checkpoint_path, "cpu")
        generator = build_diffusion_grasp_generator(checkpoint, feature_dim=8, num_diffusion_steps=5, device="cpu")

        pc = np.random.randn(100, 3).astype(np.float32)

        grasps1 = generate_candidate_grasps(generator, pc, num_grasps=10)
        grasps2 = generate_candidate_grasps(generator, pc, num_grasps=10)

        assert np.allclose(grasps1, grasps2, atol=1e-6)


def test_flow_matching_loss_is_finite():
    """Verify flow matching loss function."""
    from grasping_ai.models.flow import build_flow_field
    from grasping_ai.training.losses import build_flow_matching_loss
    flow_field = build_flow_field(8, 16, 2)
    loss_fn = build_flow_matching_loss(flow_field)

    pred = torch.randn(4, 9)
    target = torch.randn(4, 9)
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss)


def test_regression_loss_types():
    """Verify regression loss MSE, Smooth L1 and error raising."""
    loss_mse = build_grasp_pose_regression_loss("mse")
    loss_l1 = build_grasp_pose_regression_loss("smooth_l1")

    pred = torch.randn(3, 9)
    target = torch.randn(3, 9)
    assert torch.isfinite(loss_mse(pred, target))
    assert torch.isfinite(loss_l1(pred, target))

    with pytest.raises(ValueError, match="Unsupported loss_type"):
        build_grasp_pose_regression_loss("unsupported_type")


def test_flow_field_forward_shape():
    """Verify flow field forward shape."""
    from grasping_ai.models.flow import build_flow_field
    flow = build_flow_field(8, 16, 2)
    x = torch.randn(4, 9)
    cond = torch.randn(4, 8)
    out = flow(x, cond)
    assert out.shape == (4, 9)


def test_flow_integrator_shape():
    """Verify flow integrator execution."""
    from grasping_ai.models.flow import build_flow_field, build_flow_integrator
    flow = build_flow_field(8, 16, 2)
    integrator = build_flow_integrator(5)
    x0 = torch.randn(4, 9)
    cond = torch.randn(4, 8)
    out = integrator(flow, x0, cond)
    assert out.shape == (4, 9)


def test_flow_grasp_generator_inference():
    """Verify that flow grasp generator correctly generates SE(3) candidate shapes."""
    class FlowModelWrapper(torch.nn.Module):
        def __init__(self, f_dim: int, h_dim: int, n_layers: int) -> None:
            super().__init__()
            self.feature_dim = f_dim
            self.hidden_dim = h_dim
            self.num_layers = n_layers
            from grasping_ai.models.flow import build_flow_field
            self.encoder = build_equivariant_encoder(f_dim, n_layers)
            self.flow_field = build_flow_field(f_dim, h_dim, n_layers)

    model = FlowModelWrapper(f_dim=8, h_dim=16, n_layers=2)
    optimizer = build_adam_optimizer(model.parameters(), 0.01)

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        checkpoint_path = temp_path / "flow_model.pt"
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        checkpoint = load_grasp_model_checkpoint(checkpoint_path, "cpu")
        from grasping_ai.inference.grasp_generator import build_flow_grasp_generator
        generator = build_flow_grasp_generator(checkpoint, feature_dim=8, num_flow_steps=5, device="cpu")

        pc = np.random.randn(50, 3).astype(np.float32)
        grasps = generate_candidate_grasps(generator, pc, num_grasps=4)
        assert grasps.shape == (4, 4, 4)


def test_generation_pipeline_and_writing():
    """Verify end-to-end generation pipelines and np.save serialization."""
    from grasping_ai.inference.grasp_generator import build_diffusion_grasp_generator
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        checkpoint_path = temp_path / "model.pt"

        model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
        optimizer = build_adam_optimizer(model.parameters(), 0.01)
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        checkpoint = load_grasp_model_checkpoint(checkpoint_path, "cpu")
        generator = build_diffusion_grasp_generator(checkpoint, feature_dim=8, num_diffusion_steps=2, device="cpu")

        pc = np.random.randn(30, 3).astype(np.float32)
        grasps = build_generation_pipeline(pc, generator, 3)
        assert grasps.shape == (3, 4, 4)

        # Test writing
        out_file = temp_path / "output" / "grasps.npy"
        write_generated_grasps(out_file, {"obj1": grasps})
        assert out_file.exists()

        # Check loading it back
        loaded = np.load(out_file, allow_pickle=True).item()
        assert "obj1" in loaded
        assert np.allclose(loaded["obj1"], grasps)

        # Check write failure TypeError
        with pytest.raises(TypeError):
            write_generated_grasps("not_a_path", {"obj1": grasps})  # type: ignore[arg-type]


def test_load_pretrained_encoder():
    """Verify load_pretrained_encoder behavior."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        checkpoint_path = temp_path / "model.pt"

        model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
        optimizer = build_adam_optimizer(model.parameters(), 0.01)
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        enc_state = load_pretrained_encoder(checkpoint_path, "cpu")
        assert "linear.weight" in enc_state

        with pytest.raises(TypeError):
            load_pretrained_encoder("not_a_path", "cpu")  # type: ignore[arg-type]


def test_acquire_point_cloud_from_observation_errors():
    """Verify acquire_point_cloud_from_observation validation checks."""
    from grasping_ai.sensors.pointcloud_sensor import acquire_point_cloud_from_observation
    with pytest.raises(TypeError):
        acquire_point_cloud_from_observation("not_a_path")  # type: ignore[arg-type]

    with pytest.raises(FileNotFoundError):
        acquire_point_cloud_from_observation(Path("non_existent_obs_file_123.npy"))

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)

        # Test loading non-numpy or invalid file
        invalid_file = temp_path / "invalid.npy"
        with open(invalid_file, "w") as f:
            f.write("corrupted data")
        with pytest.raises(ValueError):
            acquire_point_cloud_from_observation(invalid_file)

        # Test invalid shape
        bad_shape_file = temp_path / "bad_shape.npy"
        np.save(bad_shape_file, np.random.randn(10, 4))
        with pytest.raises(ValueError, match="Invalid observation shape"):
            acquire_point_cloud_from_observation(bad_shape_file)

        # Test non-finite values
        non_finite_file = temp_path / "non_finite.npy"
        np.save(non_finite_file, np.array([[1.0, 2.0, np.nan]]))
        with pytest.raises(ValueError, match="contains non-finite values"):
            acquire_point_cloud_from_observation(non_finite_file)

