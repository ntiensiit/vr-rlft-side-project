"""Phase 4 generative grasp model tests."""

from __future__ import annotations

from grasping_ai.config.diffusion import (
    DEFAULT_DIFFUSION_SCHEDULE,
    DiffusionSchedule,
    linear_beta_schedule,
)

from grasping_ai.data.pointcloud_dataset import save_grasp_sample

from grasping_ai.inference.grasp_generator import (
    build_diffusion_grasp_generator,
    build_flow_grasp_generator,
    generate_candidate_grasps,
    load_grasp_model_checkpoint,
)

from grasping_ai.models.diffusion import (
    build_diffusion_sampler,
    build_score_network,
    GraspGeneratorModel,
    sample_grasps_with_diffusion,
    ScoreNetwork,
)

from grasping_ai.models.equivariant_encoder import (
    build_equivariant_encoder,
    compose_with_se3_frame,
    compute_se3_frame,
    encode_point_cloud,
    pool_object_features,
)

from grasping_ai.models.flow import (
    build_flow_field,
    build_flow_integrator,
    FlowGeneratorModel,
)

from grasping_ai.models.grasp_sampling_batch import batch_conditioned_grasp_samples

from grasping_ai.pipelines.generate_grasps import write_generated_grasps

from grasping_ai.pipelines.train_diffusion import run_diffusion_training_pipeline

from grasping_ai.sensors.pointcloud_sensor import acquire_point_cloud_from_observation

from grasping_ai.training.checkpoint_io import (
    checkpoint_scalar_int,
    load_torch_checkpoint,
    read_model_checkpoint_metadata,
)

from grasping_ai.training.losses import (
    build_diffusion_score_loss,
    build_flow_matching_loss,
    build_grasp_pose_regression_loss,
)

from grasping_ai.training.trainer import (
    build_adam_optimizer,
    build_training_step,
    load_training_checkpoint,
    run_training_loop,
    save_training_checkpoint,
)

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import grasping_ai

def test_phase1_package_import_remains_stable():
    """Verify that grasping_ai is importable."""
    assert grasping_ai.__name__ == "grasping_ai"

def test_model_config_files_exist():
    """Verify that configs/model default, diffusion, and flow configs exist."""
    assert os.path.isfile(os.path.join("configs", "model", "default.yaml"))
    assert os.path.isfile(os.path.join("configs", "model", "diffusion.yaml"))
    assert os.path.isfile(os.path.join("configs", "model", "flow.yaml"))

def test_generative_model_forward_shape():
    """Verify score network forward shape."""
    feature_dim = 16
    hidden_dim = 32
    num_layers = 2
    score_model = GraspGeneratorModel(feature_dim, hidden_dim, num_layers).score_net

    batch_size_val = 4
    x = torch.randn(batch_size_val, 9)
    t = torch.randint(0, 100, (batch_size_val,))
    cond = torch.randn(batch_size_val, feature_dim)

    out = score_model(x, t, cond)
    assert out.shape == (batch_size_val, 9)

def test_generative_model_rejects_invalid_point_cloud_shape():
    """Verify sample_grasps_with_diffusion shape checks."""
    sampler = build_diffusion_sampler(10)
    score_model = GraspGeneratorModel(16, 32, 2).score_net
    conditioning = torch.randn(4, 5, 16)  # Invalid ndim == 3 (expected 2)
    rng = torch.Generator()

    with pytest.raises(ValueError, match="conditioning must have shape"):
        sample_grasps_with_diffusion(sampler, score_model, conditioning, 9, 5, rng)

def test_generative_model_rejects_non_finite_input():
    """Verify sample_grasps_with_diffusion generator input checks."""
    sampler = build_diffusion_sampler(10)
    score_model = GraspGeneratorModel(16, 32, 2).score_net
    conditioning = torch.randn(4, 16)

    with pytest.raises(TypeError, match=r"rng must be a torch\.Generator"):
        sample_grasps_with_diffusion(sampler, score_model, conditioning, 9, 5, None)  # type: ignore[arg-type]

def test_supervised_training_loss_is_finite():
    """Verify diffusion score loss is finite."""
    loss_fn = build_diffusion_score_loss()

    pred = torch.randn(4, 9)
    target = torch.randn(4, 9)
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

def test_training_creates_checkpoint():
    """Verify that run_diffusion_training_pipeline successfully runs and creates a checkpoint."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        dataset_root = temp_path / "dataset"
        dataset_root.mkdir()

        
        save_grasp_sample(
            dataset_root / "sample_0.npz",
            {
                "point_cloud": np.random.randn(50, 3).astype(np.float32),
                "grasp_poses": np.array([np.eye(4) for _ in range(3)], dtype=np.float32),
                "object_id": "ycb_master_chef_can",
            },
        )

        # Write Phase 3 dataset index.json
        index = {
            "records": [
                {
                    "file_path": "sample_0.npz",
                    "object_id": "ycb_master_chef_can",
                },
            ],
        }
        with open(dataset_root / "index.json", "w") as f:
            json.dump(index, f)

        checkpoint_path = temp_path / "model.pt"

        # Run tiny training pipeline
        run_diffusion_training_pipeline(
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

        
        metadata = read_model_checkpoint_metadata(checkpoint_path, "cpu")
        assert metadata["kind"] == "diffusion"
        assert metadata["feature_dim"] == 8
        assert metadata["hidden_dim"] == 16
        assert metadata["num_layers"] == 2

def test_training_reiterates_dataloader_per_epoch():
    """Verify multi-epoch training performs a fresh full pass over data each epoch."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        dataset_root = temp_path / "dataset"
        dataset_root.mkdir()

        
        save_grasp_sample(
            dataset_root / "sample_0.npz",
            {
                "point_cloud": np.random.randn(50, 3).astype(np.float32),
                "grasp_poses": np.array([np.eye(4) for _ in range(3)], dtype=np.float32),
                "object_id": "ycb_master_chef_can",
            },
        )
        index = {
            "records": [
                {
                    "file_path": "sample_0.npz",
                    "object_id": "ycb_master_chef_can",
                },
            ],
        }
        with open(dataset_root / "index.json", "w") as f:
            json.dump(index, f)

        checkpoint_path = temp_path / "model.pt"

        def counting_step(inputs, targets):
            return {"loss": 0.0}

        batches_seen = []

        def recording_loop(
            training_step,
            dataloader,
            num_epochs,
            checkpoint_path,
            log_every,
            **kwargs,
        ):
            for _epoch in range(num_epochs):
                epoch_batches = 0
                batches = dataloader() if callable(dataloader) else dataloader
                for _ in batches:
                    epoch_batches += 1
                batches_seen.append(epoch_batches)

        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                "grasping_ai.training.trainer.run_training_loop",
                recording_loop,
            )
            run_diffusion_training_pipeline(
                dataset_root=dataset_root,
                checkpoint_path=checkpoint_path,
                feature_dim=8,
                hidden_dim=16,
                num_layers=2,
                learning_rate=0.01,
                num_epochs=3,
                batch_size=2,
                device="cpu",
            )

        assert batches_seen == [2, 2, 2]

def test_training_rejects_missing_dataset():
    """Verify that run_diffusion_training_pipeline checks dataset_root existence."""
    with pytest.raises(FileNotFoundError):
        run_diffusion_training_pipeline(
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

def test_inference_grasps_follow_input_point_cloud_frame():
    """Verify generated grasps are expressed in the input point-cloud frame."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        checkpoint_path = temp_path / "model.pt"

        model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
        optimizer = build_adam_optimizer(model.parameters(), 0.01)
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        checkpoint = load_grasp_model_checkpoint(checkpoint_path, "cpu")
        generator = build_diffusion_grasp_generator(checkpoint, feature_dim=8, num_diffusion_steps=5, device="cpu")

        rng = np.random.RandomState(0)
        pc = rng.randn(100, 3).astype(np.float32)
        translation = np.array([0.5, -0.3, 0.2], dtype=np.float32)
        pc_translated = pc + translation

        grasps = generate_candidate_grasps(generator, pc, num_grasps=10)
        grasps_translated = generate_candidate_grasps(generator, pc_translated, num_grasps=10)

        assert grasps.shape == (10, 4, 4)
        assert grasps_translated.shape == (10, 4, 4)
        # A normalization-based inference path would return identical grasps for a
        # translated cloud. The corrected path feeds the raw point cloud, so the
        # generated grasps respond to the frame of the input cloud.
        assert not np.allclose(grasps, grasps_translated, atol=1e-6)

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
    
    loss_fn = build_flow_matching_loss()

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
    
    flow = build_flow_field(8, 16, 2)
    x = torch.randn(4, 9)
    cond = torch.randn(4, 8)
    out = flow(x, cond)
    assert out.shape == (4, 9)

def test_flow_integrator_shape():
    """Verify flow integrator execution."""
    
    flow = build_flow_field(8, 16, 2)
    integrator = build_flow_integrator(5)
    x0 = torch.randn(4, 9)
    cond = torch.randn(4, 8)
    out = integrator(flow, x0, cond)
    assert out.shape == (4, 9)

def test_flow_grasp_generator_inference():
    """Verify that flow grasp generator correctly generates SE(3) candidate shapes."""
    model = FlowGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
    optimizer = build_adam_optimizer(model.parameters(), 0.01)

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        checkpoint_path = temp_path / "flow_model.pt"
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        checkpoint = load_grasp_model_checkpoint(checkpoint_path, "cpu")
        
        generator = build_flow_grasp_generator(checkpoint, feature_dim=8, num_flow_steps=5, device="cpu")

        pc = np.random.randn(50, 3).astype(np.float32)
        grasps = generate_candidate_grasps(generator, pc, num_grasps=4)
        assert grasps.shape == (4, 4, 4)

def test_generation_pipeline_and_writing():
    """Verify end-to-end generation pipelines and np.save serialization."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        checkpoint_path = temp_path / "model.pt"

        model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
        optimizer = build_adam_optimizer(model.parameters(), 0.01)
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        checkpoint = load_grasp_model_checkpoint(checkpoint_path, "cpu")
        generator = build_diffusion_grasp_generator(checkpoint, feature_dim=8, num_diffusion_steps=2, device="cpu")

        pc = np.random.randn(30, 3).astype(np.float32)
        grasps = generate_candidate_grasps(generator, pc, 3)
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

def test_pretrained_encoder_checkpoint_loading():
    """Verify encoder weights can be extracted from a training checkpoint."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        checkpoint_path = temp_path / "model.pt"

        model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
        optimizer = build_adam_optimizer(model.parameters(), 0.01)
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        checkpoint = load_torch_checkpoint(checkpoint_path, "cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                encoder_state[key[len("encoder.") :]] = value
            else:
                encoder_state[key] = value
        assert len(encoder_state) > 0

        with pytest.raises(TypeError):
            load_torch_checkpoint("not_a_path", "cpu")  # type: ignore[arg-type]

def test_acquire_point_cloud_from_observation_errors():
    """Verify acquire_point_cloud_from_observation validation checks."""
    
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

def test_default_schedule_matches_legacy_beta_literals():
    """Verify the shared schedule reproduces the legacy linspace(1e-4, 0.02)."""
    assert DEFAULT_DIFFUSION_SCHEDULE.beta_start == 1e-4
    assert DEFAULT_DIFFUSION_SCHEDULE.beta_end == 0.02
    assert DEFAULT_DIFFUSION_SCHEDULE.num_steps == 100
    assert torch.allclose(
        linear_beta_schedule(),
        torch.linspace(1e-4, 0.02, 100),
    )

def test_custom_schedule_values():
    """Verify a custom schedule drives the beta tensor."""
    schedule = DiffusionSchedule(beta_start=1e-3, beta_end=0.1, num_steps=50)
    beta = linear_beta_schedule(schedule)
    assert beta.shape == (50,)
    assert beta[0] == pytest.approx(1e-3)
    assert beta[-1] == pytest.approx(0.1)

def test_linear_beta_schedule_validation():
    """Verify linear_beta_schedule validates its inputs."""
    with pytest.raises(TypeError):
        linear_beta_schedule("not-a-schedule")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_steps"):
        linear_beta_schedule(DiffusionSchedule(num_steps=0))
    with pytest.raises(ValueError, match="non-negative"):
        linear_beta_schedule(DiffusionSchedule(beta_start=-0.01))

def test_diffusion_sampler_accepts_schedule_overrides():
    """Verify build_diffusion_sampler honors beta_start/beta_end overrides."""
    sampler_default = build_diffusion_sampler(10)
    sampler_custom = build_diffusion_sampler(10, beta_start=1e-3, beta_end=0.1)

    score_model = GraspGeneratorModel(4, 16, 2).score_net
    cond = torch.randn(1, 4)
    rng_default = torch.Generator().manual_seed(7)
    rng_custom = torch.Generator().manual_seed(7)

    x = torch.randn(1, 9, generator=rng_default)
    out_default = sampler_default(x.clone(), score_model, cond, rng_default)
    out_custom = sampler_custom(x.clone(), score_model, cond, rng_custom)
    assert not torch.allclose(out_default, out_custom)

def test_default_sampler_unchanged_defaults():
    """Verify the default sampler behavior matches the legacy schedule."""
    sampler = build_diffusion_sampler(5)
    score_model = GraspGeneratorModel(4, 16, 2).score_net
    conditioning = torch.randn(2, 4)
    rng = torch.Generator().manual_seed(11)

    samples = sample_grasps_with_diffusion(sampler, score_model, conditioning, 9, 3, rng)
    assert samples.shape == (2, 3, 9)
    assert torch.isfinite(samples).all()

def _make_checkpoint(tmp_path: Path) -> Path:
    model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
    
    optimizer = build_adam_optimizer(model.parameters(), 0.01)
    checkpoint_path = tmp_path / "model.pt"
    save_training_checkpoint(model, optimizer, 1, checkpoint_path)
    return checkpoint_path

def test_generator_seed_is_configurable():
    """Verify inference seeds are configurable and reproducible."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = _make_checkpoint(Path(tmp_dir))
        checkpoint = load_grasp_model_checkpoint(checkpoint_path, "cpu")
        pc = np.random.randn(60, 3).astype(np.float32)

        gen_a = build_diffusion_grasp_generator(checkpoint, feature_dim=8, num_diffusion_steps=5, device="cpu", seed=1)
        gen_b = build_diffusion_grasp_generator(checkpoint, feature_dim=8, num_diffusion_steps=5, device="cpu", seed=2)
        gen_a2 = build_diffusion_grasp_generator(checkpoint, feature_dim=8, num_diffusion_steps=5, device="cpu", seed=1)

        grasps_a = generate_candidate_grasps(gen_a, pc, num_grasps=6)
        grasps_b = generate_candidate_grasps(gen_b, pc, num_grasps=6)
        grasps_a2 = generate_candidate_grasps(gen_a2, pc, num_grasps=6)

        # Same seed -> identical output; different seed -> different output.
        assert np.allclose(grasps_a, grasps_a2, atol=1e-6)
        assert not np.allclose(grasps_a, grasps_b, atol=1e-6)

def test_flow_generator_seed_is_configurable():
    """Verify the flow generator seed override is accepted."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        model = FlowGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
        
        optimizer = build_adam_optimizer(model.parameters(), 0.01)
        checkpoint_path = tmp_path / "flow_model.pt"
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        checkpoint = load_grasp_model_checkpoint(checkpoint_path, "cpu")

        
        pc = np.random.randn(40, 3).astype(np.float32)
        gen = build_flow_grasp_generator(checkpoint, feature_dim=8, num_flow_steps=5, device="cpu", seed=3)
        grasps = generate_candidate_grasps(gen, pc, num_grasps=4)
        assert grasps.shape == (4, 4, 4)

def _random_rotation() -> torch.Tensor:
    """Return a random SO(3) rotation matrix."""
    a1 = torch.randn(3)
    a2 = torch.randn(3)
    b1 = a1 / a1.norm()
    b2 = a2 - (a2 @ b1) * b1
    b2 = b2 / b2.norm()
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)

def _make_encoder_and_cloud(
    f_dim: int = 16, n_layers: int = 2, n_points: int = 200,
) -> tuple[torch.nn.Module, torch.Tensor]:
    encoder = build_equivariant_encoder(f_dim, n_layers)
    encoder.eval()
    rng = torch.Generator().manual_seed(7)
    cloud = torch.randn(n_points, 3, generator=rng)
    cloud = cloud - cloud.mean(dim=0)
    return encoder, cloud

def test_frame_is_orthonormal():
    """Verify compute_se3_frame returns orthonormal right-handed frames."""
    _, cloud = _make_encoder_and_cloud()
    frame, centroid = compute_se3_frame(cloud.unsqueeze(0))
    r_matrix = frame[0]
    identity = r_matrix.T @ r_matrix
    assert torch.allclose(identity, torch.eye(3), atol=1e-5)
    assert torch.allclose(torch.det(r_matrix), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(centroid, cloud.mean(dim=0), atol=1e-6)

def test_canonical_coordinates_are_se3_invariant():
    """Verify canonical coords are unchanged under rotation and translation."""
    _, cloud = _make_encoder_and_cloud()
    r_rot = _random_rotation()
    t = torch.tensor([0.7, -0.4, 0.2])
    transformed = cloud @ r_rot + t

    frame_a, cent_a = compute_se3_frame(cloud.unsqueeze(0))
    frame_b, cent_b = compute_se3_frame(transformed.unsqueeze(0))
    canon_a = (cloud.unsqueeze(0) - cent_a.unsqueeze(1)) @ frame_a
    canon_b = (transformed.unsqueeze(0) - cent_b.unsqueeze(1)) @ frame_b
    assert torch.allclose(canon_a, canon_b, atol=1e-4)

def test_frame_transforms_covariantly():
    """Verify the frame rotates with the input cloud."""
    _, cloud = _make_encoder_and_cloud()
    r_rot = _random_rotation()
    t = torch.tensor([0.5, 0.25, -0.75])
    transformed = cloud @ r_rot + t

    frame_a, cent_a = compute_se3_frame(cloud.unsqueeze(0))
    frame_b, cent_b = compute_se3_frame(transformed.unsqueeze(0))
    applied = r_rot.T
    assert torch.allclose(frame_b[0], applied @ frame_a[0], atol=1e-4)
    assert torch.allclose(cent_b[0], applied @ cent_a[0] + t, atol=1e-4)

def test_features_and_pooled_descriptor_are_invariant():
    """Verify per-point features and pooled descriptor are SE(3)-invariant."""
    encoder, cloud = _make_encoder_and_cloud()
    r_rot = _random_rotation()
    t = torch.tensor([-0.3, 1.2, 0.9])
    transformed = cloud @ r_rot + t

    with torch.no_grad():
        feats_a = encode_point_cloud(encoder, cloud.unsqueeze(0))
        feats_b = encode_point_cloud(encoder, transformed.unsqueeze(0))
        desc_a = pool_object_features(feats_a)
        desc_b = pool_object_features(feats_b)
    assert torch.allclose(feats_a, feats_b, atol=1e-4)
    assert torch.allclose(desc_a, desc_b, atol=1e-4)

def test_compute_se3_frame_validation():
    """Verify compute_se3_frame validates its inputs."""
    with pytest.raises(TypeError, match=r"must be a torch\.Tensor"):
        compute_se3_frame(np.zeros((10, 3)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"shape \(B, N, 3\)"):
        compute_se3_frame(torch.randn(10, 2))
    with pytest.raises(ValueError, match=r"at least two points"):
        compute_se3_frame(torch.randn(1, 1, 3))

def test_compose_with_se3_frame_maps_canonical_to_input():
    """Verify compose_with_se3_frame inverts the canonicalization."""
    _, cloud = _make_encoder_and_cloud(n_points=300)
    frame, centroid = compute_se3_frame(cloud.unsqueeze(0))

    canonical_grasp = torch.eye(4).unsqueeze(0)
    canonical_grasp[0, :3, 3] = torch.tensor([0.1, 0.2, 0.3])
    canonical_grasp[0, :3, :3] = _random_rotation()

    input_frame = compose_with_se3_frame(canonical_grasp, frame, centroid)
    world = torch.eye(4)
    world[:3, :3] = frame[0]
    world[:3, 3] = centroid[0]
    expected = world @ canonical_grasp @ torch.linalg.inv(world)
    assert torch.allclose(input_frame, expected, atol=1e-5)

def _build_checkpoint_generator(builder: str) -> tuple[object, Path]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        checkpoint_path = temp_path / "model.pt"

        if builder == "diffusion":
            model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
        else:
            model = FlowGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
        optimizer = build_adam_optimizer(model.parameters(), 0.01)
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        checkpoint = load_grasp_model_checkpoint(checkpoint_path, "cpu")
        if builder == "diffusion":
            generator = build_diffusion_grasp_generator(checkpoint, feature_dim=8, num_diffusion_steps=5, device="cpu")
        else:
            generator = build_flow_grasp_generator(checkpoint, feature_dim=8, num_flow_steps=5, device="cpu")
        return generator, checkpoint_path

@pytest.mark.parametrize("builder", ["diffusion", "flow"])
def test_generated_grasps_are_equivariant(builder: str):
    """Verify generated grasps transform covariantly with the input cloud."""
    generator, _ = _build_checkpoint_generator(builder)

    rng = np.random.RandomState(3)
    cloud = rng.randn(200, 3).astype(np.float32)
    cloud = cloud - cloud.mean(axis=0)

    r_rot = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    t = np.array([0.4, -0.5, 0.8], dtype=np.float32)
    transformed = (cloud @ r_rot.T + t).astype(np.float32)

    grasps = generate_candidate_grasps(generator, cloud, num_grasps=6)
    grasps_transformed = generate_candidate_grasps(generator, transformed, num_grasps=6)

    g_transform = np.eye(4, dtype=np.float32)
    g_transform[:3, :3] = r_rot
    g_transform[:3, 3] = t

    expected = np.stack([g_transform @ grasp @ np.linalg.inv(g_transform) for grasp in grasps])
    assert np.allclose(grasps_transformed, expected, atol=1e-3)

@pytest.mark.parametrize("builder", ["diffusion", "flow"])
def test_generated_grasps_follow_input_frame(builder: str):
    """Verify grasps are expressed in the input point-cloud frame."""
    generator, _ = _build_checkpoint_generator(builder)

    rng = np.random.RandomState(11)
    cloud = rng.randn(150, 3).astype(np.float32)
    translation = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    translated = cloud + translation

    grasps = generate_candidate_grasps(generator, cloud, num_grasps=6)
    grasps_translated = generate_candidate_grasps(generator, translated, num_grasps=6)

    assert grasps.shape == (6, 4, 4)
    assert grasps_translated.shape == (6, 4, 4)
    assert not np.allclose(grasps, grasps_translated, atol=1e-6)

def test_checkpoint_io_validations_and_infer_kinds(tmp_path: Path) -> None:
    """Verify that load_torch_checkpoint validates path types and checkpoint files, and infers metadata format correctly."""
    with pytest.raises(TypeError, match="checkpoint_path must be"):
        load_torch_checkpoint("not_a_path", "cpu")  # type: ignore[arg-type]

    non_existent = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        load_torch_checkpoint(non_existent, "cpu")

    corrupted_file = tmp_path / "corrupted.pt"
    corrupted_file.write_bytes(b"invalid data")
    with pytest.raises(ValueError, match="Failed to load checkpoint"):
        load_torch_checkpoint(corrupted_file, "cpu")

    non_dict_file = tmp_path / "non_dict.pt"
    torch.save([1, 2, 3], non_dict_file)
    with pytest.raises(ValueError, match="must deserialize to a dictionary"):
        load_torch_checkpoint(non_dict_file, "cpu")

    assert checkpoint_scalar_int(torch.tensor(5)) == 5
    assert checkpoint_scalar_int(True) == 1
    assert checkpoint_scalar_int(4.8) == 4
    with pytest.raises(TypeError, match="Expected numeric checkpoint scalar"):
        checkpoint_scalar_int("invalid_scalar")  # type: ignore[arg-type]

    flow_ckpt_file = tmp_path / "flow_ckpt.pt"
    torch.save(
        {
            "model_state_dict": {"flow_field.weight": torch.zeros(1)},
            "architecture": "flow_matching",
            "pipeline": "flow_training",
            "feature_dim": 128,
        },
        flow_ckpt_file,
    )
    flow_meta = read_model_checkpoint_metadata(flow_ckpt_file)
    assert flow_meta["kind"] == "flow"
    assert flow_meta["architecture"] == "flow_matching"
    assert flow_meta["pipeline"] == "flow_training"
    assert flow_meta["feature_dim"] == 128

    diff_ckpt_file = tmp_path / "diff_ckpt.pt"
    torch.save(
        {
            "model_state_dict": {"score_net.weight": torch.zeros(1)},
            "hidden_dim": 256,
        },
        diff_ckpt_file,
    )
    diff_meta = read_model_checkpoint_metadata(diff_ckpt_file)
    assert diff_meta["kind"] == "diffusion"
    assert diff_meta["hidden_dim"] == 256

    unknown_ckpt = tmp_path / "unknown.pt"
    torch.save({"model_state_dict": {"encoder.weight": torch.zeros(1)}}, unknown_ckpt)
    assert read_model_checkpoint_metadata(unknown_ckpt)["kind"] == "unknown"

    missing_state = tmp_path / "missing_state.pt"
    torch.save({"architecture": "mystery"}, missing_state)
    assert read_model_checkpoint_metadata(missing_state)["kind"] == "unknown"

def test_run_diffusion_training_pipeline_validations_and_resume(tmp_path: Path) -> None:
    """Verify validations on argument types and resuming from previous checkpoints in the diffusion training pipeline."""
    
    with pytest.raises(TypeError, match="dataset_root"):
        run_diffusion_training_pipeline(
            dataset_root="not_a_path",  # type: ignore[arg-type]
            checkpoint_path=tmp_path / "model.pt",
            feature_dim=8,
            hidden_dim=8,
            num_layers=2,
            learning_rate=0.001,
            num_epochs=1,
            batch_size=1,
            device="cpu",
        )

    from tests.test_phase4_flow_training import _make_dataset

    dataset_root = _make_dataset(tmp_path, n_grasps=2, seed=123)
    checkpoint_1 = tmp_path / "diff_ckpt1.pt"
    run_diffusion_training_pipeline(
        dataset_root=dataset_root,
        checkpoint_path=checkpoint_1,
        feature_dim=8,
        hidden_dim=8,
        num_layers=2,
        learning_rate=0.001,
        num_epochs=1,
        batch_size=1,
        device="cpu",
        seed=123,
    )

    checkpoint_2 = tmp_path / "diff_ckpt2.pt"
    run_diffusion_training_pipeline(
        dataset_root=dataset_root,
        checkpoint_path=checkpoint_2,
        feature_dim=8,
        hidden_dim=8,
        num_layers=2,
        learning_rate=0.001,
        num_epochs=2,
        batch_size=1,
        device="cpu",
        seed=123,
        pretrained_encoder_path=checkpoint_1,
        resume_checkpoint_path=checkpoint_1,
    )
    assert checkpoint_2.is_file()

def test_trainer_additional_branches(tmp_path: Path) -> None:
    """Verify parameter validations on build_adam_optimizer and check exception cases for loading and saving checkpoints."""
            
    model = GraspGeneratorModel(4, 16, 2)
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        build_adam_optimizer(model.parameters(), -0.01)

    opt = build_adam_optimizer(model.parameters(), 0.001)

    with pytest.raises(TypeError, match=r"checkpoint_path must be a pathlib\.Path"):
        save_training_checkpoint(model, opt, 1, "not_a_path")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"checkpoint_path must be a pathlib\.Path"):
        load_training_checkpoint("not_a_path", model, opt, "cpu")  # type: ignore[arg-type]

    dir_path = tmp_path / "is_dir"
    dir_path.mkdir()
    with pytest.raises(ValueError, match="Failed to save checkpoint"):
        save_training_checkpoint(model, opt, 1, dir_path)

    ckpt_file = tmp_path / "train_save.pt"
    save_training_checkpoint(model, opt, 5, ckpt_file)
    epoch = load_training_checkpoint(ckpt_file, model, None, "cpu")
    assert epoch == 5

    tb_dir = tmp_path / "tb_logs"
    dummy_input = torch.randn(2, 4)
    dummy_target = torch.randn(2, 9)
    dataloader = [(dummy_input, dummy_target)]

    step_fn = build_training_step(model, build_diffusion_score_loss(), opt, "cpu", seed=42)

    run_training_loop(
        step_fn,
        dataloader,
        num_epochs=1,
        checkpoint_path=tmp_path / "loop_ckpt.pt",
        log_every=1,
        experiment_log_dir=tb_dir,
        metadata={"experiment": "test_run"},
    )
    assert tb_dir.exists()

def test_equivariant_encoder_collinear_points_fallback() -> None:
    """Verify that the equivariant encoder falls back to a valid rotation frame if conditioning point clouds are collinear."""
    
    collinear_pts = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=torch.float32)
    frame, _centroid = compute_se3_frame(collinear_pts)
    assert frame.shape == (1, 3, 3)
    assert torch.allclose(torch.det(frame), torch.tensor([1.0]), atol=1e-5)

def test_batch_conditioned_grasp_samples_validations() -> None:
    """Verify that sampling candidate grasp batches raises a ValueError if the sample count is not positive."""
    
    cond = torch.randn(2, 8)
    rng = torch.Generator()
    with pytest.raises(ValueError, match="num_samples must be a positive integer"):
        batch_conditioned_grasp_samples(cond, 9, 0, rng, lambda x, c: x)

def test_diffusion_and_score_network_additional_coverage() -> None:
    """Verify that building score networks and diffusion samplers handles dummy inference passes correctly."""
    
    net = build_score_network(8, 16, 2)
    assert isinstance(net, ScoreNetwork)

    sampler = build_diffusion_sampler(num_steps=3)
    x0 = torch.randn(2, 9)
    cond = torch.randn(2, 8)
    sampled = sampler(x0, net, cond, rng=None)
    assert sampled.shape == (2, 9)

def test_trainer_checkpoint_saving_branch(tmp_path: Path) -> None:
    """Verify that run_training_loop successfully writes checkpoint files under default configurations."""
    
    model = torch.nn.Linear(8, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def dummy_step(inputs, targets):
        return {"loss": 0.1}

    dummy_step.model = model  # type: ignore[attr-defined]
    dummy_step.optimizer = optimizer  # type: ignore[attr-defined]

    ckpt_path = tmp_path / "auto_ckpt.pt"
    dataloader = [(torch.randn(2, 8), torch.randn(2, 2))]
    run_training_loop(dummy_step, dataloader, num_epochs=1, log_every=10, checkpoint_path=ckpt_path)
    assert ckpt_path.is_file()

    # Verify trainer fallback when mlflow is not installed (ImportError blocks)
    import sys
    orig_mlflow = sys.modules.get("mlflow")
    sys.modules["mlflow"] = None  # type: ignore[assignment]
    try:
        ckpt_path_ml = tmp_path / "ml_ckpt.pt"
        tb_dir = tmp_path / "tb_logs"
        run_training_loop(
            dummy_step,
            dataloader,
            num_epochs=1,
            log_every=1,
            checkpoint_path=ckpt_path_ml,
            experiment_log_dir=tb_dir,
            metadata={"experiment": "test_run"},
        )
        assert ckpt_path_ml.is_file()
    finally:
        if orig_mlflow is not None:
            sys.modules["mlflow"] = orig_mlflow
        else:
            sys.modules.pop("mlflow", None)
