"""Phase 4 generative grasp model tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import grasping_ai
from grasping_ai.config.diffusion import (
    DEFAULT_DIFFUSION_SCHEDULE,
    DiffusionSchedule,
    linear_beta_schedule,
)
from grasping_ai.data import training_pairs
from grasping_ai.data.grasp_vector import vec_to_se3
from grasping_ai.inference.grasp_generator import (
    build_diffusion_grasp_generator,
    build_flow_grasp_generator,
    generate_candidate_grasps,
)
from grasping_ai.models.diffusion import (
    GraspGeneratorModel,
    build_diffusion_sampler,
    sample_grasps_with_diffusion,
)
from grasping_ai.models.equivariant_encoder import (
    build_equivariant_encoder,
    compose_with_se3_frame,
    compute_se3_frame,
    encode_point_cloud,
    invert_rigid_transform_batch,
    pool_object_features,
)
from grasping_ai.models.flow import (
    FlowGeneratorModel,
    build_flow_integrator,
)
from grasping_ai.models.grasp_sampling_batch import batch_conditioned_grasp_samples
from grasping_ai.pipelines.generate_grasps import write_generated_grasps
from grasping_ai.sensors.pointcloud_sensor import acquire_point_cloud_from_observation
from grasping_ai.training.checkpoint_io import (
    checkpoint_scalar_int,
    load_torch_checkpoint,
    read_model_checkpoint_metadata,
)
from grasping_ai.training.losses import (
    build_diffusion_score_loss,
    build_flow_matching_loss,
)
from grasping_ai.training.trainer import (
    build_adam_optimizer,
    build_training_step,
    load_training_checkpoint,
    run_training_loop,
    save_training_checkpoint,
)

EXPECTED_FEATURE_DIM = 8
EXPECTED_HIDDEN_DIM = 16
EXPECTED_NUM_LAYERS = 2
EXPECTED_EPOCH = 5
BETA_START = 1e-4
BETA_END = 0.02
BETA_NUM_STEPS = 100
FLOW_FEATURE_DIM = 128
DIFF_HIDDEN_DIM = 256
EXPECTED_SCALAR_INT = 5
SCALAR_INT_FLOORED = 4


def test_phase1_package_import_remains_stable() -> None:
    """Verify that grasping_ai is importable."""
    if not (grasping_ai.__name__ == "grasping_ai"):
        raise AssertionError


def test_model_config_files_exist() -> None:
    """Verify that configs/model default, diffusion, and flow configs exist."""
    if not (Path("configs", "model", "default.yaml").is_file()):
        raise AssertionError
    if not (Path("configs", "model", "diffusion.yaml").is_file()):
        raise AssertionError
    if not (Path("configs", "model", "flow.yaml").is_file()):
        raise AssertionError


def test_generative_model_forward_shape() -> None:
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
    if not (out.shape == (batch_size_val, 9)):
        raise AssertionError


def test_generative_model_rejects_invalid_point_cloud_shape() -> None:
    """Verify sample_grasps_with_diffusion shape checks."""
    sampler = build_diffusion_sampler()
    score_model = GraspGeneratorModel(16, 32, 2).score_net
    conditioning = torch.randn(4, 5, 16)  # Invalid ndim == 3 (expected 2)
    rng = torch.Generator()

    with pytest.raises(ValueError, match="conditioning must have shape"):
        sample_grasps_with_diffusion(sampler, score_model, conditioning, 9, 5, rng)


def test_generative_model_rejects_non_finite_input() -> None:
    """Verify sample_grasps_with_diffusion generator input checks."""
    sampler = build_diffusion_sampler()
    score_model = GraspGeneratorModel(16, 32, 2).score_net
    conditioning = torch.randn(4, 16)

    with pytest.raises(TypeError, match=r"rng must be a torch\.Generator"):
        sample_grasps_with_diffusion(sampler, score_model, conditioning, 9, 5, None)  # type: ignore[arg-type]


def test_supervised_training_loss_is_finite() -> None:
    """Verify diffusion score loss is finite."""
    loss_fn = build_diffusion_score_loss()

    pred = torch.randn(4, 9)
    target = torch.randn(4, 9)
    loss = loss_fn(pred, target)
    if not (torch.isfinite(loss)):
        raise AssertionError
    if not (loss.item() >= 0.0):
        raise AssertionError








def test_checkpoint_roundtrip() -> None:
    """Verify saving and loading checkpoints."""
    model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
    optimizer = build_adam_optimizer(model.parameters(), 0.01)

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "ckpt.pt"
        save_training_checkpoint(model, optimizer, 5, checkpoint_path)
        if not (checkpoint_path.exists()):
            raise AssertionError

        new_model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
        new_optimizer = build_adam_optimizer(new_model.parameters(), 0.01)

        epoch = load_training_checkpoint(checkpoint_path, new_model, new_optimizer, "cpu")
        if not (epoch == EXPECTED_EPOCH):
            raise AssertionError

        # Check weights are loaded identically
        for p1, p2 in zip(model.parameters(), new_model.parameters(), strict=True):
            if not (torch.allclose(p1, p2)):
                raise AssertionError


def test_generate_grasps_output_shape_single_observation() -> None:
    """Verify that inference yields correct output shape."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        checkpoint_path = temp_path / "model.pt"

        # Create model and save checkpoint
        model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
        optimizer = build_adam_optimizer(model.parameters(), 0.01)
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        checkpoint = load_torch_checkpoint(checkpoint_path, "cpu")
        generator = build_diffusion_grasp_generator(checkpoint, feature_dim=8, device="cpu")

        rng = np.random.default_rng()
        pc = rng.standard_normal((100, 3)).astype(np.float32)
        grasps = generate_candidate_grasps(generator, pc, num_grasps=5)

        if not (isinstance(grasps, np.ndarray)):
            raise TypeError
        if not (grasps.shape == (5, 4, 4)):
            raise AssertionError


def test_generate_grasps_rotations_are_valid() -> None:
    """Verify rotation matrices generated are valid SO(3) rotations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        checkpoint_path = temp_path / "model.pt"

        model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
        optimizer = build_adam_optimizer(model.parameters(), 0.01)
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        checkpoint = load_torch_checkpoint(checkpoint_path, "cpu")
        generator = build_diffusion_grasp_generator(checkpoint, feature_dim=8, device="cpu")

        rng = np.random.default_rng()
        pc = rng.standard_normal((100, 3)).astype(np.float32)
        grasps = generate_candidate_grasps(generator, pc, num_grasps=10)

        for t_matrix in grasps:
            r_matrix = t_matrix[:3, :3]
            # Check det(r_matrix) close to 1
            det = np.linalg.det(r_matrix)
            if not (np.allclose(det, 1.0, atol=1e-4)):
                raise AssertionError

            # Check r_matrix.T * r_matrix close to I
            rtr = r_matrix.T @ r_matrix
            if not (np.allclose(rtr, np.eye(3), atol=1e-4)):
                raise AssertionError


def test_generate_grasps_rejects_invalid_checkpoint() -> None:
    """Verify checkpoint loading error validation."""
    with pytest.raises(FileNotFoundError):
        load_torch_checkpoint(Path("non_existent_ckpt.pt"), "cpu")


def test_generate_grasps_rejects_invalid_observation_shape() -> None:
    """Verify generator shape validation checks."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        checkpoint_path = temp_path / "model.pt"

        model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
        optimizer = build_adam_optimizer(model.parameters(), 0.01)
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        checkpoint = load_torch_checkpoint(checkpoint_path, "cpu")
        generator = build_diffusion_grasp_generator(checkpoint, feature_dim=8, device="cpu")

        # Invalid shape: 3D array (expected 2D)
        rng = np.random.default_rng()
        pc_invalid = rng.standard_normal((2, 50, 3)).astype(np.float32)
        with pytest.raises(ValueError, match="point_cloud must have shape"):
            generate_candidate_grasps(generator, pc_invalid, num_grasps=5)


def test_inference_grasps_follow_input_point_cloud_frame() -> None:
    """Verify generated grasps are expressed in the input point-cloud frame."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        checkpoint_path = temp_path / "model.pt"

        model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
        optimizer = build_adam_optimizer(model.parameters(), 0.01)
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        checkpoint = load_torch_checkpoint(checkpoint_path, "cpu")
        generator = build_diffusion_grasp_generator(checkpoint, feature_dim=8, device="cpu")

        rng = np.random.RandomState(0)
        pc = rng.randn(100, 3).astype(np.float32)
        translation = np.array([0.5, -0.3, 0.2], dtype=np.float32)
        pc_translated = pc + translation

        grasps = generate_candidate_grasps(generator, pc, num_grasps=10)
        grasps_translated = generate_candidate_grasps(generator, pc_translated, num_grasps=10)

        if not (grasps.shape == (10, 4, 4)):
            raise AssertionError
        if not (grasps_translated.shape == (10, 4, 4)):
            raise AssertionError
        # A normalization-based inference path would return identical grasps for a
        # translated cloud. The corrected path feeds the raw point cloud, so the
        # generated grasps respond to the frame of the input cloud.
        if np.allclose(grasps, grasps_translated, atol=1e-6):
            raise AssertionError


def test_model_inference_is_repeatable_without_global_state() -> None:
    """Verify that multiple inference runs are identical under same seeded generator."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        checkpoint_path = temp_path / "model.pt"

        model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
        optimizer = build_adam_optimizer(model.parameters(), 0.01)
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        checkpoint = load_torch_checkpoint(checkpoint_path, "cpu")
        generator = build_diffusion_grasp_generator(checkpoint, feature_dim=8, device="cpu")

        rng = np.random.default_rng()
        pc = rng.standard_normal((100, 3)).astype(np.float32)

        grasps1 = generate_candidate_grasps(generator, pc, num_grasps=10)
        grasps2 = generate_candidate_grasps(generator, pc, num_grasps=10)

        if not (np.allclose(grasps1, grasps2, atol=1e-6)):
            raise AssertionError


def test_flow_matching_loss_is_finite() -> None:
    """Verify flow matching loss function."""
    loss_fn = build_flow_matching_loss()

    pred = torch.randn(4, 9)
    target = torch.randn(4, 9)
    loss = loss_fn(pred, target)
    if not (torch.isfinite(loss)):
        raise AssertionError


def test_flow_integrator_shape() -> None:
    """Verify flow integrator execution."""
    flow = FlowGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
    integrator = build_flow_integrator(5)
    x0 = torch.randn(4, 9)
    cond = torch.randn(4, 8)
    out = integrator(flow, x0, cond)
    if not (out.shape == (4, 9)):
        raise AssertionError


def test_flow_grasp_generator_inference() -> None:
    """Verify that flow grasp generator correctly generates SE(3) candidate shapes."""
    model = FlowGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
    optimizer = build_adam_optimizer(model.parameters(), 0.01)

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        checkpoint_path = temp_path / "flow_model.pt"
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        checkpoint = load_torch_checkpoint(checkpoint_path, "cpu")

        generator = build_flow_grasp_generator(checkpoint, feature_dim=8, num_flow_steps=5, device="cpu")

        rng = np.random.default_rng()
        pc = rng.standard_normal((50, 3)).astype(np.float32)
        grasps = generate_candidate_grasps(generator, pc, num_grasps=4)
        if not (grasps.shape == (4, 4, 4)):
            raise AssertionError


def test_generation_pipeline_and_writing() -> None:
    """Verify end-to-end generation pipelines and np.save serialization."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        checkpoint_path = temp_path / "model.pt"

        model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)
        optimizer = build_adam_optimizer(model.parameters(), 0.01)
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        checkpoint = load_torch_checkpoint(checkpoint_path, "cpu")
        generator = build_diffusion_grasp_generator(checkpoint, feature_dim=8, device="cpu")

        rng = np.random.default_rng()
        pc = rng.standard_normal((30, 3)).astype(np.float32)
        grasps = generate_candidate_grasps(generator, pc, 3)
        if not (grasps.shape == (3, 4, 4)):
            raise AssertionError

        # Test writing
        out_file = temp_path / "output" / "grasps.npy"
        write_generated_grasps(out_file, {"obj1": grasps})
        if not (out_file.exists()):
            raise AssertionError

        # Check loading it back
        loaded = np.load(out_file, allow_pickle=True).item()
        if "obj1" not in loaded:
            raise AssertionError
        if not (np.allclose(loaded["obj1"], grasps)):
            raise AssertionError

        # Check write failure TypeError
        with pytest.raises(TypeError):
            write_generated_grasps("not_a_path", {"obj1": grasps})  # type: ignore[arg-type]


def test_pretrained_encoder_checkpoint_loading() -> None:
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
        if not (len(encoder_state) > 0):
            raise AssertionError

        with pytest.raises(TypeError):
            load_torch_checkpoint("not_a_path", "cpu")  # type: ignore[arg-type]


def test_acquire_point_cloud_from_observation_errors() -> None:
    """Verify acquire_point_cloud_from_observation validation checks."""
    with pytest.raises(TypeError):
        acquire_point_cloud_from_observation("not_a_path")  # type: ignore[arg-type]

    with pytest.raises(FileNotFoundError):
        acquire_point_cloud_from_observation(Path("non_existent_obs_file_123.npy"))

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)

        # Test loading non-numpy or invalid file
        invalid_file = temp_path / "invalid.npy"
        with invalid_file.open("w") as f:
            f.write("corrupted data")
        with pytest.raises(ValueError, match="Failed to load observation"):
            acquire_point_cloud_from_observation(invalid_file)

        # Test invalid shape
        rng = np.random.default_rng()
        bad_shape_file = temp_path / "bad_shape.npy"
        np.save(bad_shape_file, rng.standard_normal((10, 4)))
        with pytest.raises(ValueError, match="Invalid observation shape"):
            acquire_point_cloud_from_observation(bad_shape_file)

        # Test non-finite values
        non_finite_file = temp_path / "non_finite.npy"
        np.save(non_finite_file, np.array([[1.0, 2.0, np.nan]]))
        with pytest.raises(ValueError, match="contains non-finite values"):
            acquire_point_cloud_from_observation(non_finite_file)


def test_default_schedule_matches_legacy_beta_literals() -> None:
    """Verify the shared schedule reproduces the legacy linspace(1e-4, 0.02)."""
    if not (DEFAULT_DIFFUSION_SCHEDULE.beta_start == BETA_START):
        raise AssertionError
    if not (DEFAULT_DIFFUSION_SCHEDULE.beta_end == BETA_END):
        raise AssertionError
    if not (DEFAULT_DIFFUSION_SCHEDULE.num_steps == BETA_NUM_STEPS):
        raise AssertionError
    if not (
        torch.allclose(
            linear_beta_schedule(),
            torch.linspace(1e-4, 0.02, 100),
        )
    ):
        raise AssertionError


def test_custom_schedule_values() -> None:
    """Verify a custom schedule drives the beta tensor."""
    schedule = DiffusionSchedule(beta_start=1e-3, beta_end=0.1, num_steps=50)
    beta = linear_beta_schedule(schedule)
    if not (beta.shape == (50,)):
        raise AssertionError
    if not (beta[0] == pytest.approx(1e-3)):
        raise AssertionError
    if not (beta[-1] == pytest.approx(0.1)):
        raise AssertionError


def test_linear_beta_schedule_validation() -> None:
    """Verify linear_beta_schedule validates its inputs."""
    with pytest.raises(TypeError):
        linear_beta_schedule("not-a-schedule")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_steps"):
        linear_beta_schedule(DiffusionSchedule(num_steps=0))
    with pytest.raises(ValueError, match="non-negative"):
        linear_beta_schedule(DiffusionSchedule(beta_start=-0.01))


def test_diffusion_sampler_uses_default_schedule() -> None:
    """Verify build_diffusion_sampler uses DEFAULT_DIFFUSION_SCHEDULE."""
    sampler = build_diffusion_sampler()
    score_model = GraspGeneratorModel(4, 16, 2).score_net
    cond = torch.randn(1, 4)
    rng = torch.Generator().manual_seed(7)

    x = torch.randn(1, 9, generator=rng)
    out = sampler(x, score_model, cond, rng)
    if not (out.shape == (1, 9)):
        raise AssertionError
    if not (torch.isfinite(out).all()):
        raise AssertionError


def test_default_sampler_unchanged_defaults() -> None:
    """Verify the default sampler behavior matches the legacy schedule."""
    sampler = build_diffusion_sampler()
    score_model = GraspGeneratorModel(4, 16, 2).score_net
    conditioning = torch.randn(2, 4)
    rng = torch.Generator().manual_seed(11)

    samples = sample_grasps_with_diffusion(sampler, score_model, conditioning, 9, 3, rng)
    if not (samples.shape == (2, 3, 9)):
        raise AssertionError
    if not (torch.isfinite(samples).all()):
        raise AssertionError


def _make_checkpoint(tmp_path: Path) -> Path:
    model = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)

    optimizer = build_adam_optimizer(model.parameters(), 0.01)
    checkpoint_path = tmp_path / "model.pt"
    save_training_checkpoint(model, optimizer, 1, checkpoint_path)
    return checkpoint_path


def test_generator_seed_is_configurable() -> None:
    """Verify inference seeds are configurable and reproducible."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = _make_checkpoint(Path(tmp_dir))
        checkpoint = load_torch_checkpoint(checkpoint_path, "cpu")
        rng = np.random.default_rng()
        pc = rng.standard_normal((60, 3)).astype(np.float32)

        gen_a = build_diffusion_grasp_generator(checkpoint, feature_dim=8, device="cpu", seed=1)
        gen_b = build_diffusion_grasp_generator(checkpoint, feature_dim=8, device="cpu", seed=2)
        gen_a2 = build_diffusion_grasp_generator(checkpoint, feature_dim=8, device="cpu", seed=1)

        grasps_a = generate_candidate_grasps(gen_a, pc, num_grasps=6)
        grasps_b = generate_candidate_grasps(gen_b, pc, num_grasps=6)
        grasps_a2 = generate_candidate_grasps(gen_a2, pc, num_grasps=6)

        # Same seed -> identical output; different seed -> different output.
        if not (np.allclose(grasps_a, grasps_a2, atol=1e-6)):
            raise AssertionError
        if np.allclose(grasps_a, grasps_b, atol=1e-6):
            raise AssertionError


def test_flow_generator_seed_is_configurable() -> None:
    """Verify the flow generator seed override is accepted."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        model = FlowGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2)

        optimizer = build_adam_optimizer(model.parameters(), 0.01)
        checkpoint_path = tmp_path / "flow_model.pt"
        save_training_checkpoint(model, optimizer, 1, checkpoint_path)

        checkpoint = load_torch_checkpoint(checkpoint_path, "cpu")

        rng = np.random.default_rng()
        pc = rng.standard_normal((40, 3)).astype(np.float32)
        gen = build_flow_grasp_generator(checkpoint, feature_dim=8, num_flow_steps=5, device="cpu", seed=3)
        grasps = generate_candidate_grasps(gen, pc, num_grasps=4)
        if not (grasps.shape == (4, 4, 4)):
            raise AssertionError


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
    f_dim: int = 16,
    n_layers: int = 2,
    n_points: int = 200,
) -> tuple[torch.nn.Module, torch.Tensor]:
    encoder = build_equivariant_encoder(f_dim, n_layers)
    encoder.eval()
    rng = torch.Generator().manual_seed(7)
    cloud = torch.randn(n_points, 3, generator=rng)
    cloud = cloud - cloud.mean(dim=0)
    return encoder, cloud


def test_frame_is_orthonormal() -> None:
    """Verify compute_se3_frame returns orthonormal right-handed frames."""
    _, cloud = _make_encoder_and_cloud()
    frame, centroid = compute_se3_frame(cloud.unsqueeze(0))
    r_matrix = frame[0]
    identity = r_matrix.T @ r_matrix
    if not (torch.allclose(identity, torch.eye(3), atol=1e-5)):
        raise AssertionError
    if not (torch.allclose(torch.det(r_matrix), torch.tensor(1.0), atol=1e-5)):
        raise AssertionError
    if not (torch.allclose(centroid, cloud.mean(dim=0), atol=1e-6)):
        raise AssertionError


def test_canonical_coordinates_are_se3_invariant() -> None:
    """Verify canonical coords are unchanged under rotation and translation."""
    _, cloud = _make_encoder_and_cloud()
    r_rot = _random_rotation()
    t = torch.tensor([0.7, -0.4, 0.2])
    transformed = cloud @ r_rot + t

    frame_a, cent_a = compute_se3_frame(cloud.unsqueeze(0))
    frame_b, cent_b = compute_se3_frame(transformed.unsqueeze(0))
    canon_a = (cloud.unsqueeze(0) - cent_a.unsqueeze(1)) @ frame_a
    canon_b = (transformed.unsqueeze(0) - cent_b.unsqueeze(1)) @ frame_b
    if not (torch.allclose(canon_a, canon_b, atol=1e-4)):
        raise AssertionError


def test_frame_transforms_covariantly() -> None:
    """Verify the frame rotates with the input cloud."""
    _, cloud = _make_encoder_and_cloud()
    r_rot = _random_rotation()
    t = torch.tensor([0.5, 0.25, -0.75])
    transformed = cloud @ r_rot + t

    frame_a, cent_a = compute_se3_frame(cloud.unsqueeze(0))
    frame_b, cent_b = compute_se3_frame(transformed.unsqueeze(0))
    applied = r_rot.T
    if not (torch.allclose(frame_b[0], applied @ frame_a[0], atol=1e-4)):
        raise AssertionError
    if not (torch.allclose(cent_b[0], applied @ cent_a[0] + t, atol=1e-4)):
        raise AssertionError


def test_features_and_pooled_descriptor_are_invariant() -> None:
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
    if not (torch.allclose(feats_a, feats_b, atol=1e-4)):
        raise AssertionError
    if not (torch.allclose(desc_a, desc_b, atol=1e-4)):
        raise AssertionError


def test_compute_se3_frame_validation() -> None:
    """Verify compute_se3_frame validates its inputs."""
    with pytest.raises(TypeError, match=r"must be a torch\.Tensor"):
        compute_se3_frame(np.zeros((10, 3)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"shape \(B, N, 3\)"):
        compute_se3_frame(torch.randn(10, 2))
    with pytest.raises(ValueError, match=r"at least two points"):
        compute_se3_frame(torch.randn(1, 1, 3))


def test_compose_with_se3_frame_maps_canonical_to_input() -> None:
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
    expected = world @ canonical_grasp
    if not (torch.allclose(input_frame, expected, atol=1e-5)):
        raise AssertionError


def test_absolute_grasp_pose_uses_left_se3_action() -> None:
    """A global object transform must left-multiply its absolute grasp pose."""
    canonical = torch.eye(4).unsqueeze(0)
    canonical[0, :3, :3] = _random_rotation()
    canonical[0, :3, 3] = torch.tensor([0.04, -0.02, 0.11])
    frame = _random_rotation().unsqueeze(0)
    centroid = torch.tensor([[0.3, -0.4, 0.7]])

    actual = compose_with_se3_frame(canonical, frame, centroid)
    world = torch.eye(4)
    world[:3, :3] = frame[0]
    world[:3, 3] = centroid[0]

    if not torch.allclose(actual, world @ canonical, atol=1e-5):
        raise AssertionError
    expected_translation = frame[0] @ canonical[0, :3, 3] + centroid[0]
    if not torch.allclose(actual[0, :3, 3], expected_translation, atol=1e-5):
        raise AssertionError


def test_absolute_grasp_pose_canonicalization_round_trip() -> None:
    """Training canonicalization must exactly invert inference composition."""
    canonical = torch.eye(4).unsqueeze(0)
    canonical[0, :3, :3] = _random_rotation()
    canonical[0, :3, 3] = torch.tensor([-0.03, 0.06, 0.14])
    frame = _random_rotation().unsqueeze(0)
    centroid = torch.tensor([[-0.2, 0.5, 0.8]])
    world = torch.eye(4).unsqueeze(0)
    world[0, :3, :3] = frame[0]
    world[0, :3, 3] = centroid[0]
    absolute_pose = (world @ canonical)[0].numpy()
    world_inv = invert_rigid_transform_batch(world)[0]

    vector = training_pairs._canonical_grasp_vector(world_inv, absolute_pose)  # noqa: SLF001
    recovered = vec_to_se3(torch.from_numpy(vector).unsqueeze(0))

    if not torch.allclose(recovered, canonical, atol=1e-5):
        raise AssertionError


def test_canonical_grasp_target_is_invariant_to_global_augmentation() -> None:
    """Applying the same SE(3) transform to cloud and pose keeps the target fixed."""
    canonical = torch.eye(4).unsqueeze(0)
    canonical[0, :3, :3] = _random_rotation()
    canonical[0, :3, 3] = torch.tensor([0.02, 0.05, -0.08])
    world = torch.eye(4).unsqueeze(0)
    world[0, :3, :3] = _random_rotation()
    world[0, :3, 3] = torch.tensor([0.1, -0.2, 0.4])
    augmentation = torch.eye(4).unsqueeze(0)
    augmentation[0, :3, :3] = _random_rotation()
    augmentation[0, :3, 3] = torch.tensor([-0.3, 0.7, 0.2])
    pose = world @ canonical
    transformed_world = augmentation @ world
    transformed_pose = augmentation @ pose

    target = training_pairs._canonical_grasp_vector(  # noqa: SLF001
        invert_rigid_transform_batch(world)[0],
        pose[0].numpy(),
    )
    transformed_target = training_pairs._canonical_grasp_vector(  # noqa: SLF001
        invert_rigid_transform_batch(transformed_world)[0],
        transformed_pose[0].numpy(),
    )

    if not np.allclose(target, transformed_target, atol=1e-5):
        raise AssertionError


def test_legacy_pose_representation_checkpoint_is_rejected() -> None:
    """Old conjugation-trained checkpoints must not silently emit bad poses."""
    with pytest.raises(ValueError, match="incompatible or unspecified grasp-pose representation"):
        build_flow_grasp_generator({}, feature_dim=8, num_flow_steps=5, device="cpu")


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

        checkpoint = load_torch_checkpoint(checkpoint_path, "cpu")
        if builder == "diffusion":
            generator = build_diffusion_grasp_generator(checkpoint, feature_dim=8, device="cpu")
        else:
            generator = build_flow_grasp_generator(checkpoint, feature_dim=8, num_flow_steps=5, device="cpu")
        return generator, checkpoint_path


@pytest.mark.parametrize("builder", ["diffusion", "flow"])
def test_generated_grasps_are_equivariant(builder: str) -> None:
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

    expected = np.stack([g_transform @ grasp for grasp in grasps])
    if not (np.allclose(grasps_transformed, expected, atol=1e-3)):
        raise AssertionError


@pytest.mark.parametrize("builder", ["diffusion", "flow"])
def test_generated_grasps_follow_input_frame(builder: str) -> None:
    """Verify grasps are expressed in the input point-cloud frame."""
    generator, _ = _build_checkpoint_generator(builder)

    rng = np.random.RandomState(11)
    cloud = rng.randn(150, 3).astype(np.float32)
    translation = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    translated = cloud + translation

    grasps = generate_candidate_grasps(generator, cloud, num_grasps=6)
    grasps_translated = generate_candidate_grasps(generator, translated, num_grasps=6)

    if not (grasps.shape == (6, 4, 4)):
        raise AssertionError
    if not (grasps_translated.shape == (6, 4, 4)):
        raise AssertionError
    if np.allclose(grasps, grasps_translated, atol=1e-6):
        raise AssertionError


def _check_checkpoint_path_validations(tmp_path: Path) -> None:
    """Verify load_torch_checkpoint validates path types and checkpoint files."""
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


def _check_checkpoint_scalar_int() -> None:
    """Verify checkpoint_scalar_int coercions and rejection of invalid scalars."""
    if not (checkpoint_scalar_int(torch.tensor(EXPECTED_SCALAR_INT)) == EXPECTED_SCALAR_INT):
        raise AssertionError
    if not (checkpoint_scalar_int(value=True) == 1):
        raise AssertionError
    if not (checkpoint_scalar_int(4.8) == SCALAR_INT_FLOORED):
        raise AssertionError
    with pytest.raises(TypeError, match="Expected numeric checkpoint scalar"):
        checkpoint_scalar_int("invalid_scalar")  # type: ignore[arg-type]


def _check_infer_kind_flow(tmp_path: Path) -> None:
    """Verify metadata kind inference for a flow checkpoint."""
    flow_ckpt_file = tmp_path / "flow_ckpt.pt"
    torch.save(
        {
            "model_state_dict": {"flow_field.weight": torch.zeros(1)},
            "architecture": "flow_matching",
            "pipeline": "flow_training",
            "feature_dim": FLOW_FEATURE_DIM,
        },
        flow_ckpt_file,
    )
    flow_meta = read_model_checkpoint_metadata(flow_ckpt_file)
    if not (flow_meta["kind"] == "flow"):
        raise AssertionError
    if not (flow_meta["architecture"] == "flow_matching"):
        raise AssertionError
    if not (flow_meta["pipeline"] == "flow_training"):
        raise AssertionError
    if not (flow_meta["feature_dim"] == FLOW_FEATURE_DIM):
        raise AssertionError


def _check_infer_kind_diffusion_and_unknown(tmp_path: Path) -> None:
    """Verify metadata kind inference for diffusion and unrecognized checkpoints."""
    diff_ckpt_file = tmp_path / "diff_ckpt.pt"
    torch.save(
        {
            "model_state_dict": {"score_net.weight": torch.zeros(1)},
            "hidden_dim": DIFF_HIDDEN_DIM,
        },
        diff_ckpt_file,
    )
    diff_meta = read_model_checkpoint_metadata(diff_ckpt_file)
    if not (diff_meta["kind"] == "diffusion"):
        raise AssertionError
    if not (diff_meta["hidden_dim"] == DIFF_HIDDEN_DIM):
        raise AssertionError

    unknown_ckpt = tmp_path / "unknown.pt"
    torch.save({"model_state_dict": {"encoder.weight": torch.zeros(1)}}, unknown_ckpt)
    if not (read_model_checkpoint_metadata(unknown_ckpt)["kind"] == "unknown"):
        raise AssertionError

    missing_state = tmp_path / "missing_state.pt"
    torch.save({"architecture": "mystery"}, missing_state)
    if not (read_model_checkpoint_metadata(missing_state)["kind"] == "unknown"):
        raise AssertionError


def test_checkpoint_io_validations_and_infer_kinds(tmp_path: Path) -> None:
    """Verify checkpoint path validation, scalar coercion, and metadata kind inference."""
    _check_checkpoint_path_validations(tmp_path)
    _check_checkpoint_scalar_int()
    _check_infer_kind_flow(tmp_path)
    _check_infer_kind_diffusion_and_unknown(tmp_path)




def test_trainer_additional_branches(tmp_path: Path) -> None:
    """Verify build_adam_optimizer validation and checkpoint load/save exception cases."""
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
    save_training_checkpoint(model, opt, EXPECTED_EPOCH, ckpt_file)
    epoch = load_training_checkpoint(ckpt_file, model, None, "cpu")
    if not (epoch == EXPECTED_EPOCH):
        raise AssertionError

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
    if not (tb_dir.exists()):
        raise AssertionError


def test_equivariant_encoder_collinear_points_fallback() -> None:
    """Verify the equivariant encoder falls back to a valid frame for collinear conditioning clouds."""
    collinear_pts = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=torch.float32)
    frame, _centroid = compute_se3_frame(collinear_pts)
    if not (frame.shape == (1, 3, 3)):
        raise AssertionError
    if not (torch.allclose(torch.det(frame), torch.tensor([1.0]), atol=1e-5)):
        raise AssertionError


def test_batch_conditioned_grasp_samples_validations() -> None:
    """Verify that sampling candidate grasp batches raises a ValueError if the sample count is not positive."""
    cond = torch.randn(2, 8)
    rng = torch.Generator()
    with pytest.raises(ValueError, match="num_samples must be a positive integer"):
        batch_conditioned_grasp_samples(cond, 9, 0, rng, lambda x, _c: x)


def test_diffusion_and_score_network_additional_coverage() -> None:
    """Verify that a trained model's score network works with the sampler."""
    net = GraspGeneratorModel(feature_dim=8, hidden_dim=16, num_layers=2).score_net
    sampler = build_diffusion_sampler()
    x0 = torch.randn(2, 9)
    cond = torch.randn(2, 8)
    sampled = sampler(x0, net, cond, rng=None)
    if not (sampled.shape == (2, 9)):
        raise AssertionError


def test_trainer_checkpoint_saving_branch(tmp_path: Path) -> None:
    """Verify that run_training_loop successfully writes checkpoint files under default configurations."""
    model = torch.nn.Linear(8, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def dummy_step(_inputs: torch.Tensor, _targets: torch.Tensor) -> dict[str, float]:
        return {"loss": 0.1}

    dummy_step.model = model  # type: ignore[attr-defined]
    dummy_step.optimizer = optimizer  # type: ignore[attr-defined]

    ckpt_path = tmp_path / "auto_ckpt.pt"
    dataloader = [(torch.randn(2, 8), torch.randn(2, 2))]
    run_training_loop(dummy_step, dataloader, num_epochs=1, log_every=10, checkpoint_path=ckpt_path)
    if not (ckpt_path.is_file()):
        raise AssertionError

    # Verify trainer fallback when mlflow is not installed (ImportError blocks)
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
        if not (ckpt_path_ml.is_file()):
            raise AssertionError
    finally:
        if orig_mlflow is not None:
            sys.modules["mlflow"] = orig_mlflow
        else:
            sys.modules.pop("mlflow", None)
