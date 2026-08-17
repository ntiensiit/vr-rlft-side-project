"""Flow-matching training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
from grasping_ai.data.training_pairs import (
    SupervisedGraspDataset,
    validate_grasp_dataset,
)
from grasping_ai.models.flow import (
    FlowGeneratorModel,
)
from grasping_ai.pipelines.supervised_training import build_supervised_dataloader
from grasping_ai.training.checkpoint_io import load_torch_checkpoint
from grasping_ai.training.losses import build_flow_matching_loss
from grasping_ai.training.trainer import (
    SupervisedTrainingStep,
    build_adam_optimizer,
    load_training_checkpoint,
    run_training_loop,
)
from grasping_ai.utils.path_validation import require_path

if TYPE_CHECKING:
    from collections.abc import Callable

DATASET_ROOT = Path(FLATTENED_YAML_CONFIG.get("paths.dataset_root", "data/processed"))
CHECKPOINT_PATH = Path(
    FLATTENED_YAML_CONFIG.get("model.checkpoint", "artifacts/checkpoints/flow_grasp_generator.pt"),
)
FEATURE_DIM = int(FLATTENED_YAML_CONFIG.get("architecture.feature_dim", 64))
HIDDEN_DIM = int(FLATTENED_YAML_CONFIG.get("architecture.hidden_dim", 64))
NUM_LAYERS = int(FLATTENED_YAML_CONFIG.get("architecture.num_layers", 3))
LEARNING_RATE = float(FLATTENED_YAML_CONFIG.get("supervised.learning_rate", 0.001))
NUM_EPOCHS = int(FLATTENED_YAML_CONFIG.get("supervised.num_epochs", 3))
BATCH_SIZE = int(FLATTENED_YAML_CONFIG.get("supervised.batch_size", 2))
DEVICE = str(FLATTENED_YAML_CONFIG.get("device", "cpu"))
SEED = int(FLATTENED_YAML_CONFIG.get("seed", 42))
AUGMENT = bool(FLATTENED_YAML_CONFIG.get("training.augment", False))
MIN_GRASP_SCORE = float(FLATTENED_YAML_CONFIG.get("supervised.min_grasp_score", 0.0))
SCORE_REPEAT_FACTOR = int(FLATTENED_YAML_CONFIG.get("supervised.score_repeat_factor", 0))
SCORE_REPEAT_POWER = float(FLATTENED_YAML_CONFIG.get("supervised.score_repeat_power", 1.0))
LOG_EVERY = int(FLATTENED_YAML_CONFIG.get("training.log_every", 10))
FLOW_NOISE_SAMPLES = int(FLATTENED_YAML_CONFIG.get("training.flow_noise_samples", 1))


def build_flow_training_step(
    model: FlowGeneratorModel,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: str,
    seed: int | None = None,
    noise_samples: int = FLOW_NOISE_SAMPLES,
) -> SupervisedTrainingStep:
    """Build a callable training step closure for a flow-matching model."""
    device_obj = torch.device(device)
    if noise_samples <= 0:
        raise ValueError("noise_samples must be positive")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device_obj).manual_seed(seed)

    def flow_forward(point_clouds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pcs = point_clouds
        x_1 = targets
        batch_size_val = x_1.shape[0]
        cond = model.condition(pcs)
        losses = []
        for _ in range(noise_samples):
            t = torch.rand(
                batch_size_val,
                dtype=x_1.dtype,
                device=device_obj,
                generator=generator,
            )
            x_0 = torch.randn(
                x_1.shape,
                dtype=x_1.dtype,
                device=device_obj,
                generator=generator,
            )
            t_view = t.view(batch_size_val, 1)
            x_t = (1.0 - t_view) * x_0 + t_view * x_1
            target_velocity = x_1 - x_0
            predicted_velocity = model.flow_field(x_t, cond)
            losses.append(loss_fn(predicted_velocity, target_velocity))
        return torch.stack(losses).mean()

    return SupervisedTrainingStep(model, optimizer, device, flow_forward)


def run_flow_training_pipeline(
    dataset_root: Path = DATASET_ROOT,
    checkpoint_path: Path = CHECKPOINT_PATH,
    *,
    experiment_log_dir: Path | None = None,
    pretrained_encoder_path: Path | None = None,
    resume_checkpoint_path: Path | None = None,
    **options: Any,  # noqa: ANN401
) -> None:
    """Run the end-to-end flow-matching training pipeline for grasp generation.

    Mirrors ``run_diffusion_training_pipeline`` but uses a continuous-time flow-matching
    objective on the canonical-frame 9D grasp vectors instead of the discrete
    diffusion score-matching loss.

    The encoder and flow field are trained jointly on a single
    ``FlowGeneratorModel`` and saved together by the checkpoint writer, so
    the train/inference model contract is explicit: ``state_dict()``
    contains both ``encoder.*`` and ``flow_field.*`` keys.

    Args:
        dataset_root: Root directory of the grasp-pose dataset.
        checkpoint_path: Destination path for the trained flow checkpoint.
        feature_dim: Conditioning feature dimension from the encoder.
        hidden_dim: Width of the flow-field hidden layers.
        num_layers: Number of hidden layers in the flow field.
        learning_rate: Learning rate for the Adam optimizer.
        num_epochs: Number of training epochs to perform.
        batch_size: Training batch size.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.
        seed: Optional random seed for reproducible training.
        experiment_log_dir: Optional path to write TensorBoard experiment events.
        pretrained_encoder_path: Optional checkpoint whose encoder weights warm-start
            the flow model before training.
        resume_checkpoint_path: Optional path to a checkpoint to resume training from.
        augment: Whether to apply data augmentation to training samples.
        min_grasp_score: Minimum grasp score threshold for training samples.
        score_repeat_factor: Number of times to repeat samples based on score.
        score_repeat_power: Power scaling factor for score-based repetition.
        options: Optional training overrides keyed by configuration name.
    """
    feature_dim = int(options.pop("feature_dim", FEATURE_DIM))
    hidden_dim = int(options.pop("hidden_dim", HIDDEN_DIM))
    num_layers = int(options.pop("num_layers", NUM_LAYERS))
    learning_rate = float(options.pop("learning_rate", LEARNING_RATE))
    num_epochs = int(options.pop("num_epochs", NUM_EPOCHS))
    batch_size = int(options.pop("batch_size", BATCH_SIZE))
    device = str(options.pop("device", DEVICE))
    seed = options.pop("seed", SEED)
    augment = bool(options.pop("augment", AUGMENT))
    min_grasp_score = float(options.pop("min_grasp_score", MIN_GRASP_SCORE))
    score_repeat_factor = int(options.pop("score_repeat_factor", SCORE_REPEAT_FACTOR))
    score_repeat_power = float(options.pop("score_repeat_power", SCORE_REPEAT_POWER))
    noise_samples = int(options.pop("flow_noise_samples", FLOW_NOISE_SAMPLES))
    if options:
        unexpected = ", ".join(sorted(options))
        msg = f"Unexpected flow training options: {unexpected}"
        raise TypeError(msg)

    require_path(dataset_root, "dataset_root")
    if not dataset_root.exists():
        msg = f"Dataset root does not exist: {dataset_root}"
        raise FileNotFoundError(msg)

    validate_grasp_dataset(dataset_root)

    if seed is not None:
        torch.manual_seed(seed)

    model = FlowGeneratorModel(feature_dim, hidden_dim, num_layers).to(device)
    optimizer = build_adam_optimizer(model.parameters(), learning_rate)

    resume_epoch = 0
    if resume_checkpoint_path is not None:
        resume_epoch = load_training_checkpoint(
            resume_checkpoint_path,
            model,
            optimizer,
            device,
        )

    if pretrained_encoder_path is not None:
        require_path(pretrained_encoder_path, "pretrained_encoder_path")
        checkpoint = load_torch_checkpoint(pretrained_encoder_path, device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        encoder_state: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                encoder_state[key[len("encoder.") :]] = value
            else:
                encoder_state[key] = value
        model.encoder.load_state_dict(encoder_state, strict=False)

    training_dataset = SupervisedGraspDataset(
        dataset_root,
        augment=augment,
        seed=seed,
        min_grasp_score=min_grasp_score,
        score_repeat_factor=score_repeat_factor,
        score_repeat_power=score_repeat_power,
    )

    loss_fn = build_flow_matching_loss()
    training_step = build_flow_training_step(
        model,
        loss_fn,
        optimizer,
        device,
        seed=seed,
        noise_samples=noise_samples,
    )

    dataloader = build_supervised_dataloader(training_dataset, batch_size, seed)

    metadata = {
        "pipeline": "flow",
        "dataset_root": str(dataset_root),
        "checkpoint_path": str(checkpoint_path),
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "device": device,
        "augment": augment,
        **({"seed": seed} if seed is not None else {}),
        **({"resume_from_epoch": resume_epoch} if resume_epoch > 0 else {}),
    }

    run_training_loop(
        training_step,
        dataloader,
        num_epochs,
        checkpoint_path,
        log_every=LOG_EVERY,
        experiment_log_dir=experiment_log_dir,
        metadata=metadata,
        seed=seed,
    )
