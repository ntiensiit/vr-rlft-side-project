from pathlib import Path
from typing import Any

import torch


def load_torch_checkpoint(checkpoint_path: Path, device: str) -> dict[str, Any]:
    """Load a PyTorch checkpoint from disk with shared validation and errors.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        device: Device identifier passed to ``torch.load`` as ``map_location``.

    Returns:
        Deserialized checkpoint dictionary.

    Raises:
        TypeError: If ``checkpoint_path`` is not a ``pathlib.Path`` instance.
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If ``torch.load`` fails.
    """
    if not isinstance(checkpoint_path, Path):
        raise TypeError("checkpoint_path must be a pathlib.Path instance")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint: {e}") from e

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} must deserialize to a dictionary"
        )
    return checkpoint


def _checkpoint_scalar_int(value: object) -> int:
    """Coerce a checkpoint scalar value to ``int``."""
    if isinstance(value, torch.Tensor):
        return int(value.item())
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raise TypeError(f"Expected numeric checkpoint scalar, got {type(value)!r}")


def _infer_checkpoint_kind(checkpoint: dict[str, Any]) -> str:
    """Infer the high-level checkpoint family from its stored fields."""
    from grasping_ai.models.rl_policy import read_rl_policy_metadata

    if read_rl_policy_metadata(checkpoint) is not None:
        return "rl_policy"

    state_dict = checkpoint.get("model_state_dict")
    if isinstance(state_dict, dict):
        keys = list(state_dict.keys())
        if any(key.startswith("flow_field.") for key in keys):
            return "flow"
        if any(key.startswith("score_net.") for key in keys):
            return "diffusion"

    return "unknown"


def read_model_checkpoint_metadata(
    checkpoint_path: Path,
    device: str = "cpu",
) -> dict[str, int | str | None]:
    """Read architecture metadata from a training or RL policy checkpoint.

    Args:
        checkpoint_path: Path to a ``.pt`` checkpoint produced by the training
            or RL export pipelines.
        device: Device identifier passed to ``torch.load`` as ``map_location``.

    Returns:
        A mapping of human-readable metadata fields. Grasp-generation
        checkpoints expose ``feature_dim``, ``hidden_dim``, and ``num_layers``.
        RL policy checkpoints expose ``observation_dim`` and ``action_dim`` in
        addition to ``hidden_dim`` and ``num_layers``.
    """
    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    kind = _infer_checkpoint_kind(checkpoint)

    metadata: dict[str, int | str | None] = {
        "checkpoint_path": str(checkpoint_path),
        "kind": kind,
    }

    for key in ("feature_dim", "hidden_dim", "num_layers", "epoch", "seed"):
        if key in checkpoint:
            metadata[key] = _checkpoint_scalar_int(checkpoint[key])

    from grasping_ai.models.rl_policy import read_rl_policy_metadata

    rl_metadata = read_rl_policy_metadata(checkpoint)
    if rl_metadata is not None:
        obs_dim, action_dim, hidden_dim, num_layers = rl_metadata
        metadata["observation_dim"] = obs_dim
        metadata["action_dim"] = action_dim
        metadata["hidden_dim"] = hidden_dim
        metadata["num_layers"] = num_layers

    if "architecture" in checkpoint and isinstance(checkpoint["architecture"], str):
        metadata["architecture"] = checkpoint["architecture"]

    if "pipeline" in checkpoint and isinstance(checkpoint["pipeline"], str):
        metadata["pipeline"] = checkpoint["pipeline"]

    return metadata
