from __future__ import annotations

import argparse
from pathlib import Path

from grasping_ai.training.checkpoint_io import read_model_checkpoint_metadata


def print_model_info_main(checkpoint_path: Path, device: str) -> None:
    """Print architecture metadata stored in a model checkpoint.

    Intended as a quick pre-inference sanity check so callers can confirm
    ``feature_dim``, ``hidden_dim``, and ``num_layers`` (or RL policy dims)
    before invoking ``scripts/run_grasp_inference.py`` or
    ``scripts/run_rl_evaluation.py``.

    Args:
        checkpoint_path: Path to a ``.pt`` checkpoint on disk.
        device: Device identifier passed to ``torch.load`` as ``map_location``.
    """
    metadata = read_model_checkpoint_metadata(checkpoint_path, device)
    for key in (
        "checkpoint_path",
        "kind",
        "pipeline",
        "architecture",
        "feature_dim",
        "hidden_dim",
        "num_layers",
        "observation_dim",
        "action_dim",
        "epoch",
        "seed",
    ):
        if key in metadata and metadata[key] is not None:
            print(f"{key}: {metadata[key]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print architecture metadata from a model checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used when loading the checkpoint (default: cpu)",
    )
    args = parser.parse_args()
    print_model_info_main(checkpoint_path=args.checkpoint, device=args.device)
