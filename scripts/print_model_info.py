from __future__ import annotations

import argparse
from pathlib import Path

from grasping_ai.training.checkpoint_io import read_model_checkpoint_metadata

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
    metadata = read_model_checkpoint_metadata(args.checkpoint, args.device)
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
