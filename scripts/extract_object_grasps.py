from __future__ import annotations

import argparse
from pathlib import Path

from grasping_ai.pipelines.generate_grasps import (
    load_generated_grasps,
    write_generated_grasps_array,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a single object's grasp poses")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key", type=str, default="object_0")
    args = parser.parse_args()
    grasps = load_generated_grasps(args.input, object_key=args.key)
    write_generated_grasps_array(args.output, grasps)
