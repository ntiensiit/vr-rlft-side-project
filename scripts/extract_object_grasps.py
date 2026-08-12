import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a single object's grasp poses")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key", type=str, default="object_0")
    args = parser.parse_args()
    data = np.load(args.input, allow_pickle=True).item()
    np.save(args.output, data[args.key])


if __name__ == "__main__":
    main()
