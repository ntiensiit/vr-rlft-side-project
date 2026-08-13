"""Shared pytest configuration."""

import os

# Disable Torch Dynamo / compile before any torch.optim construction. On headless
# Linux CI, Adam.__init__ otherwise loads triton via torch._dynamo and segfaults.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# Open3D must initialize before NumPy/torch trigger alternate libGL loading on
# headless Linux runners (GitHub Actions). See numpy#27589.
import open3d as _open3d  # noqa: F401
