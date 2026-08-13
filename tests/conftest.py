"""Shared pytest configuration."""

import sys

# Torch Optimizer.__init__ is wrapped by torch._compile._disable_dynamo, which
# imports torch._dynamo -> triton. Official Linux torch wheels pull in triton;
# importing it segfaults on headless GitHub runners. Blocking the import makes
# has_triton_package() return False so Adam construction stays eager/CPU-safe.
# (Windows local installs are torch+cpu and do not ship triton.)
sys.modules.setdefault("triton", None)

# Open3D must initialize before NumPy/torch trigger alternate libGL loading on
# headless Linux runners (GitHub Actions). See numpy#27589.
import open3d as _open3d  # noqa: F401
