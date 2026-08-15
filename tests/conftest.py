from __future__ import annotations
import contextlib
import sys
from pathlib import Path

import pytest

# Torch Optimizer.__init__ is wrapped by torch._compile._disable_dynamo, which
# imports torch._dynamo -> triton. Official Linux torch wheels pull in triton;
# importing it segfaults on headless GitHub runners. Blocking the import makes
# has_triton_package() return False so Adam construction stays eager/CPU-safe.
# (Windows local installs are torch+cpu and do not ship triton.)
sys.modules.setdefault("triton", None)

# Open3D must initialize before NumPy/torch trigger alternate libGL loading on
# headless Linux runners (GitHub Actions). See numpy#27589.
with contextlib.suppress(Exception):
    import open3d as _open3d  # noqa: F401


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers for the pytest suite.

    Adds the 'slow' marker to identify long-running end-to-end tests.
    """
    config.addinivalue_line(
        "markers",
        "slow: end-to-end / artifact-chain tests that exercise full pipelines "
        "and may take tens of seconds; skip with ``-m 'not slow'``.",
    )


@pytest.fixture(scope="session")
def panda_robot_xml() -> Path:
    """Return the shipped Franka Emika Panda MJCF used by runtime scripts.

    Returns:
        Path to ``deploy/robot.xml``.

    Raises:
        pytest.skip.Exception: If the Panda MJCF is not present in the tree.
    """
    path = Path(__file__).resolve().parents[1] / "deploy" / "robot.xml"
    if not path.is_file():
        pytest.skip(f"Franka Panda MJCF not found: {path}")
    return path
