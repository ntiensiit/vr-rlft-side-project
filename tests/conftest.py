"""Shared pytest configuration."""

# Open3D must initialize before NumPy/torch trigger alternate libGL loading on
# headless Linux runners (GitHub Actions). See numpy#27589.
import open3d as _open3d  # noqa: F401
