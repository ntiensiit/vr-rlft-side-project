from __future__ import annotations

from pathlib import Path

import pytest

from grasping_ai.utils.numerics import GRASP_DISTANCE_EPS, NORM_EPS
from grasping_ai.utils.path_validation import require_optional_path, require_path


def test_require_path_accepts_path_instance() -> None:
    """Accept a pathlib.Path without raising."""
    path = Path("example.pt")
    assert require_path(path, "example_path") is path


def test_require_path_rejects_non_path() -> None:
    """Reject non-Path values with a descriptive TypeError."""
    with pytest.raises(TypeError, match=r"example_path must be a pathlib\.Path instance"):
        require_path("not-a-path", "example_path")


def test_require_optional_path_accepts_none() -> None:
    """Allow None for optional path parameters."""
    assert require_optional_path(None, "optional_path") is None


def test_require_optional_path_rejects_invalid_type() -> None:
    """Reject non-Path, non-None values for optional paths."""
    with pytest.raises(TypeError, match=r"optional_path must be a pathlib\.Path instance or None"):
        require_optional_path("bad", "optional_path")


def test_numerics_constants_are_positive() -> None:
    """Shared tolerances must remain strictly positive."""
    for value in (NORM_EPS, GRASP_DISTANCE_EPS):
        assert value > 0.0
