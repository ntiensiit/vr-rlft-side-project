"""Validate filesystem paths used by pipelines."""

from __future__ import annotations

from pathlib import Path


def require_path(value: object, name: str) -> Path:
    """Validate that ``value`` is a ``pathlib.Path`` instance.

    Args:
        value: Candidate path object to validate.
        name: Parameter name used in error messages.

    Returns:
        ``value`` unchanged when it is a ``pathlib.Path`` instance.

    Raises:
        TypeError: If ``value`` is not a ``pathlib.Path`` instance.
    """
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be a pathlib.Path instance")
    return value


def require_optional_path(value: object | None, name: str) -> Path | None:
    """Validate an optional ``pathlib.Path`` argument.

    Args:
        value: Candidate path object or ``None``.
        name: Parameter name used in error messages.

    Returns:
        ``None`` when ``value`` is ``None``; otherwise ``value`` as a
        ``pathlib.Path`` instance.

    Raises:
        TypeError: If ``value`` is neither ``None`` nor a ``pathlib.Path``
            instance.
    """
    if value is None:
        return None
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be a pathlib.Path instance or None")
    return value
