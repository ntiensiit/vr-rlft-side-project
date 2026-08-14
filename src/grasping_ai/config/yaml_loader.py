from __future__ import annotations

import argparse
from pathlib import Path

import yaml  # type: ignore[import-untyped]


def load_yaml_mapping(path: Path) -> dict[str, object]:
    """Load a YAML mapping from disk.

    Args:
        path: Path to a YAML file containing a top-level mapping.

    Returns:
        The parsed mapping. An empty file yields an empty mapping.

    Raises:
        TypeError: If ``path`` is not a ``pathlib.Path`` instance.
        FileNotFoundError: If ``path`` does not exist.
        TypeError: If the YAML root is not a mapping.
    """
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path instance")
    if not path.is_file():
        raise FileNotFoundError(f"YAML config file not found: {path}")
    with path.open(encoding="utf-8") as fp:
        loaded = yaml.safe_load(fp)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"YAML root in {path} must be a mapping")
    return loaded


def merge_yaml_mappings(*mappings: dict[str, object]) -> dict[str, object]:
    """Deep-merge YAML mappings left-to-right.

    Nested mappings are merged recursively. Later scalar and list values
    replace earlier ones.

    Args:
        *mappings: Mapping objects to merge.

    Returns:
        A new merged mapping.
    """
    merged: dict[str, object] = {}
    for mapping in mappings:
        if not isinstance(mapping, dict):
            raise TypeError("each mapping must be a dict")
        merged = _deep_merge_mappings(merged, mapping)
    return merged


def _deep_merge_mappings(
    base: dict[str, object],
    override: dict[str, object],
) -> dict[str, object]:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_mappings(existing, value)
        else:
            merged[key] = value
    return merged


def load_project_yaml_config(config_dir: Path, *config_names: str) -> dict[str, object]:
    """Load and deep-merge named project YAML configs from a directory.

    Missing config files are skipped so callers can request optional layers.

    Args:
        config_dir: Directory containing ``<name>.yaml`` files.
        *config_names: Basenames without the ``.yaml`` suffix, merged in order.

    Returns:
        A merged mapping of all present config files.

    Raises:
        TypeError: If ``config_dir`` is not a ``pathlib.Path`` instance.
    """
    if not isinstance(config_dir, Path):
        raise TypeError("config_dir must be a pathlib.Path instance")
    merged: dict[str, object] = {}
    for name in config_names:
        path = config_dir / f"{name}.yaml"
        if path.is_file():
            merged = merge_yaml_mappings(merged, load_yaml_mapping(path))
    return merged


def parse_config_dir_from_argv(argv: list[str] | None = None) -> Path:
    """Parse ``--config-dir`` from command-line arguments.

    Args:
        argv: Optional argument vector. When omitted, ``sys.argv[1:]`` is used.

    Returns:
        The config directory path, defaulting to ``configs`` relative to the
        current working directory.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parsed, _unknown = parser.parse_known_args(argv)
    return parsed.config_dir


def config_get(
    config: dict[str, object],
    *keys: str,
    default: object | None = None,
    required: bool = False,
) -> object | None:
    """Read a nested config value using object-style key paths.

    Args:
        config: Loaded configuration mapping.
        *keys: Nested mapping keys, e.g. ``"diffusion"``, ``"checkpoint"``.
        default: Value returned when any key in the path is absent.
        required: When ``True``, raise if the path is missing instead of
            returning ``default``.

    Returns:
        The configured value at the nested path, or ``default``.

    Raises:
        ValueError: If ``required`` is ``True`` and the path is missing.
        TypeError: If an intermediate value is not a mapping.
    """
    current: object = config
    visited: list[str] = []
    for key in keys:
        visited.append(key)
        if not isinstance(current, dict):
            raise TypeError(
                f"Config path {'.'.join(keys)!r} traverses a non-mapping value"
            )
        if key not in current:
            if required:
                raise ValueError(f"Missing config key: {'.'.join(visited)}")
            return default
        current = current[key]
    return current


def require_config_value(config: dict[str, object], *keys: str) -> object:
    """Return a required nested config value or raise.

    Args:
        config: Loaded configuration mapping.
        *keys: Nested mapping keys.

    Returns:
        The configured value.

    Raises:
        ValueError: If the nested path is absent.
        TypeError: If an intermediate value is not a mapping.
    """
    return config_get(config, *keys, required=True)


def config_path(
    config: dict[str, object],
    *keys: str,
    default: Path | None = None,
) -> Path | None:
    """Read a nested path-valued config entry.

    Args:
        config: Loaded configuration mapping.
        *keys: Nested mapping keys.
        default: Value returned when the path is absent.

    Returns:
        A ``pathlib.Path`` when the configured value is a non-empty string,
        otherwise ``default``.
    """
    value = config_get(config, *keys, default=default)
    if value is default:
        return default
    if value is None:
        return default
    if not isinstance(value, str) or not value:
        raise ValueError(f"Config path {'.'.join(keys)!r} must be a non-empty string path")
    return Path(value)


def config_str_list(
    config: dict[str, object],
    *keys: str,
    default: list[str] | None = None,
) -> list[str] | None:
    """Read a nested list-of-strings config entry.

    Args:
        config: Loaded configuration mapping.
        *keys: Nested mapping keys.
        default: Value returned when the path is absent.

    Returns:
        A list of strings, or ``default`` when the path is absent.

    Raises:
        ValueError: If the configured value is not a list of strings.
    """
    value = config_get(config, *keys, default=default)
    if value is default or value is None:
        return default
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Config path {'.'.join(keys)!r} must be a list of strings")
    return value


def config_float_list(
    config: dict[str, object],
    *keys: str,
    default: list[float] | None = None,
) -> list[float] | None:
    """Read a nested list-of-floats config entry.

    Args:
        config: Loaded configuration mapping.
        *keys: Nested mapping keys.
        default: Value returned when the path is absent.

    Returns:
        A list of floats, or ``default`` when the path is absent.

    Raises:
        TypeError: If the configured value is not a list of numbers.
    """
    value = config_get(config, *keys, default=default)
    if value is default or value is None:
        return default
    if not isinstance(value, list):
        raise TypeError(f"Config path {'.'.join(keys)!r} must be a list of numbers")
    converted: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise TypeError(f"Config path {'.'.join(keys)!r} must be a list of numbers")
        converted.append(float(item))
    return converted
