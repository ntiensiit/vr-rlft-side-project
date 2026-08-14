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


def load_project_yaml_config(
    config_dir: Path,
    *config_names: str,
    overrides: list[str] | None = None,
) -> dict[str, object]:
    """Load and compose the project Hydra config from a directory.

    When ``config.yaml`` is present, composes all configured defaults via
    Hydra and returns a plain mapping. Otherwise falls back to deep-merging
    the requested ``<name>.yaml`` layers for legacy or partial test configs.

    Args:
        config_dir: Directory containing ``config.yaml`` or layer YAML files.
        *config_names: Ignored when ``config.yaml`` exists; otherwise basenames
            without ``.yaml``, merged in order.
        overrides: Optional Hydra override strings such as ``seed=100``.

    Returns:
        A merged configuration mapping.

    Raises:
        TypeError: If ``config_dir`` is not a ``pathlib.Path`` instance or the
            composed root is not a mapping.
        FileNotFoundError: If Hydra cannot find ``config.yaml``.
    """
    if not isinstance(config_dir, Path):
        raise TypeError("config_dir must be a pathlib.Path instance")

    config_yaml = config_dir / "config.yaml"
    if config_yaml.is_file():
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        if overrides is None:
            overrides = parse_config_overrides_from_argv()
        with initialize_config_dir(
            config_dir=str(config_dir.resolve()),
            version_base=None,
        ):
            cfg = compose(config_name="config", overrides=overrides or [])
        container = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(container, dict):
            raise TypeError("Hydra config root must be a mapping")
        from typing import cast

        return cast(dict[str, object], container)

    merged: dict[str, object] = {}
    for name in config_names:
        path = config_dir / f"{name}.yaml"
        if path.is_file():
            merged = merge_yaml_mappings(merged, load_yaml_mapping(path))
    return merged


def parse_config_overrides_from_argv(argv: list[str] | None = None) -> list[str]:
    """Parse Hydra-style override strings from command-line arguments.

    Args:
        argv: Optional argument vector. When omitted, ``sys.argv[1:]`` is used.

    Returns:
        Override strings of the form ``key=value`` or ``group=key`` that are
        not consumed by argparse flags.
    """
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    overrides: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--config-dir":
            index += 2
            continue
        if arg.startswith("--config-dir="):
            index += 1
            continue
        if arg.startswith("--") and "=" not in arg:
            index += 2 if index + 1 < len(args) and not args[index + 1].startswith("-") else 1
            continue
        if "=" in arg and not arg.startswith("-"):
            overrides.append(arg)
        index += 1
    return overrides


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


def optional_cli_path(value: str) -> Path | None:
    """Parse an optional filesystem path from a CLI string value.

    Args:
        value: Raw argument text from argparse. The literals ``none`` and an
            empty string are treated as absent.

    Returns:
        A ``pathlib.Path`` when ``value`` names a path, otherwise ``None``.
    """
    text = str(value).strip()
    if text.lower() in {"", "none"}:
        return None
    return Path(text)


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
            raise TypeError(f"Config path {'.'.join(keys)!r} traverses a non-mapping value")
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


def config_float(
    config: dict[str, object],
    *keys: str,
    default: float,
) -> float:
    """Read a nested float-valued config entry.

    Args:
        config: Loaded configuration mapping.
        *keys: Nested mapping keys.
        default: Value returned when the path is absent.

    Returns:
        The configured floating-point value, or ``default``.

    Raises:
        TypeError: If the configured value is not a number.
    """
    value = config_get(config, *keys, default=default)
    if value is default:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Config path {'.'.join(keys)!r} must be a number")
    return float(value)


def config_int(
    config: dict[str, object],
    *keys: str,
    default: int,
) -> int:
    """Read a nested integer-valued config entry.

    Args:
        config: Loaded configuration mapping.
        *keys: Nested mapping keys.
        default: Value returned when the path is absent.

    Returns:
        The configured integer value, or ``default``.

    Raises:
        TypeError: If the configured value is not a number.
    """
    value = config_get(config, *keys, default=default)
    if value is default:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Config path {'.'.join(keys)!r} must be an integer")
    return int(value)


def config_bool(
    config: dict[str, object],
    *keys: str,
    default: bool,
) -> bool:
    """Read a nested boolean-valued config entry.

    Args:
        config: Loaded configuration mapping.
        *keys: Nested mapping keys.
        default: Value returned when the path is absent.

    Returns:
        The configured boolean value, or ``default``.

    Raises:
        TypeError: If the configured value is not a boolean.
    """
    value = config_get(config, *keys, default=default)
    if value is default:
        return default
    if not isinstance(value, bool):
        raise TypeError(f"Config path {'.'.join(keys)!r} must be a boolean")
    return value


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
