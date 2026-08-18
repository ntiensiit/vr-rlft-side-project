"""Hydra configuration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf

if TYPE_CHECKING:
    from collections.abc import Callable

# Relative to ``scripts/*.py`` when using ``@hydra.main(config_path=...)``.
SCRIPTS_CONFIG_PATH = "../configs"
DEFAULT_CONFIG_DIR = Path("configs")


def _key_path(*keys: str) -> str:
    """Join multiple configuration keys with a dot separator.

    Args:
        *keys: String configuration keys to join.

    Returns:
        A dot-separated string path.
    """
    return ".".join(keys)


def compose_config(
    config_dir: Path = DEFAULT_CONFIG_DIR,
    config_name: str = "config",
    overrides: list[str] | None = None,
) -> DictConfig:
    """Compose a Hydra ``DictConfig`` from ``configs/<config_name>.yaml`` and overrides."""
    config_yaml = (config_dir / config_name).with_suffix(".yaml")
    if not config_yaml.is_file():
        msg = f"Hydra config entrypoint not found: {config_yaml}"
        raise FileNotFoundError(msg)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        return compose(config_name=config_name, overrides=overrides or [])


def hydra_cfg_to_dict(cfg: DictConfig) -> dict[str, object]:
    """Resolve a Hydra ``DictConfig`` to a plain mapping."""
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        msg = "Hydra config root must be a mapping"
        raise TypeError(msg)
    return {str(key): value for key, value in container.items()}


def config_get(
    cfg: DictConfig,
    *keys: str,
    default: object | None = None,
    required: bool = False,
) -> object | None:
    """Read a nested value from a composed Hydra config."""
    selected_default: object = MISSING if required else default
    value = OmegaConf.select(cfg, _key_path(*keys), default=selected_default)
    if value is MISSING:
        if required:
            msg = f"Missing config key: {_key_path(*keys)}"
            raise ValueError(msg)
        return default
    return value


_NO_EARLY_RESULT = object()


@dataclass(frozen=True)
class _ConfigValueQuery:
    """Resolved lookup context for a typed config read."""

    value_type: type[object]
    default: object
    required: bool
    script_or: bool
    config_default: bool
    script_key: str
    domain_keys: tuple[str, ...]


def _domain_or_default(domain_keys: tuple[str, ...], default: object) -> tuple[tuple[str, ...], object]:
    """Return domain lookup keys when present, otherwise an early ``default`` result."""
    if domain_keys:
        return domain_keys, _NO_EARLY_RESULT
    return (), default


def _missing_script_key_result(
    script_key: str,
    value_type: type[object],
    default: object,
) -> tuple[tuple[str, ...], object]:
    """Handle a missing script override that has no domain keys."""
    if default is MISSING and value_type in {list[str], list[float], list[Path]}:
        msg = f"Missing config key: script.{script_key}"
        raise ValueError(msg)
    if default is MISSING:
        return (), default
    if not isinstance(default, (list, ListConfig)):
        msg = f"Default for config key {script_key!r} must be a list"
        raise TypeError(msg)
    return (), list(default)


def _script_lookup_keys(
    cfg: DictConfig,
    keys: tuple[str, ...],
    value_type: type[object],
    default: object,
) -> tuple[tuple[str, ...], object]:
    """Resolve lookup keys for ``script_or=True``; may yield an early result value."""
    if not keys:
        msg = "Script override accessors require a script key"
        raise TypeError(msg)
    script_key, domain_keys = keys[0], keys[1:]
    script_override = OmegaConf.select(cfg, f"script.{script_key}", default=MISSING)
    has_override = script_override is not MISSING and script_override is not None
    if value_type is bool:
        if script_override is not MISSING:
            return ("script", script_key), _NO_EARLY_RESULT
    elif value_type is object:
        if has_override:
            return (), script_override
    elif has_override:
        return ("script", script_key), _NO_EARLY_RESULT
    elif domain_keys:
        return domain_keys, _NO_EARLY_RESULT
    else:
        return _missing_script_key_result(script_key, value_type, default)
    return _domain_or_default(domain_keys, default)


def _lookup_path_value(
    cfg: DictConfig,
    lookup_keys: tuple[str, ...],
    query: _ConfigValueQuery,
) -> tuple[object, bool]:
    """Fetch a raw ``Path`` value, honoring required-path errors."""
    if query.default is MISSING and query.required:
        value = config_get(cfg, *lookup_keys, default=None)
    else:
        value = config_get(cfg, *lookup_keys, default=query.default)
    if value is not query.default and value is not None:
        return value, True
    if query.required:
        if query.script_or:
            msg = f"Missing config path: script.{query.script_key} or {'.'.join(query.domain_keys)}"
        else:
            msg = f"Missing config path: {_key_path(*lookup_keys)}"
        raise ValueError(msg)
    if query.default is MISSING:
        return None, False
    return query.default, False


def _lookup_typed_value(
    cfg: DictConfig,
    lookup_keys: tuple[str, ...],
    query: _ConfigValueQuery,
) -> tuple[object, bool]:
    """Fetch the raw value; the bool result is ``False`` when coercion must be skipped."""
    if query.config_default:
        value = config_get(cfg, *lookup_keys, default=query.default)
        if value is query.default or (query.value_type in {list[str], list[float], Path} and value is None):
            return query.default, False
        return value, True
    if query.value_type is Path:
        return _lookup_path_value(cfg, lookup_keys, query)
    if query.default is MISSING:
        return config_get(cfg, *lookup_keys, required=True), True
    value = config_get(cfg, *lookup_keys, default=query.default)
    if query.required and value is None:
        msg = f"Missing config key: {_key_path(*lookup_keys)}"
        raise ValueError(msg)
    return value, True


def _coerce_config_value(value_type: type[object], raw: object, path: str, *, use_config_default: bool) -> object:
    """Coerce a raw config value to ``value_type`` or raise a descriptive error."""
    coerce_strategies: dict[type[object], Callable[[object, str, bool], object]] = {
        object: lambda raw, _path, _use_config_default: raw,
        float: lambda raw, path, _use_config_default: (
            float(raw)
            if isinstance(raw, int | float) and not isinstance(raw, bool)
            else (_ for _ in ()).throw(TypeError(f"Config path {path!r} must be a number"))
        ),
        int: lambda raw, path, _use_config_default: (
            int(raw)
            if isinstance(raw, int | float) and not isinstance(raw, bool)
            else (_ for _ in ()).throw(TypeError(f"Config path {path!r} must be an integer"))
        ),
        bool: lambda raw, path, _use_config_default: (
            raw
            if isinstance(raw, bool)
            else (_ for _ in ()).throw(TypeError(f"Config path {path!r} must be a boolean"))
        ),
        Path: lambda raw, path, _use_config_default: (
            Path(raw)
            if isinstance(raw, str) and raw
            else (_ for _ in ()).throw(ValueError(f"Config path {path!r} must be a non-empty string path"))
        ),
        list[str]: lambda raw, path, use_config_default: (
            raw
            if use_config_default and isinstance(raw, (list, ListConfig)) and all(isinstance(item, str) for item in raw)
            else list(raw)
            if isinstance(raw, (list, ListConfig)) and all(isinstance(item, str) for item in raw)
            else (_ for _ in ()).throw(ValueError(f"Config path {path!r} must be a list of strings"))
        ),
        list[float]: lambda raw, path, _use_config_default: (
            [float(item) for item in raw]
            if isinstance(raw, (list, ListConfig))
            and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in raw)
            else (_ for _ in ()).throw(TypeError(f"Config path {path!r} must be a list of numbers"))
        ),
        list[Path]: lambda raw, path, use_config_default: (
            [Path(item) for item in (raw if use_config_default else list(raw))]
            if isinstance(raw, (list, ListConfig)) and all(isinstance(item, str) for item in raw)
            else (_ for _ in ()).throw(ValueError(f"Config path {path!r} must be a list of strings"))
        ),
    }

    coerce = coerce_strategies.get(value_type)
    if coerce is None:
        msg = f"Unsupported typed config value: {value_type!r}"
        raise TypeError(msg)
    return coerce(raw, path, use_config_default)


def config_value(  # noqa: PLR0913  # keyword flags are part of the public typed-access API used by tests
    cfg: DictConfig,
    *keys: str,
    value_type: type[object],
    default: object = MISSING,
    script_or: bool = False,
    config_default: bool = False,
    required: bool = False,
) -> object:
    """Read and coerce a nested Hydra config value to ``value_type``."""
    script_key = ""
    domain_keys: tuple[str, ...] = ()
    lookup_keys: tuple[str, ...]

    if script_or:
        lookup_keys, early_result = _script_lookup_keys(cfg, keys, value_type, default)
        script_key, domain_keys = keys[0], keys[1:]
        if early_result is not _NO_EARLY_RESULT:
            return early_result
    else:
        lookup_keys = keys

    query = _ConfigValueQuery(
        value_type=value_type,
        default=default,
        required=required,
        script_or=script_or,
        config_default=config_default,
        script_key=script_key,
        domain_keys=domain_keys,
    )
    value, should_coerce = _lookup_typed_value(cfg, lookup_keys, query)
    if not should_coerce:
        return value
    return _coerce_config_value(value_type, value, _key_path(*lookup_keys), use_config_default=config_default)
