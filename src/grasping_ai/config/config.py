"""Hydra configuration utilities."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, MISSING, OmegaConf

# Relative to ``scripts/*.py`` when using ``@hydra.main(config_path=...)``.
SCRIPTS_CONFIG_PATH = "../configs"
DEFAULT_CONFIG_DIR = Path("configs")


def _key_path(*keys: str) -> str:
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


def load_project_yaml_config(
    config_dir: Path,
    config_name: str = "config",
    overrides: list[str] | None = None,
) -> DictConfig:
    """Compose project configuration via Hydra."""
    return compose_config(config_dir=config_dir, config_name=config_name, overrides=overrides)


def hydra_cfg_to_dict(cfg: DictConfig) -> dict[str, object]:
    """Resolve a Hydra ``DictConfig`` to a plain mapping."""
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        msg = "Hydra config root must be a mapping"
        raise TypeError(msg)
    return container


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


def config_value(
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
        if not keys:
            msg = "Script override accessors require a script key"
            raise TypeError(msg)
        script_key = keys[0]
        domain_keys = keys[1:]
        script_override = OmegaConf.select(cfg, f"script.{script_key}", default=MISSING)
        if value_type is bool:
            if script_override is not MISSING:
                lookup_keys = ("script", script_key)
            elif domain_keys:
                lookup_keys = domain_keys
            else:
                return default
        elif value_type is object:
            if script_override is not MISSING and script_override is not None:
                return script_override
            elif domain_keys:
                lookup_keys = domain_keys
            else:
                return default
        elif script_override is not MISSING and script_override is not None:
            lookup_keys = ("script", script_key)
        elif domain_keys:
            lookup_keys = domain_keys
        else:
            if default is MISSING and value_type in {list[str], list[float], list[Path]}:
                msg = f"Missing config key: script.{script_key}"
                raise ValueError(msg)
            return list(default) if default is not MISSING else default  # type: ignore[arg-type]
    else:
        lookup_keys = keys

    key_path = _key_path(*lookup_keys)

    if config_default:
        value = config_get(cfg, *lookup_keys, default=default)
        if value is default:
            return default
        elif value_type in {list[str], list[float], Path} and value is None:
            return default
    elif value_type is Path:
        if default is MISSING and required:
            value = config_get(cfg, *lookup_keys, default=None)
        else:
            value = config_get(cfg, *lookup_keys, default=default)
        if value is default or value is None:
            if required:
                if script_or:
                    msg = f"Missing config path: script.{script_key} or {'.'.join(domain_keys)}"
                else:
                    msg = f"Missing config path: {key_path}"
                raise ValueError(msg)
            else:
                return default if default is not MISSING else None
    elif default is MISSING:
        value = config_get(cfg, *lookup_keys, required=True)
    else:
        value = config_get(cfg, *lookup_keys, default=default)
        if required and value is None:
            msg = f"Missing config key: {key_path}"
            raise ValueError(msg)

    coerce_strategies: dict[type[object], object] = {
        object: lambda raw, path, use_config_default: raw,
        float: lambda raw, path, use_config_default: (
            float(raw)
            if isinstance(raw, int | float) and not isinstance(raw, bool)
            else (_ for _ in ()).throw(TypeError(f"Config path {path!r} must be a number"))
        ),
        int: lambda raw, path, use_config_default: (
            int(raw)
            if isinstance(raw, int | float) and not isinstance(raw, bool)
            else (_ for _ in ()).throw(TypeError(f"Config path {path!r} must be an integer"))
        ),
        bool: lambda raw, path, use_config_default: (
            raw
            if isinstance(raw, bool)
            else (_ for _ in ()).throw(TypeError(f"Config path {path!r} must be a boolean"))
        ),
        Path: lambda raw, path, use_config_default: (
            Path(raw)
            if isinstance(raw, str) and raw
            else (_ for _ in ()).throw(ValueError(f"Config path {path!r} must be a non-empty string path"))
        ),
        list[str]: lambda raw, path, use_config_default: (
            raw
            if use_config_default and isinstance(raw, list) and all(isinstance(item, str) for item in raw)
            else list(raw)
            if isinstance(raw, list) and all(isinstance(item, str) for item in raw)
            else (_ for _ in ()).throw(ValueError(f"Config path {path!r} must be a list of strings"))
        ),
        list[float]: lambda raw, path, use_config_default: (
            [float(item) for item in raw]
            if isinstance(raw, list)
            and all(isinstance(item, int | float) and not isinstance(item, bool) for item in raw)
            else (_ for _ in ()).throw(TypeError(f"Config path {path!r} must be a list of numbers"))
        ),
        list[Path]: lambda raw, path, use_config_default: (
            [Path(item) for item in (raw if use_config_default else list(raw))]
            if isinstance(raw, list) and all(isinstance(item, str) for item in raw)
            else (_ for _ in ()).throw(ValueError(f"Config path {path!r} must be a list of strings"))
        ),
    }

    coerce = coerce_strategies.get(value_type)
    if coerce is None:
        msg = f"Unsupported typed config value: {value_type!r}"
        raise TypeError(msg)
    return coerce(value, key_path, config_default)
