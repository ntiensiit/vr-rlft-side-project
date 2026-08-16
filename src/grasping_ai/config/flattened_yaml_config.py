"""Flattened dot-key access to composed Hydra project configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, overload

import numpy as np
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf

from .config import (
    DEFAULT_CONFIG_DIR,
    compose_config,
    config_get,
    config_value,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

T = TypeVar("T")

# ``project.yaml`` omits the evaluation group; library modules still need its metrics.
_DEFAULT_LIBRARY_OVERRIDES = ("+evaluation=default",)


class FlattenedYAMLConfig:
    """Read nested configuration through flattened dot keys backed by Hydra."""

    def __init__(self, cfg: DictConfig | Mapping[str, Any]) -> None:
        """Initialize from a composed Hydra config or a plain mapping."""
        self._cfg: DictConfig = (
            cfg if isinstance(cfg, DictConfig) else OmegaConf.create(dict(cfg))
        )

    @classmethod
    def from_hydra(
        cls,
        config_dir: Path | None = None,
        config_name: str = "project",
        overrides: list[str] | None = None,
        *,
        library_defaults: bool = True,
    ) -> FlattenedYAMLConfig:
        """Build a flattened view of the composed Hydra project config."""
        resolved_dir = config_dir or DEFAULT_CONFIG_DIR
        merged_overrides = list(overrides or [])
        if library_defaults and config_name == "project":
            merged_overrides = list(_DEFAULT_LIBRARY_OVERRIDES) + merged_overrides
        cfg = compose_config(resolved_dir, config_name, merged_overrides)
        return cls(cfg)

    @staticmethod
    def _split_key(key: str) -> tuple[str, ...]:
        return tuple(part for part in key.split(".") if part)

    @overload
    def get(self, key: str) -> Any: ...

    @overload
    def get(self, key: str, default: T) -> T: ...

    def get(self, key: str, default: object = None) -> object:
        """Return a nested configuration value addressed by a dot-separated key."""
        parts = self._split_key(key)
        if not parts:
            return default
        return config_get(self._cfg, *parts, default=default)

    @overload
    def get_path(self, *keys: str) -> Any: ...

    @overload
    def get_path(self, *keys: str, default: T) -> T: ...

    def get_path(self, *keys: str, default: object = None) -> object:
        """Return a nested configuration value addressed by path segments."""
        if not keys:
            return default
        value = config_get(self._cfg, *keys, default=default)
        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(value, resolve=True)
        return value

    def value(
        self,
        *keys: str,
        value_type: type[T],
        default: object = MISSING,
        **kwargs: bool,
    ) -> T:
        """Return a typed nested configuration value.

        Pass a single dot-separated key (``"paths.dataset_root"``) or multiple
        path segments. Use multiple segments with ``script_or=True``.
        """
        if kwargs.get("script_or") or len(keys) != 1:
            parts = keys
        elif "." in keys[0]:
            parts = self._split_key(keys[0])
        else:
            parts = keys
        if not parts:
            msg = "Config key must contain at least one segment"
            raise ValueError(msg)
        return config_value(self._cfg, *parts, value_type=value_type, default=default, **kwargs)  # type: ignore[return-value]

    def numpy_dtype(self, key: str = "array_dtype", default: str = "float32") -> type[np.floating[Any]]:
        """Resolve a configured dtype name to a NumPy floating scalar type."""
        name = str(self.get(key, default))
        dtype = getattr(np, name, None)
        if dtype is None or not isinstance(dtype, type):
            msg = f"Unsupported numpy dtype name: {name!r}"
            raise ValueError(msg)
        return dtype

    @property
    def cfg(self) -> DictConfig:
        """Return the composed Hydra ``DictConfig``."""
        return self._cfg

    @property
    def source(self) -> dict[str, Any]:
        """Return the nested source mapping for ``OmegaConf.create(...)`` callers."""
        container = OmegaConf.to_container(self._cfg, resolve=False)
        if not isinstance(container, dict):
            msg = "Hydra config root must be a mapping"
            raise TypeError(msg)
        return {str(key): value for key, value in container.items()}

    def __getitem__(self, key: str) -> object:
        """Return the value at ``key``, raising when it is missing."""
        parts = self._split_key(key)
        return config_get(self._cfg, *parts, required=True)

    def __contains__(self, key: str) -> bool:
        """Return whether ``key`` addresses an existing config value."""
        parts = self._split_key(key)
        if not parts:
            return False
        return OmegaConf.select(self._cfg, ".".join(parts), default=MISSING) is not MISSING

    def __repr__(self) -> str:
        """Return a short summary of the flattened config."""
        return f"FlattenedYAMLConfig({len(self.source)} top-level keys)"


FLATTENED_YAML_CONFIG = FlattenedYAMLConfig.from_hydra()
