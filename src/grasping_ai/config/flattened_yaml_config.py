"""Flattened dot-key access to composed Hydra project configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, MISSING, OmegaConf

from .config import (
    DEFAULT_CONFIG_DIR,
    config_get,
    config_value,
    load_project_yaml_config,
)

# ``project.yaml`` omits the evaluation group; library modules still need its metrics.
_DEFAULT_LIBRARY_OVERRIDES = ("+evaluation=default",)


class FlattenedYAMLConfig:
    """Read nested configuration through flattened dot keys backed by Hydra."""

    def __init__(self, cfg: DictConfig | Mapping[str, Any]) -> None:
        self._cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)

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
        cfg = load_project_yaml_config(resolved_dir, config_name, merged_overrides)
        return cls(cfg)

    @staticmethod
    def _split_key(key: str) -> tuple[str, ...]:
        return tuple(part for part in key.split(".") if part)

    def get(self, key: str, default: Any = None) -> Any:
        """Return a nested configuration value addressed by a dot-separated key."""
        parts = self._split_key(key)
        if not parts:
            return default
        return config_get(self._cfg, *parts, default=default)

    def get_path(self, *keys: str, default: Any = None) -> Any:
        """Return a nested configuration value addressed by path segments."""
        if not keys:
            return default
        return config_get(self._cfg, *keys, default=default)

    def value(
        self,
        *keys: str,
        value_type: type[object],
        default: object = MISSING,
        **kwargs: Any,
    ) -> object:
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
        return config_value(self._cfg, *parts, value_type=value_type, default=default, **kwargs)

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
        return container

    def __getitem__(self, key: str) -> Any:
        parts = self._split_key(key)
        return config_get(self._cfg, *parts, required=True)

    def __contains__(self, key: str) -> bool:
        parts = self._split_key(key)
        if not parts:
            return False
        return OmegaConf.select(self._cfg, ".".join(parts), default=MISSING) is not MISSING

    def __repr__(self) -> str:
        return f"FlattenedYAMLConfig({len(self.source)} top-level keys)"


FLATTENED_YAML_CONFIG = FlattenedYAMLConfig.from_hydra()
