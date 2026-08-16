"""Tests for Hydra configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from grasping_ai.config.config import (
    compose_config,
    config_get,
    config_value,
    hydra_cfg_to_dict,
    load_project_yaml_config,
)


def test_load_project_yaml_config_merges_layers() -> None:
    """Verify that load_project_yaml_config merges all configuration layers correctly."""
    cfg = load_project_yaml_config(Path("configs"))
    assert cfg["device"] == "cpu"
    assert config_get(cfg, "architecture", "feature_dim") == 32
    assert config_get(cfg, "supervised", "batch_size") == 2
    assert config_get(cfg, "diffusion", "checkpoint") == ("artifacts/checkpoints/diffusion_grasp_generator.pt")
    assert config_get(cfg, "rl", "learning_rate") == 0.0003
    assert config_get(cfg, "rl", "observation_dim") == 31


def test_load_project_yaml_config_composes_evaluation_groups() -> None:
    """Verify that evaluation defaults compose common metrics and method settings."""
    cfg = load_project_yaml_config(Path("configs"))
    assert config_get(cfg, "metrics", "friction_coefficient") == 0.5
    assert config_get(cfg, "limits", "max_linear_velocity") == 0.05
    assert config_get(cfg, "evaluation", "method") == "diffusion"


def test_load_project_yaml_config_applies_evaluation_override() -> None:
    """Verify that evaluation=rl selects RL rollout settings."""
    cfg = load_project_yaml_config(Path("configs"), overrides=["evaluation=rl"])
    assert config_get(cfg, "evaluation", "method") == "rl"
    assert config_get(cfg, "evaluation", "episodes") == 5
    assert config_get(cfg, "metrics", "friction_coefficient") == 0.5


def test_load_project_yaml_config_applies_training_override() -> None:
    """Verify that training=flow selects the flow supervised config group."""
    cfg = load_project_yaml_config(
        Path("configs"),
        overrides=["model=flow", "training=flow"],
    )
    assert config_get(cfg, "default_method") == "flow"
    assert config_get(cfg, "supervised", "batch_size") == 2


def test_load_project_yaml_config_applies_hydra_overrides() -> None:
    """Verify that load_project_yaml_config applies Hydra overrides correctly."""
    cfg = load_project_yaml_config(
        Path("configs"),
        overrides=["seed=100"],
    )
    assert cfg["seed"] == 100


def test_config_value_path_and_list_helpers() -> None:
    """Verify typed config_value helpers retrieve paths and string lists."""
    cfg = load_project_yaml_config(Path("configs"))
    assert config_value(cfg, "paths", "dataset_root", value_type=Path) == Path("data/processed")
    assert config_value(cfg, "objects", "ids", value_type=list[str]) == [
        "003_cracker_box",
        "004_sugar_box",
        "006_mustard_bottle",
    ]


def test_load_project_yaml_config_skips_missing_layers(tmp_path: Path) -> None:
    """Compose only defaults that exist when ``config.yaml`` is present."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "base.yaml").write_text("seed: 7\n", encoding="utf-8")
    (config_dir / "config.yaml").write_text(
        "defaults:\n  - base\n  - _self_\n",
        encoding="utf-8",
    )
    cfg = load_project_yaml_config(config_dir)
    assert cfg["seed"] == 7


def test_load_project_yaml_config_uses_named_entrypoint() -> None:
    """Verify that config_name selects a full preset composition."""
    cfg = load_project_yaml_config(Path("configs"), config_name="training/flow")
    assert config_get(cfg, "default_method") == "flow"
    assert config_get(cfg, "supervised", "batch_size") == 2


def test_load_project_yaml_config_includes_notebook_settings() -> None:
    """Verify that notebook entrypoints compose shared notebook run settings."""
    cfg = load_project_yaml_config(Path("configs"), config_name="training/diffusion")
    assert config_get(cfg, "notebook", "experiment") == "diffusion_grasp_colab"
    assert config_get(cfg, "notebook", "download_ycb") is True
    assert config_get(cfg, "notebook", "augment") is False
    assert config_get(cfg, "notebook", "object_index") == 0
    assert config_get(cfg, "notebook", "mount_drive") is False
    assert config_get(cfg, "notebook", "drive_storage_dir") == "vr-rlft-side-project"


def test_load_project_yaml_config_invalid_name_raises() -> None:
    """Verify that an invalid config entrypoint name raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Hydra config entrypoint not found"):
        load_project_yaml_config(Path("configs"), config_name="nonexistent_config")


def test_config_get_returns_default_for_missing_path() -> None:
    """Return the configured default for absent nested keys."""
    cfg = OmegaConf.create({})
    assert config_get(cfg, "missing", default=42) == 42


def test_config_get_required_raises_for_missing_path() -> None:
    """Raise when a required nested key is absent."""
    cfg = OmegaConf.create({})
    with pytest.raises(ValueError, match="Missing config key"):
        config_get(cfg, "missing", required=True)


def test_config_value_path_returns_default_for_missing_value() -> None:
    """Return the default when a path-valued key is absent."""
    cfg = OmegaConf.create({})
    assert config_value(cfg, "missing", value_type=Path, default=Path("fallback")) == Path("fallback")


def test_config_value_path_returns_default_for_none_value() -> None:
    """Return the default when a path-valued key is explicitly ``None``."""
    cfg = OmegaConf.create({"paths": {"root": None}})
    assert config_value(cfg, "paths", "root", value_type=Path, default=Path("x")) == Path("x")


def test_config_value_path_rejects_empty_string() -> None:
    """Reject empty path strings in config mappings."""
    cfg = OmegaConf.create({"paths": {"root": ""}})
    with pytest.raises(ValueError, match="must be a non-empty string path"):
        config_value(cfg, "paths", "root", value_type=Path)


def test_config_value_path_rejects_non_string_value() -> None:
    """Reject non-string path values in config mappings."""
    cfg = OmegaConf.create({"paths": {"root": 1}})
    with pytest.raises(ValueError, match="must be a non-empty string path"):
        config_value(cfg, "paths", "root", value_type=Path)


def test_config_value_str_list_returns_default_for_missing_value() -> None:
    """Return the default when a string-list key is absent."""
    cfg = OmegaConf.create({})
    assert config_value(cfg, "objects", "ids", value_type=list[str], default=["a"]) == ["a"]


def test_config_value_str_list_rejects_non_string_items() -> None:
    """Reject string-list config values that contain non-strings."""
    cfg = OmegaConf.create({"objects": {"ids": ["a", 1]}})
    with pytest.raises(ValueError, match="must be a list of strings"):
        config_value(cfg, "objects", "ids", value_type=list[str])


def test_config_value_float_list_returns_default_for_missing_value() -> None:
    """Return the default when a float-list key is absent."""
    cfg = OmegaConf.create({})
    assert config_value(cfg, "gripper", "close_command", value_type=list[float], default=[0.0]) == [0.0]


def test_config_value_float_list_rejects_non_list() -> None:
    """Reject float-list config values that are not lists."""
    cfg = OmegaConf.create({"gripper": {"close_command": 1}})
    with pytest.raises(TypeError, match="must be a list of numbers"):
        config_value(cfg, "gripper", "close_command", value_type=list[float])


def test_config_value_float_list_rejects_bool_items() -> None:
    """Reject bool entries in float-list config values."""
    cfg = OmegaConf.create({"gripper": {"close_command": [True]}})
    with pytest.raises(TypeError, match="must be a list of numbers"):
        config_value(cfg, "gripper", "close_command", value_type=list[float])


def test_config_value_float_list_rejects_non_numeric_items() -> None:
    """Reject non-numeric entries in float-list config values."""
    cfg = OmegaConf.create({"gripper": {"close_command": ["bad"]}})
    with pytest.raises(TypeError, match="must be a list of numbers"):
        config_value(cfg, "gripper", "close_command", value_type=list[float])


def test_config_value_float_list_converts_integers() -> None:
    """Convert integer list entries to floats."""
    cfg = OmegaConf.create({"g": {"v": [0, 1]}})
    assert config_value(cfg, "g", "v", value_type=list[float]) == [0.0, 1.0]


def test_config_value_scalar_helpers() -> None:
    """Verify typed scalar config readers and validation errors."""
    empty = OmegaConf.create({})
    assert config_value(empty, "missing", value_type=float, default=0.5) == 0.5
    cfg = OmegaConf.create({"m": {"lr": 0.001}})
    assert config_value(cfg, "m", "lr", value_type=float, default=0.1) == pytest.approx(0.001)
    bad_cfg = OmegaConf.create({"m": {"lr": "bad"}})
    with pytest.raises(TypeError, match="must be a number"):
        config_value(bad_cfg, "m", "lr", value_type=float, default=0.1)
    bool_cfg = OmegaConf.create({"m": {"lr": True}})
    with pytest.raises(TypeError, match="must be a number"):
        config_value(bool_cfg, "m", "lr", value_type=float, default=0.1)

    assert config_value(empty, "missing", value_type=int, default=100) == 100
    int_cfg = OmegaConf.create({"d": {"steps": 64}})
    assert config_value(int_cfg, "d", "steps", value_type=int, default=1) == 64
    bad_int_cfg = OmegaConf.create({"d": {"steps": "bad"}})
    with pytest.raises(TypeError, match="must be an integer"):
        config_value(bad_int_cfg, "d", "steps", value_type=int, default=1)

    assert config_value(empty, "missing", value_type=bool, default=True) is True
    bool_cfg = OmegaConf.create({"f": {"enabled": False}})
    assert config_value(bool_cfg, "f", "enabled", value_type=bool, default=True) is False
    bad_bool_cfg = OmegaConf.create({"f": {"enabled": 1}})
    with pytest.raises(TypeError, match="must be a boolean"):
        config_value(bad_bool_cfg, "f", "enabled", value_type=bool, default=True)


def test_config_value_script_or_bool() -> None:
    """Verify script override resolution for booleans."""
    cfg = OmegaConf.create(
        {
            "script": {"multi_object": True},
            "evaluation": {"filter_collisions": False},
        }
    )
    assert config_value(cfg, "multi_object", value_type=bool, default=False, script_or=True) is True
    assert (
        config_value(
            cfg,
            "filter_collisions",
            "evaluation",
            "filter_collisions",
            value_type=bool,
            default=True,
            script_or=True,
        )
        is False
    )


def test_config_value_script_or_required_path() -> None:
    """Verify required script-or path resolution."""
    cfg = OmegaConf.create({"model": {"checkpoint": "artifacts/checkpoints/model.pt"}})
    resolved = config_value(
        cfg,
        "checkpoint",
        "model",
        "checkpoint",
        value_type=Path,
        script_or=True,
        required=True,
    )
    assert resolved == Path("artifacts/checkpoints/model.pt")


def test_hydra_cfg_to_dict_resolves_interpolation() -> None:
    """Verify hydra_cfg_to_dict resolves interpolated values."""
    cfg = load_project_yaml_config(Path("configs"))
    cfg_dict = hydra_cfg_to_dict(cfg)
    assert isinstance(cfg_dict, dict)
    assert cfg_dict["seed"] == 42


def test_compose_config_named_entrypoint() -> None:
    """Verify compose_config loads a named Hydra entrypoint."""
    cfg = compose_config(Path("configs"), config_name="training/flow")
    assert isinstance(cfg, DictConfig)
    assert config_get(cfg, "default_method") == "flow"
