from pathlib import Path

import pytest

from grasping_ai.config.yaml_loader import (
    config_float_list,
    config_get,
    config_path,
    config_str_list,
    load_project_yaml_config,
    load_yaml_mapping,
    merge_yaml_mappings,
    optional_cli_path,
    parse_config_dir_from_argv,
    require_config_value,
)


def test_load_yaml_mapping_reads_base_config() -> None:
    """Verify that load_yaml_mapping correctly parses the default base.yaml configuration."""
    cfg = load_yaml_mapping(Path("configs/base.yaml"))
    assert cfg["device"] == "cpu"
    assert cfg["seed"] == 42
    assert config_get(cfg, "paths", "output_dir") == '${oc.env:MGS_OUTPUT_DIR,"artifacts"}'


def test_load_project_yaml_config_merges_layers() -> None:
    """Verify that load_project_yaml_config merges all configuration layers correctly."""
    cfg = load_project_yaml_config(Path("configs"))
    assert cfg["device"] == "cpu"
    assert config_get(cfg, "architecture", "feature_dim") == 32
    assert config_get(cfg, "supervised", "batch_size") == 2
    assert config_get(cfg, "diffusion", "checkpoint") == ("artifacts/checkpoints/diffusion_grasp_generator.pt")
    assert config_get(cfg, "rl", "learning_rate") == 0.0003
    assert config_get(cfg, "rl", "observation_dim") == 31


def test_load_project_yaml_config_applies_hydra_overrides() -> None:
    """Verify that load_project_yaml_config applies command-line Hydra overrides correctly."""
    cfg = load_project_yaml_config(
        Path("configs"),
        overrides=["seed=100"],
    )
    assert cfg["seed"] == 100


def test_merge_yaml_mappings_deep_merges_nested_objects() -> None:
    """Verify that merge_yaml_mappings recursively deep-merges nested dictionaries."""
    merged = merge_yaml_mappings(
        {"rl": {"checkpoint": "a.pt", "observation_dim": 31}},
        {"rl": {"learning_rate": 0.001}},
    )
    assert merged == {
        "rl": {
            "checkpoint": "a.pt",
            "observation_dim": 31,
            "learning_rate": 0.001,
        }
    }


def test_merge_yaml_mappings_overrides_scalar_keys() -> None:
    """Verify that merge_yaml_mappings overrides scalar keys left-to-right."""
    merged = merge_yaml_mappings({"a": 1, "b": 2}, {"b": 3, "c": 4})
    assert merged == {"a": 1, "b": 3, "c": 4}


def test_config_path_and_list_helpers() -> None:
    """Verify that path and list utility functions retrieve typed configuration values."""
    cfg = load_project_yaml_config(Path("configs"))
    assert config_path(cfg, "paths", "dataset_root") == Path("data/processed")
    assert config_str_list(cfg, "objects", "ids") == [
        "003_cracker_box",
        "004_sugar_box",
        "006_mustard_bottle",
    ]


def test_require_config_value_raises_for_missing_key() -> None:
    """Verify that require_config_value raises a ValueError when a required key is missing."""
    with pytest.raises(ValueError, match="missing"):
        require_config_value({}, "missing", "key")


def test_load_yaml_mapping_validates_path_type(tmp_path: Path) -> None:
    """Reject non-``Path`` inputs to ``load_yaml_mapping``."""
    with pytest.raises(TypeError, match=r"path must be a pathlib\.Path"):
        load_yaml_mapping("configs/base.yaml")  # type: ignore[arg-type]


def test_load_yaml_mapping_missing_file(tmp_path: Path) -> None:
    """Raise when the requested YAML file does not exist."""
    with pytest.raises(FileNotFoundError, match="YAML config file not found"):
        load_yaml_mapping(tmp_path / "missing.yaml")


def test_load_yaml_mapping_empty_file(tmp_path: Path) -> None:
    """Treat an empty YAML file as an empty mapping."""
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_yaml_mapping(path) == {}


def test_load_yaml_mapping_rejects_non_mapping_root(tmp_path: Path) -> None:
    """Reject YAML files whose root value is not a mapping."""
    path = tmp_path / "list.yaml"
    path.write_text("- a\n", encoding="utf-8")
    with pytest.raises(TypeError, match="must be a mapping"):
        load_yaml_mapping(path)


def test_merge_yaml_mappings_rejects_non_dict() -> None:
    """Reject non-mapping operands passed to ``merge_yaml_mappings``."""
    with pytest.raises(TypeError, match="each mapping must be a dict"):
        merge_yaml_mappings({"a": 1}, "not-a-mapping")  # type: ignore[arg-type]


def test_load_project_yaml_config_validates_config_dir() -> None:
    """Reject non-``Path`` config directories."""
    with pytest.raises(TypeError, match=r"config_dir must be a pathlib\.Path"):
        load_project_yaml_config("configs", "base")  # type: ignore[arg-type]


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


def test_parse_config_dir_from_argv_defaults_to_configs() -> None:
    """Default ``--config-dir`` to ``configs`` when argv is empty."""
    assert parse_config_dir_from_argv([]) == Path("configs")


def test_parse_config_dir_from_argv_reads_flag() -> None:
    """Parse an explicit ``--config-dir`` flag from argv."""
    assert parse_config_dir_from_argv(["--config-dir", "custom"]) == Path("custom")


def test_config_get_returns_default_for_missing_path() -> None:
    """Return the configured default for absent nested keys."""
    assert config_get({}, "missing", default=42) == 42


def test_config_get_required_raises_for_missing_path() -> None:
    """Raise when a required nested key is absent."""
    with pytest.raises(ValueError, match="Missing config key"):
        config_get({}, "missing", required=True)


def test_config_get_raises_when_traversing_non_mapping() -> None:
    """Raise when a config path traverses a non-mapping value."""
    with pytest.raises(TypeError, match="traverses a non-mapping"):
        config_get({"a": 1}, "a", "b")


def test_config_path_returns_default_for_missing_value() -> None:
    """Return the default when a path-valued key is absent."""
    assert config_path({}, "missing", default=Path("fallback")) == Path("fallback")


def test_config_path_returns_default_for_none_value() -> None:
    """Return the default when a path-valued key is explicitly ``None``."""
    assert config_path({"paths": {"root": None}}, "paths", "root", default=Path("x")) == Path("x")


def test_config_path_rejects_empty_string() -> None:
    """Reject empty path strings in config mappings."""
    with pytest.raises(ValueError, match="must be a non-empty string path"):
        config_path({"paths": {"root": ""}}, "paths", "root")


def test_config_path_rejects_non_string_value() -> None:
    """Reject non-string path values in config mappings."""
    with pytest.raises(ValueError, match="must be a non-empty string path"):
        config_path({"paths": {"root": 1}}, "paths", "root")


def test_config_str_list_returns_default_for_missing_value() -> None:
    """Return the default when a string-list key is absent."""
    assert config_str_list({}, "objects", "ids", default=["a"]) == ["a"]


def test_config_str_list_rejects_non_string_items() -> None:
    """Reject string-list config values that contain non-strings."""
    with pytest.raises(ValueError, match="must be a list of strings"):
        config_str_list({"objects": {"ids": ["a", 1]}}, "objects", "ids")


def test_config_float_list_returns_default_for_missing_value() -> None:
    """Return the default when a float-list key is absent."""
    assert config_float_list({}, "gripper", "close_command", default=[0.0]) == [0.0]


def test_config_float_list_rejects_non_list() -> None:
    """Reject float-list config values that are not lists."""
    with pytest.raises(TypeError, match="must be a list of numbers"):
        config_float_list({"gripper": {"close_command": 1}}, "gripper", "close_command")


def test_config_float_list_rejects_bool_items() -> None:
    """Reject bool entries in float-list config values."""
    with pytest.raises(TypeError, match="must be a list of numbers"):
        config_float_list({"gripper": {"close_command": [True]}}, "gripper", "close_command")


def test_config_float_list_rejects_non_numeric_items() -> None:
    """Reject non-numeric entries in float-list config values."""
    with pytest.raises(TypeError, match="must be a list of numbers"):
        config_float_list({"gripper": {"close_command": ["bad"]}}, "gripper", "close_command")


def test_config_float_list_converts_integers() -> None:
    """Convert integer list entries to floats."""
    assert config_float_list({"g": {"v": [0, 1]}}, "g", "v") == [0.0, 1.0]


def test_optional_cli_path_treats_none_literal_as_absent() -> None:
    """Treat CLI ``none`` literals as absent optional paths."""
    assert optional_cli_path("None") is None
    assert optional_cli_path("none") is None
    assert optional_cli_path("") is None
    assert optional_cli_path(" deploy/table.xml ") == Path("deploy/table.xml")
