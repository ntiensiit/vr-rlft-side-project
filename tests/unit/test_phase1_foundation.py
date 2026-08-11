import os

import numpy as np
import pytransform3d
import scipy  # type: ignore[import-untyped]

import grasping_ai


def test_package_imports():
    """Verify that grasping_ai is importable."""
    assert grasping_ai.__name__ == "grasping_ai"


def test_math_dependencies_available():
    """Verify that core math libraries can be imported."""
    assert pytransform3d.__version__ is not None
    assert scipy.__version__ is not None
    assert np.__version__ is not None


def test_base_config_file_exists():
    """Verify that configs/base.yaml exists."""
    config_path = os.path.join("configs", "base.yaml")
    assert os.path.isfile(config_path)


def test_base_config_contains_contract_keys():
    """Verify base.yaml contains the required contract keys by plain text."""
    config_path = os.path.join("configs", "base.yaml")
    with open(config_path, encoding="utf-8") as f:
        content = f.read()
    # Expect keys in YAML
    assert "seed:" in content or "random_seed:" in content
    assert "device:" in content
    assert "output_dir:" in content


def test_pyproject_preserves_src_package_layout():
    """Verify pyproject.toml preserves src package layout."""
    with open("pyproject.toml", encoding="utf-8") as f:
        content = f.read()
    assert 'packages = ["src"]' in content or "packages = ['src']" in content or '"src"' in content
