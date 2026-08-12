import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: end-to-end / artifact-chain tests that exercise full pipelines "
        "and may take tens of seconds; skip with ``-m 'not slow'``.",
    )