"""Structured logging helpers."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import mlflow
from loguru import logger

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG

LOG_ROTATION = str(FLATTENED_YAML_CONFIG.get("logging.rotation", "10 MB"))
LOG_RETENTION = str(FLATTENED_YAML_CONFIG.get("logging.retention", "10 days"))


def setup_logging(module_name: str | None = None, level: str = "INFO") -> None:
    """Configure loguru logging to stderr and optionally to a log file.

    Args:
        module_name: Optional name of the module/script to format log file path.
        level: Minimum log level to display.
    """
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    logger.add(sys.stderr, format=log_format, level=level)

    if module_name:
        current_date = datetime.now(tz=UTC).date().isoformat()
        log_file_path = Path("logs") / f"{current_date}-{module_name}.log"
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file_path),
            rotation=LOG_ROTATION,
            retention=LOG_RETENTION,
            format=log_format,
            level=level,
        )

    logger.info("Logging configured with level: {}", level)


def init_mlflow(config: dict) -> bool:
    """Initialize MLflow tracking from the project configuration.

    Args:
        config: The project configuration dictionary.

    Returns:
        True if MLflow was successfully initialized, False otherwise.
    """
    tracking_cfg = config.get("tracking", {})
    backend = tracking_cfg.get("backend", "none")

    if backend != "mlflow":
        return False

    mlflow_cfg = tracking_cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "./mlruns")
    experiment_name = mlflow_cfg.get("experiment_name", "default")

    logger.info("Initializing MLflow with URI: {} and Experiment: {}", tracking_uri, experiment_name)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return True
