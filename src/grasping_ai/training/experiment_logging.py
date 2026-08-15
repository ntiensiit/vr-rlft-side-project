from __future__ import annotations


def try_log_mlflow_param(key: str, value: str) -> None:
    """Log a single MLflow parameter when an active run exists.

    Args:
        key: Parameter name.
        value: Serialized parameter value.
    """
    try:
        import mlflow  # noqa: PLC0415

        if mlflow.active_run():
            mlflow.log_param(key, value)
    except ImportError:
        pass


def try_log_mlflow_metric(key: str, value: float, step: int) -> None:
    """Log a single MLflow metric when an active run exists.

    Args:
        key: Metric name.
        value: Metric value.
        step: Training step index.
    """
    try:
        import mlflow  # noqa: PLC0415

        if mlflow.active_run():
            mlflow.log_metric(key, value, step=step)
    except ImportError:
        pass


def try_log_mlflow_artifact(path: str) -> None:
    """Log a local artifact path to MLflow when an active run exists.

    Args:
        path: Filesystem path to the artifact.
    """
    try:
        import mlflow  # noqa: PLC0415

        if mlflow.active_run():
            mlflow.log_artifact(path)
    except ImportError:
        pass
