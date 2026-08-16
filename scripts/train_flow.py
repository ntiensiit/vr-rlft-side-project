"""Train flow-matching grasp models from the command line."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from grasping_ai.config import SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig

try:
    from scripts._supervised_training import run_supervised_training_script
except ModuleNotFoundError:
    from _supervised_training import run_supervised_training_script
from grasping_ai.pipelines.train_flow import run_flow_training_pipeline

_LEGACY_OVERRIDES: dict[str, object] = {}


def _normalize_legacy_flags() -> None:
    """Translate legacy flags into Hydra overrides before composition."""
    mapping = {
        "--dataset-root": "paths.dataset_root",
        "--checkpoint": "model.checkpoint",
        "--feature-dim": "architecture.feature_dim",
        "--hidden-dim": "architecture.hidden_dim",
        "--num-layers": "architecture.num_layers",
        "--learning-rate": "supervised.learning_rate",
        "--num-epochs": "supervised.num_epochs",
        "--batch-size": "supervised.batch_size",
        "--device": "device",
        "--seed": "seed",
    }
    argv = sys.argv[1:]
    index = 0
    while index < len(argv):
        flag = argv[index]
        if flag not in mapping:
            raise ValueError(f"Unsupported train_flow argument: {flag}")
        if index + 1 >= len(argv):
            raise ValueError(f"Missing value for train_flow argument: {flag}")
        raw_value = argv[index + 1]
        if flag in {"--feature-dim", "--hidden-dim", "--num-layers", "--num-epochs", "--batch-size", "--seed"}:
            value: str | int | float = int(raw_value)
        elif flag == "--learning-rate":
            value = float(raw_value)
        else:
            value = raw_value
        _LEGACY_OVERRIDES[mapping[flag]] = value
        index += 2
    sys.argv[1:] = []


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="training/flow")
def _hydra_main(cfg: object) -> None:
    """Train the flow model from the project's Hydra configuration."""
    for key, value in _LEGACY_OVERRIDES.items():
        OmegaConf.update(cfg, key, value, merge=True)
    yaml_config = FlattenedYAMLConfig(cfg)
    run_supervised_training_script(
        yaml_config,
        module_name="train_flow",
        experiment_log_dir=yaml_config.value("flow", "tensorboard", value_type=Path),
        mlflow_run_name="flow_training",
        pipeline_fn=run_flow_training_pipeline,
    )


def main() -> None:
    """Run the Hydra entrypoint, retaining compatibility with legacy flags."""
    if "--dataset-root" in sys.argv:
        _normalize_legacy_flags()
    _hydra_main()


if __name__ == "__main__":
    main()
