"""Run artifact generation steps for CI and local workflows."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import wandb
from loguru import logger

from grasping_ai.config import SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.pipelines.evaluate import write_jsonl_records

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _resolve_artifacts_root(yaml_config: FlattenedYAMLConfig, root: Path) -> Path:
    """Resolve the artifact output directory relative to the repo root."""
    artifacts = yaml_config.value("output_dir", "artifacts", "root", value_type=Path, script_or=True, required=True)
    if not artifacts.is_absolute():
        artifacts = root / artifacts
    return artifacts


def _build_env_vars(yaml_config: FlattenedYAMLConfig, root: Path) -> dict[str, str]:
    """Build the subprocess environment with ``src`` on ``PYTHONPATH``."""
    return {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
        "PYTHONPYCACHEPREFIX": str(root / yaml_config.value("artifacts", "pycache_dir", value_type=object)),
    }


def _build_command_chain(yaml_config: FlattenedYAMLConfig) -> list[list[str]]:
    """Build the ordered artifact-chain command list from the configuration."""
    prepare_data_mode = str(
        yaml_config.value(
            "prepare_data_mode", "artifact_chain", "prepare_data_mode", value_type=object, script_or=True,
        ),
    )
    evaluate_multi_object = yaml_config.value(
        "evaluate_multi_object",
        "artifact_chain",
        "evaluate_multi_object",
        value_type=bool,
        default=True,
        script_or=True,
    )

    return [
        [sys.executable, "scripts/prepare_ycb_mjcf.py"],
        [sys.executable, "scripts/prepare_data.py", f"prepare.mode={prepare_data_mode}"],
        [sys.executable, "scripts/prepare_observations.py"],
        [sys.executable, "scripts/train_diffusion.py"],
        [sys.executable, "scripts/generate_grasps.py"],
        [sys.executable, "scripts/extract_object_grasps.py"],
        [sys.executable, "scripts/run_simulation.py"],
        [
            sys.executable,
            "scripts/evaluate.py",
            f"script.multi_object={'true' if evaluate_multi_object else 'false'}",
        ],
        [sys.executable, "scripts/train_rl.py"],
    ]


def _run_command_chain(
    commands: list[list[str]],
    root: Path,
    env_vars: dict[str, str],
    log: list[dict[str, object]],
) -> None:
    """Run each artifact-chain command in order, appending to ``log``."""
    for cmd in commands:
        logger.info(">>> {}", " ".join(cmd))
        subprocess.run(  # noqa: S603  # fixed internal command list, no shell
            cmd, cwd=root, env=env_vars, check=True, capture_output=False,
        )
        log.append({"command": "python " + " ".join(cmd[1:]), "cwd": "."})


def _collect_retained_artifacts(yaml_config: FlattenedYAMLConfig, root: Path, artifacts: Path) -> list[str]:
    """Collect repo-relative paths of retained dataset, MJCF, and artifact files."""
    data_processed = yaml_config.value("paths", "dataset_root", value_type=Path, required=True)
    mjcf_root = yaml_config.value("paths", "ycb_mjcf", value_type=Path, required=True)
    if not data_processed.is_absolute():
        data_processed = root / data_processed
    if not mjcf_root.is_absolute():
        mjcf_root = root / mjcf_root

    root_resolved = root.resolve()
    retained_cfg = yaml_config.get_path("artifacts", "retained")
    if not isinstance(retained_cfg, dict):
        msg = "artifacts.retained must be a mapping"
        raise TypeError(msg)
    dataset_globs = retained_cfg.get("dataset_globs", [])
    mjcf_glob = str(retained_cfg.get("mjcf_glob", "**/*.xml"))
    artifact_globs = retained_cfg.get("artifact_globs", [])
    retained_paths: list[Path] = []
    if isinstance(dataset_globs, list):
        for pattern in dataset_globs:
            retained_paths.extend(data_processed.glob(str(pattern)))
    retained_paths.extend(mjcf_root.rglob(mjcf_glob))
    if isinstance(artifact_globs, list):
        for pattern in artifact_globs:
            retained_paths.extend(artifacts.rglob(str(pattern)))
    return sorted(
        p.resolve().relative_to(root_resolved).as_posix()
        for p in retained_paths
    )


def _write_manifest(
    yaml_config: FlattenedYAMLConfig,
    artifacts: Path,
    retained_artifacts: list[str],
    log: list[dict[str, object]],
) -> tuple[Path, list[dict[str, object]]]:
    """Write the artifact-chain manifest and return its path and records."""
    manifest_cfg = yaml_config.value("artifacts", "manifest", value_type=Path, required=True)
    manifest_path = manifest_cfg if manifest_cfg.is_absolute() else artifacts / manifest_cfg.name
    config_dir_arg = str(yaml_config.value("artifacts", "config_dir", value_type=object, required=True))
    chain_cfg = yaml_config.get_path("artifacts", "chain")
    if not isinstance(chain_cfg, dict):
        msg = "artifacts.chain must be a mapping"
        raise TypeError(msg)
    manifest_records: list[dict[str, object]] = [
        {
            "record_type": "manifest",
            "description": str(chain_cfg.get("description", "")),
            "config_dir": config_dir_arg,
        },
    ]
    manifest_records.extend({"record_type": "command", **entry} for entry in log)
    manifest_records.extend({"record_type": "retained_artifact", "path": rel} for rel in retained_artifacts)
    write_jsonl_records(manifest_path, manifest_records)
    return manifest_path, manifest_records


def _publish_wandb_artifact(
    yaml_config: FlattenedYAMLConfig,
    root: Path,
    manifest_path: Path,
    manifest_records: list[dict[str, object]],
    retained_artifacts: list[str],
) -> None:
    """Publish the artifact chain to W&B when ``tracking.backend`` is ``wandb``."""
    tracking_backend = str(yaml_config.get_path("tracking", "backend", default="none")).lower()
    if tracking_backend != "wandb":
        return
    config_dir_arg = str(yaml_config.value("artifacts", "config_dir", value_type=object, required=True))
    chain_cfg = yaml_config.get_path("artifacts", "chain")
    if not isinstance(chain_cfg, dict):
        msg = "artifacts.chain must be a mapping"
        raise TypeError(msg)
    wandb_cfg = chain_cfg.get("wandb", {})
    if not isinstance(wandb_cfg, dict):
        wandb_cfg = {}
    root_resolved = root.resolve()
    wandb_project = str(yaml_config.get_path("tracking", "project", default="vr-rlft-side-project"))
    wandb_entity = yaml_config.get_path("tracking", "entity", default=None)
    wandb_mode = str(yaml_config.get_path("tracking", "mode", default="offline"))
    wandb_init: dict[str, object] = {
        "project": wandb_project,
        "job_type": str(wandb_cfg.get("job_type", "artifact-chain")),
        "mode": wandb_mode,
        "config": {
            "config_dir": config_dir_arg,
            "artifact_count": len(retained_artifacts),
        },
        "tags": list(wandb_cfg.get("tags", ["artifact-chain"])),
    }
    if isinstance(wandb_entity, str) and wandb_entity:
        wandb_init["entity"] = wandb_entity
    wandb_run = wandb.init(**wandb_init)
    try:
        wandb_artifact = wandb.Artifact(
            name=str(wandb_cfg.get("artifact_name", "artifact-chain")),
            type=str(wandb_cfg.get("artifact_type", "pipeline-output")),
            metadata={"config_dir": config_dir_arg},
        )
        manifest_rel = manifest_path.resolve().relative_to(root_resolved).as_posix()
        wandb_artifact.add_file(str(manifest_path), name=manifest_rel)
        for rel in retained_artifacts:
            artifact_file = root / rel
            if artifact_file.is_file():
                wandb_artifact.add_file(str(artifact_file), name=rel)
        wandb_run.log_artifact(wandb_artifact)
        if wandb_mode == "offline":
            artifact_digest = getattr(wandb_artifact, "digest", None)
            artifact_version = str(artifact_digest) if artifact_digest else "offline"
        else:
            wandb_artifact.wait()
            artifact_version = str(wandb_artifact.version)
        manifest_records.append(
            {
                "record_type": "wandb_tracking",
                "run_id": wandb_run.id,
                "artifact_version": artifact_version,
                "project": wandb_project,
            },
        )
        write_jsonl_records(manifest_path, manifest_records)
        logger.info(
            "W&B artifact chain published: run_id={} version={}",
            wandb_run.id,
            artifact_version,
        )
    finally:
        wandb_run.finish()


def _run_rl_inference_smoke(
    yaml_config: FlattenedYAMLConfig,
    root: Path,
    artifacts: Path,
    env_vars: dict[str, str],
    log: list[dict[str, object]],
) -> None:
    """Write and execute the RL policy inference smoke-check script."""
    observation_dim = yaml_config.value("rl", "observation_dim", value_type=int)
    action_dim = yaml_config.value("rl", "action_dim", value_type=int)
    rl_checkpoint = yaml_config.value("rl", "checkpoint", value_type=Path, required=True)
    if not rl_checkpoint.is_absolute():
        rl_checkpoint = root / rl_checkpoint
    root_resolved = root.resolve()
    rl_checkpoint_arg = rl_checkpoint.resolve().relative_to(root_resolved).as_posix()
    infer_cfg = yaml_config.value("artifacts", "rl_inference_smoke", value_type=Path, required=True)
    infer_path = infer_cfg if infer_cfg.is_absolute() else artifacts / infer_cfg.name
    infer_path_arg = infer_path.resolve().relative_to(root_resolved).as_posix()
    infer_script = (
        "from pathlib import Path\n"
        "import numpy as np\n"
        "from grasping_ai.inference.policy_runner import (\n"
        "    load_rl_policy_checkpoint, build_rl_policy_runner, run_policy_step,\n"
        ")\n"
        f"ckpt = load_rl_policy_checkpoint(\n"
        f"    Path('{rl_checkpoint_arg}'), 'cpu')\n"
        f"runner = build_rl_policy_runner(ckpt, {observation_dim}, {action_dim}, 'cpu')\n"
        f"obs = np.zeros({observation_dim}, dtype=np.float32)\n"
        "act = run_policy_step(runner, obs)\n"
        "print('policy inference OK', np.asarray(act).shape)\n"
    )
    infer_path.write_text(infer_script, encoding="utf-8")
    logger.info(">>> {} {}", sys.executable, infer_path)
    subprocess.run(  # noqa: S603  # generated internal smoke script, no shell
        [sys.executable, str(infer_path)], cwd=root, env=env_vars, check=True,
    )
    log.append({"command": f"python {infer_path_arg}", "cwd": "."})


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/run_artifacts")
def main(cfg: DictConfig) -> None:
    """Run reproducible supervised and RL artifact chains and write a manifest."""
    yaml_config = FlattenedYAMLConfig(cfg)
    root = Path(__file__).resolve().parents[1]
    artifacts = _resolve_artifacts_root(yaml_config, root)
    env_vars = _build_env_vars(yaml_config, root)
    log: list[dict[str, object]] = []

    commands = _build_command_chain(yaml_config)
    _run_command_chain(commands, root, env_vars, log)

    retained_artifacts = _collect_retained_artifacts(yaml_config, root, artifacts)
    manifest_path, manifest_records = _write_manifest(yaml_config, artifacts, retained_artifacts, log)
    _publish_wandb_artifact(yaml_config, root, manifest_path, manifest_records, retained_artifacts)
    _run_rl_inference_smoke(yaml_config, root, artifacts, env_vars, log)

    logger.info("All artifact-chain steps completed.")
    logger.info("Manifest written to {}", manifest_path)


if __name__ == "__main__":
    main()
