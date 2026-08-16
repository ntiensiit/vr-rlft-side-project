"""Run artifact generation steps for CI and local workflows."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import hydra
import wandb
from loguru import logger
from omegaconf import DictConfig

from grasping_ai.config.config import (
    SCRIPTS_CONFIG_PATH,
    config_get,
    config_value,
    hydra_cfg_to_dict,
)
from grasping_ai.pipelines.evaluate import write_jsonl_records


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/run_artifacts")
def main(cfg: DictConfig) -> None:
    """Run reproducible supervised and RL artifact chains and write a manifest."""
    cfg_dict = hydra_cfg_to_dict(cfg)
    root = Path(__file__).resolve().parents[1]
    artifacts = config_value(cfg, "output_dir", "artifacts", "root", value_type=Path, script_or=True, required=True)
    if not artifacts.is_absolute():
        artifacts = root / artifacts

    env_vars = {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
        "PYTHONPYCACHEPREFIX": str(root / ".pycache"),
    }
    log: list[dict[str, object]] = []

    prepare_data_mode = str(
        config_value(cfg, "prepare_data_mode", "artifact_chain", "prepare_data_mode", value_type=object, script_or=True)
    )
    evaluate_multi_object = config_value(
        cfg,
        "evaluate_multi_object",
        "artifact_chain",
        "evaluate_multi_object",
        value_type=bool,
        default=True,
        script_or=True,
    )

    commands: list[list[str]] = [
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

    for cmd in commands:
        logger.info(">>> {}", " ".join(cmd))
        subprocess.run(cmd, cwd=root, env=env_vars, check=True, capture_output=False)
        log.append({"command": "python " + " ".join(cmd[1:]), "cwd": "."})

    data_processed = config_value(cfg, "paths", "dataset_root", value_type=Path, required=True)
    mjcf_root = config_value(cfg, "paths", "ycb_mjcf", value_type=Path, required=True)
    if not data_processed.is_absolute():
        data_processed = root / data_processed
    if not mjcf_root.is_absolute():
        mjcf_root = root / mjcf_root

    root_resolved = root.resolve()
    retained_artifacts = sorted(
        p.resolve().relative_to(root_resolved).as_posix()
        for p in (
            *data_processed.glob("*.npz"),
            *data_processed.glob("index.json"),
            *mjcf_root.rglob("*.xml"),
            *artifacts.rglob("*.pt"),
            *artifacts.rglob("*.npy"),
            *artifacts.rglob("*.jsonl"),
        )
    )
    manifest_path = artifacts / "manifest.jsonl"
    config_dir_arg = "configs"
    manifest_records: list[dict[str, object]] = [
        {
            "record_type": "manifest",
            "description": (
                "Reproducible artifact chain: supervised (YCB mesh -> synthetic dataset "
                "-> grasp checkpoint -> generated grasps -> MuJoCo simulation -> eval report) "
                "and RL (SB3 PPO -> legacy checkpoint -> policy_runner inference)"
            ),
            "config_dir": config_dir_arg,
        },
    ]
    manifest_records.extend({"record_type": "command", **entry} for entry in log)
    manifest_records.extend({"record_type": "retained_artifact", "path": rel} for rel in retained_artifacts)
    write_jsonl_records(manifest_path, manifest_records)

    tracking_backend = str(config_get(cfg, "tracking", "backend", default="none")).lower()
    if tracking_backend == "wandb":
        wandb_project = str(config_get(cfg, "tracking", "project", default="vr-rlft-side-project"))
        wandb_entity = config_get(cfg, "tracking", "entity", default=None)
        wandb_mode = str(config_get(cfg, "tracking", "mode", default="offline"))
        wandb_init: dict[str, object] = {
            "project": wandb_project,
            "job_type": "artifact-chain",
            "mode": wandb_mode,
            "config": {
                "config_dir": config_dir_arg,
                "artifact_count": len(retained_artifacts),
            },
            "tags": ["artifact-chain"],
        }
        if isinstance(wandb_entity, str) and wandb_entity:
            wandb_init["entity"] = wandb_entity
        wandb_run = wandb.init(**wandb_init)
        try:
            wandb_artifact = wandb.Artifact(
                name="artifact-chain",
                type="pipeline-output",
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

    observation_dim = config_value(cfg, "rl", "observation_dim", value_type=int)
    action_dim = config_value(cfg, "rl", "action_dim", value_type=int)
    rl_checkpoint = config_value(cfg, "rl", "checkpoint", value_type=Path, required=True)
    if not rl_checkpoint.is_absolute():
        rl_checkpoint = root / rl_checkpoint
    rl_checkpoint_arg = rl_checkpoint.resolve().relative_to(root_resolved).as_posix()
    infer_path_arg = (artifacts / "rl_inference_smoke.py").resolve().relative_to(root_resolved).as_posix()
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
    infer_path = artifacts / "rl_inference_smoke.py"
    infer_path.write_text(infer_script, encoding="utf-8")
    logger.info(">>> {} {}", sys.executable, infer_path)
    subprocess.run([sys.executable, str(infer_path)], cwd=root, env=env_vars, check=True)
    log.append({"command": f"python {infer_path_arg}", "cwd": "."})

    logger.info("All artifact-chain steps completed.")
    logger.info("Manifest written to artifacts/manifest.jsonl")


if __name__ == "__main__":
    main()
