#!/usr/bin/env python
"""Reproducible end-to-end experiment artifact generation.

Runs both retained artifact chains with small, CPU-friendly hyperparameters:

  1. Supervised chain: YCB mesh -> synthetic dataset -> trained grasp checkpoint
     -> generated grasps -> MuJoCo simulation -> evaluation report.
  2. RL chain: MuJoCo+YCB env -> SB3 PPO -> exported legacy policy checkpoint
     -> policy_runner inference smoke test.

All commands are logged to ``artifacts/manifest.json`` together with the
parameters used, so the retained artifacts can be reproduced from scratch.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
ENV_VARS = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

LOG: list[dict[str, object]] = []


def run(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=ENV_VARS, check=True, capture_output=False)
    LOG.append({"command": " ".join(cmd), "cwd": str(ROOT)})


def main() -> None:
    ycb_root = ROOT / "data" / "raw" / "ycb"
    mjcf_root = DATA_PROCESSED / "ycb_mjcf"
    observations = ROOT / "data" / "observations"
    mjcf_root.mkdir(parents=True, exist_ok=True)
    observations.mkdir(parents=True, exist_ok=True)

    # Step 0: YCB mesh -> MJCF wrappers (raw YCB uses OpenRAVE KinBody XML).
    run([sys.executable, "scripts/prepare_ycb_mjcf.py",
         "--ycb-root", str(ycb_root), "--output-root", str(mjcf_root)])

    # Step 1: YCB mesh -> synthetic grasp dataset + index. The three shipped
    # YCB objects are declared as required: any missing/failed object aborts
    # the chain rather than silently producing a partial dataset.
    run([sys.executable, "scripts/prepare_data.py",
         "--mode", "synthetic",
         "--ycb-root", str(ycb_root),
         "--dataset-root", str(DATA_PROCESSED),
         "--output-index", str(DATA_PROCESSED / "index.json"),
         "--num-samples", "256", "--num-grasps", "8",
         "--gripper-width", "0.08", "--seed", "42",
         "--required-objects", "003_cracker_box", "004_sugar_box",
         "006_mustard_bottle"])

    # Observations for inference/evaluation.
    run([sys.executable, "scripts/prepare_observations.py",
         "--ycb-root", str(ycb_root),
         "--output-dir", str(observations),
         "--num-samples", "256", "--seed", "7"])

    # Step 2: supervised training -> grasp checkpoint.
    run([sys.executable, "scripts/train.py",
         "--dataset-root", str(DATA_PROCESSED),
         "--checkpoint", str(ARTIFACTS / "checkpoints" / "grasp_generation.pt"),
         "--feature-dim", "32", "--hidden-dim", "32", "--num-layers", "2",
         "--learning-rate", "0.001", "--num-epochs", "3", "--batch-size", "2",
         "--device", "cpu", "--seed", "42",
         "--experiment-log-dir", str(ARTIFACTS / "exports" / "tensorboard" / "train")])

    # Step 3: checkpoint -> generated grasps.
    run([sys.executable, "scripts/generate_grasps.py",
         "--checkpoint", str(ARTIFACTS / "checkpoints" / "grasp_generation.pt"),
         "--observations",
         str(observations / "003_cracker_box.npy"),
         str(observations / "004_sugar_box.npy"),
         str(observations / "006_mustard_bottle.npy"),
         "--output", str(ARTIFACTS / "exports" / "generated_grasps.npy"),
         "--feature-dim", "32", "--num-diffusion-steps", "5", "--num-grasps", "8",
         "--device", "cpu"])

    # Step 4: generated grasps -> MuJoCo simulation outcomes.
    run([sys.executable, "scripts/extract_object_grasps.py",
         "--input", str(ARTIFACTS / "exports" / "generated_grasps.npy"),
         "--output", str(ARTIFACTS / "exports" / "grasp_poses_cracker.npy"),
         "--key", "object_0"])
    run([sys.executable, "scripts/run_simulation.py",
         "--grasps", str(ARTIFACTS / "exports" / "grasp_poses_cracker.npy"),
         "--object-id", "003_cracker_box",
         "--ycb-root", str(mjcf_root),
         "--robot-xml", str(ROOT / "deploy" / "robot.xml"),
         "--output", str(ARTIFACTS / "reports" / "simulation_cracker.json"),
         "--num-simulation-steps", "50", "--gripper-close-command", "0.02", "0.02"])

    # Step 5: evaluation report — same object identity (003_cracker_box) as the
    # simulation step, so the chain tracks one object end-to-end.
    run([sys.executable, "scripts/evaluate.py",
         "--grasps", str(ARTIFACTS / "exports" / "generated_grasps.npy"),
         "--object-id", "object_0",
         "--object-point-cloud", str(observations / "003_cracker_box.npy"),
         "--gripper-point-cloud", str(observations / "gripper.npy"),
         "--report", str(ARTIFACTS / "reports" / "evaluation_report.json"),
         "--friction-coefficient", "0.5", "--lift-height-threshold", "0.05",
         "--contact-clearance", "0.005", "--wrench-regularization", "1.0"])

    # Step 6: RL chain — SB3 PPO training + legacy checkpoint export.
    run([sys.executable, "scripts/train_rl.py",
         "--robot-xml", str(ROOT / "deploy" / "robot.xml"),
         "--ycb-root", str(mjcf_root),
         "--object-ids", "003_cracker_box",
         "--policy-checkpoint", str(ARTIFACTS / "checkpoints" / "rl_policy.pt"),
         "--observation-dim", "21", "--action-dim", "4", "--hidden-dim", "32",
         "--learning-rate", "0.0003", "--num-updates", "10", "--gamma", "0.99",
         "--device", "cpu", "--seed", "42",
         "--experiment-log-dir", str(ARTIFACTS / "exports" / "tensorboard" / "rl")])

    retained_artifacts = sorted(
        str(p.relative_to(ROOT)) for p in (
            *DATA_PROCESSED.glob("*.npy"),
            *DATA_PROCESSED.glob("index.json"),
            *mjcf_root.rglob("*.xml"),
            *ARTIFACTS.rglob("*.pt"),
            *ARTIFACTS.rglob("*.npy"),
            *ARTIFACTS.rglob("*.json"),
        )
    )
    manifest = {
        "description": (
            "Reproducible artifact chain: supervised (YCB mesh -> synthetic dataset "
            "-> grasp checkpoint -> generated grasps -> MuJoCo simulation -> eval report) "
            "and RL (SB3 PPO -> legacy checkpoint -> policy_runner inference)"
        ),
        "generated": LOG,
        "retained_artifacts": retained_artifacts,
    }
    (ARTIFACTS / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    # Step 7: policy_runner inference smoke test on the exported RL checkpoint.
    infer_script = (
        "from pathlib import Path\n"
        "import numpy as np\n"
        "from grasping_ai.inference.policy_runner import (\n"
        "    load_rl_policy_checkpoint, build_rl_policy_runner, run_policy_step,\n"
        ")\n"
        f"ckpt = load_rl_policy_checkpoint(\n"
        f"    Path(r'{(ARTIFACTS / 'checkpoints' / 'rl_policy.pt').as_posix()}'), 'cpu')\n"
        "runner = build_rl_policy_runner(ckpt, 21, 4, 'cpu')\n"
        "obs = np.zeros(21, dtype=np.float32)\n"
        "act = run_policy_step(runner, obs)\n"
        "print('policy inference OK', np.asarray(act).shape)\n"
    )
    infer_path = ARTIFACTS / "rl_inference_smoke.py"
    infer_path.write_text(infer_script, encoding="utf-8")
    run([sys.executable, str(infer_path)])

    print("All artifact-chain steps completed.")
    print("Manifest written to artifacts/manifest.json")


if __name__ == "__main__":
    main()
