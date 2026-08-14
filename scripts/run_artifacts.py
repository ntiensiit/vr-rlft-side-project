import os
import subprocess
import sys
from pathlib import Path

from grasping_ai.config.yaml_loader import (
    config_float_list,
    config_get,
    config_str_list,
    load_project_yaml_config,
    parse_config_dir_from_argv,
)
from grasping_ai.pipelines.evaluate import write_jsonl_records


def main() -> None:
    """Run reproducible supervised and RL artifact chains and write a manifest.

    Executes the full pipeline from YCB MJCF preparation through diffusion
    training, grasp generation, simulation, evaluation, and RL training.
    Records repo-relative commands in ``artifacts/manifest.jsonl``.
    """
    root = Path(__file__).resolve().parents[1]
    config_dir = parse_config_dir_from_argv()
    if not config_dir.is_absolute():
        config_dir = root / config_dir
    cfg = load_project_yaml_config(
        config_dir, "base", "data", "model", "training", "evaluation", "robot", "simulation"
    )
    artifacts = root / str(config_get(cfg, "paths", "output_dir"))
    data_processed = root / str(config_get(cfg, "paths", "dataset_root"))
    ycb_root = root / str(config_get(cfg, "paths", "ycb_root"))
    mjcf_root = root / str(config_get(cfg, "paths", "ycb_mjcf"))
    observations = root / str(config_get(cfg, "paths", "observations"))
    object_ids = config_str_list(cfg, "objects", "ids") or []
    gripper_close = config_float_list(cfg, "robot", "gripper", "close_command") or [0.0]
    env_vars = {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
        "PYTHONPYCACHEPREFIX": str(root / ".pycache"),
    }
    log: list[dict[str, object]] = []

    mjcf_root.mkdir(parents=True, exist_ok=True)
    observations.mkdir(parents=True, exist_ok=True)

    diffusion_checkpoint = root / str(config_get(cfg, "diffusion", "checkpoint"))
    rl_checkpoint = root / str(config_get(cfg, "rl", "checkpoint"))
    grasp_candidates = root / str(
        config_get(cfg, "diffusion", "exports", "grasp_candidates")
    )
    grasp_poses = root / str(config_get(cfg, "diffusion", "exports", "grasp_poses"))
    simulation_report = root / str(
        config_get(cfg, "diffusion", "exports", "simulation_report")
    )
    evaluation_report = root / str(
        config_get(cfg, "diffusion", "exports", "evaluation_report")
    )
    robot_xml = root / str(config_get(cfg, "robot", "description"))
    diffusion_tb = root / str(config_get(cfg, "diffusion", "tensorboard"))
    rl_tb = root / str(config_get(cfg, "rl", "tensorboard"))
    output_index = root / str(config_get(cfg, "paths", "output_index"))

    root_resolved = root.resolve()
    config_dir_arg = config_dir.resolve().relative_to(root_resolved).as_posix()
    ycb_root_arg = ycb_root.resolve().relative_to(root_resolved).as_posix()
    mjcf_root_arg = mjcf_root.resolve().relative_to(root_resolved).as_posix()
    observations_arg = observations.resolve().relative_to(root_resolved).as_posix()
    data_processed_arg = data_processed.resolve().relative_to(root_resolved).as_posix()
    output_index_arg = output_index.resolve().relative_to(root_resolved).as_posix()
    diffusion_checkpoint_arg = (
        diffusion_checkpoint.resolve().relative_to(root_resolved).as_posix()
    )
    grasp_candidates_arg = grasp_candidates.resolve().relative_to(root_resolved).as_posix()
    grasp_poses_arg = grasp_poses.resolve().relative_to(root_resolved).as_posix()
    simulation_report_arg = simulation_report.resolve().relative_to(root_resolved).as_posix()
    evaluation_report_arg = evaluation_report.resolve().relative_to(root_resolved).as_posix()
    robot_xml_arg = robot_xml.resolve().relative_to(root_resolved).as_posix()
    diffusion_tb_arg = diffusion_tb.resolve().relative_to(root_resolved).as_posix()
    rl_checkpoint_arg = rl_checkpoint.resolve().relative_to(root_resolved).as_posix()
    rl_tb_arg = rl_tb.resolve().relative_to(root_resolved).as_posix()
    observation_files = [
        (observations / f"{object_id}.npy").resolve().relative_to(root_resolved).as_posix()
        for object_id in object_ids
    ]

    commands: list[list[str]] = [
        [
            sys.executable,
            "scripts/prepare_ycb_mjcf.py",
            "--ycb-root",
            ycb_root_arg,
            "--output-root",
            mjcf_root_arg,
        ],
        [
            sys.executable,
            "scripts/prepare_data.py",
            "--config-dir",
            config_dir_arg,
            "--mode",
            "synthetic",
            "--ycb-root",
            ycb_root_arg,
            "--dataset-root",
            data_processed_arg,
            "--output-index",
            output_index_arg,
            "--num-samples",
            str(config_get(cfg, "synthetic", "num_samples")),
            "--num-grasps",
            str(config_get(cfg, "synthetic", "num_grasps")),
            "--gripper-width",
            str(config_get(cfg, "synthetic", "gripper_width")),
            "--seed",
            str(config_get(cfg, "synthetic", "seed")),
            "--required-objects",
            *object_ids,
        ],
        [
            sys.executable,
            "scripts/prepare_observations.py",
            "--ycb-root",
            ycb_root_arg,
            "--output-dir",
            observations_arg,
            "--num-samples",
            str(config_get(cfg, "observations", "num_samples")),
            "--seed",
            str(config_get(cfg, "observations", "seed")),
        ],
        [
            sys.executable,
            "scripts/train_diffusion.py",
            "--config-dir",
            config_dir_arg,
            "--dataset-root",
            data_processed_arg,
            "--checkpoint",
            diffusion_checkpoint_arg,
            "--experiment-log-dir",
            diffusion_tb_arg,
        ],
        [
            sys.executable,
            "scripts/generate_grasps.py",
            "--config-dir",
            config_dir_arg,
            "--checkpoint",
            diffusion_checkpoint_arg,
            "--observations",
            *observation_files,
            "--output",
            grasp_candidates_arg,
        ],
        [
            sys.executable,
            "scripts/extract_object_grasps.py",
            "--input",
            grasp_candidates_arg,
            "--output",
            grasp_poses_arg,
            "--key",
            "object_0",
        ],
        [
            sys.executable,
            "scripts/run_simulation.py",
            "--config-dir",
            config_dir_arg,
            "--grasps",
            grasp_poses_arg,
            "--object-id",
            object_ids[0],
            "--ycb-root",
            mjcf_root_arg,
            "--robot-xml",
            robot_xml_arg,
            "--output",
            simulation_report_arg,
            "--gripper-close-command",
            *[str(value) for value in gripper_close],
        ],
        [
            sys.executable,
            "scripts/evaluate.py",
            "--config-dir",
            config_dir_arg,
            "--multi-object",
            "--grasps",
            grasp_candidates_arg,
            "--observations-dir",
            observations_arg,
            "--gripper-point-cloud",
            f"{observations_arg}/gripper.npy",
            "--report",
            evaluation_report_arg,
        ],
        [
            sys.executable,
            "scripts/train_rl.py",
            "--config-dir",
            config_dir_arg,
            "--robot-xml",
            robot_xml_arg,
            "--ycb-root",
            mjcf_root_arg,
            "--object-ids",
            object_ids[0],
            "--policy-checkpoint",
            rl_checkpoint_arg,
            "--experiment-log-dir",
            rl_tb_arg,
        ],
    ]

    for cmd in commands:
        print(">>>", " ".join(cmd))
        subprocess.run(cmd, cwd=root, env=env_vars, check=True, capture_output=False)
        log.append({"command": "python " + " ".join(cmd[1:]), "cwd": "."})

    retained_artifacts = sorted(
        p.resolve().relative_to(root_resolved).as_posix()
        for p in (
            *data_processed.glob("*.npy"),
            *data_processed.glob("index.json"),
            *mjcf_root.rglob("*.xml"),
            *artifacts.rglob("*.pt"),
            *artifacts.rglob("*.npy"),
            *artifacts.rglob("*.jsonl"),
        )
    )
    manifest_path = artifacts / "manifest.jsonl"
    manifest_records: list[dict[str, object]] = [
        {
            "record_type": "manifest",
            "description": (
                "Reproducible artifact chain: supervised (YCB mesh -> synthetic dataset "
                "-> grasp checkpoint -> generated grasps -> MuJoCo simulation -> eval report) "
                "and RL (SB3 PPO -> legacy checkpoint -> policy_runner inference)"
            ),
            "config_dir": config_dir_arg,
        }
    ]
    manifest_records.extend(
        {"record_type": "command", **entry} for entry in log
    )
    manifest_records.extend(
        {"record_type": "retained_artifact", "path": rel}
        for rel in retained_artifacts
    )
    write_jsonl_records(manifest_path, manifest_records)

    observation_dim = int(config_get(cfg, "rl", "observation_dim"))
    action_dim = int(config_get(cfg, "rl", "action_dim"))
    infer_path_arg = (artifacts / "rl_inference_smoke.py").resolve().relative_to(
        root_resolved
    ).as_posix()
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
    print(">>>", sys.executable, str(infer_path))
    subprocess.run([sys.executable, str(infer_path)], cwd=root, env=env_vars, check=True)
    log.append({"command": f"python {infer_path_arg}", "cwd": "."})

    print("All artifact-chain steps completed.")
    print("Manifest written to artifacts/manifest.jsonl")


if __name__ == "__main__":
    main()
