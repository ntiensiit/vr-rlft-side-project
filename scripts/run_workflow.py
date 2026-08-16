"""Orchestrate multi-step grasping workflows."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import hydra
import numpy as np
from loguru import logger

from grasping_ai.config import SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG
from grasping_ai.data.pointcloud_dataset import resolve_ycb_object_id
from grasping_ai.evaluation.metrics import aggregate_grasp_success_rate
from grasping_ai.pipelines.evaluate import read_jsonl_records
from grasping_ai.sensors.pointcloud_sensor import sample_point_cloud_from_mesh

if TYPE_CHECKING:
    from omegaconf import DictConfig


CHECKPOINT_PATH = Path(FLATTENED_YAML_CONFIG.get("script.checkpoint", ""))
OUTPUT_DIR = Path(FLATTENED_YAML_CONFIG.get("script.output_dir", "artifacts"))
METHOD = str(FLATTENED_YAML_CONFIG.get("script.method", "diffusion"))
FEATURE_DIM = int(FLATTENED_YAML_CONFIG.get("script.feature_dim", 32))
NUM_STEPS = int(FLATTENED_YAML_CONFIG.get("script.num_steps", 10))
NUM_GRASPS = int(FLATTENED_YAML_CONFIG.get("script.num_grasps", 32))
DEVICE = str(FLATTENED_YAML_CONFIG.get("script.device", "cpu"))
SEED = int(FLATTENED_YAML_CONFIG.get("script.seed", 42))
GRIPPER_POINT_CLOUD_PATH = Path(FLATTENED_YAML_CONFIG.get("script.gripper_point_cloud", ""))
EVAL_OBJECT_KEY = str(FLATTENED_YAML_CONFIG.get("script.eval_object_key", "object"))
RL_ROLLOUT_REPORT_NAME = str(FLATTENED_YAML_CONFIG.get("script.rl_rollout_report", "rl_rollout.jsonl"))
GRASP_CANDIDATES_NAME = str(FLATTENED_YAML_CONFIG.get("script.grasp_candidates_name", "grasps_{method}.npz"))
SIMULATION_OUTCOMES_NAME = str(
    FLATTENED_YAML_CONFIG.get("script.simulation_outcomes_name", "simulation_{method}.jsonl"),
)
ANALYTICAL_EVALUATION_NAME = str(
    FLATTENED_YAML_CONFIG.get("script.analytical_evaluation_name", "evaluation_{method}.json"),
)
OBJECT_POINT_CLOUD_NAME = str(FLATTENED_YAML_CONFIG.get("script.object_point_cloud", "object.npy"))
POINT_CLOUD_SAMPLE_MULTIPLIER = int(FLATTENED_YAML_CONFIG.get("script.point_cloud_sample_multiplier", 4))
PYCACHE_DIR = str(FLATTENED_YAML_CONFIG.get("script.pycache_dir", ".pycache"))
OBSERVATION_PATH = FLATTENED_YAML_CONFIG.get("script.observation")
YCB_ROOT_RAW = FLATTENED_YAML_CONFIG.get("script.ycb_root")
YCB_ROOT_MJCF = FLATTENED_YAML_CONFIG.get("script.ycb_mjcf")
OBJECT_ID = FLATTENED_YAML_CONFIG.get("script.object_id")
ROBOT_XML_PATH = FLATTENED_YAML_CONFIG.get("script.robot_xml")
OBSERVATION_DIM = FLATTENED_YAML_CONFIG.get("script.observation_dim")
ACTION_DIM = FLATTENED_YAML_CONFIG.get("script.action_dim")
RL_POLICY_CHECKPOINT_PATH = FLATTENED_YAML_CONFIG.get("script.rl_policy_checkpoint")
RL_EPISODES = FLATTENED_YAML_CONFIG.get("script.rl_episodes")
RL_MAX_STEPS = FLATTENED_YAML_CONFIG.get("script.rl_max_steps")
TABLE_XML_PATH = FLATTENED_YAML_CONFIG.get("script.table_xml")
NUM_SIMULATION_STEPS = int(FLATTENED_YAML_CONFIG.get("script.num_simulation_steps", 100))
GRIPPER_CLOSE_COMMAND = float(FLATTENED_YAML_CONFIG.get("script.gripper_close_command", 0.0))
FRICTION_COEFFICIENT = float(FLATTENED_YAML_CONFIG.get("script.friction_coefficient", 0.5))
LIFT_HEIGHT_THRESHOLD = float(FLATTENED_YAML_CONFIG.get("script.lift_height_threshold", 0.02))
CONTACT_CLEARANCE = float(FLATTENED_YAML_CONFIG.get("script.contact_clearance", 0.01))
WRENCH_REGULARIZATION = float(FLATTENED_YAML_CONFIG.get("script.wrench_regularization", 0.001))
GRASP_POSE_FORMAT = str(FLATTENED_YAML_CONFIG.get("script.grasp_pose_format", "object"))


def _validate_workflow(options: SimpleNamespace) -> None:
    """Raise ``ValueError`` when a required scalar workflow setting is missing."""
    for name in (
        "num_simulation_steps",
        "gripper_close_command",
        "friction_coefficient",
        "lift_height_threshold",
        "contact_clearance",
        "wrench_regularization",
        "grasp_pose_format",
    ):
        if getattr(options, name) is None:
            msg = f"{name} is required"
            raise ValueError(msg)


def _build_workflow_env(root: Path, pycache_dir: str) -> dict[str, str]:
    """Build the subprocess environment for workflow child commands."""
    return {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
        "PYTHONPYCACHEPREFIX": str(root / pycache_dir),
    }


def _create_output_paths(options: SimpleNamespace, root: Path) -> SimpleNamespace:
    """Create the export/report directories and derive the stage artifact paths."""
    exports_dir = options.output_dir / "exports"
    reports_dir = options.output_dir / "reports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        root=root,
        grasps=exports_dir / options.grasp_candidates_name.format(method=options.method),
        simulation=reports_dir / options.simulation_outcomes_name.format(method=options.method),
        evaluation=reports_dir / options.analytical_evaluation_name.format(method=options.method),
        rl_rollout=reports_dir / options.rl_rollout_report_name,
    )


def _run_grasp_inference_stage(options: SimpleNamespace, paths: SimpleNamespace, workflow_env: dict[str, str]) -> None:
    """Run the grasp-generation stage."""
    grasp_inference_cmd: list[str] = [
        sys.executable,
        "scripts/run_grasp_inference.py",
        f"script.checkpoint={options.checkpoint_path.as_posix()}",
        f"script.output={paths.grasps.as_posix()}",
        f"script.method={options.method}",
        f"architecture.feature_dim={options.feature_dim}",
        f"model.inference_steps={options.num_steps}",
        f"architecture.num_grasps={options.num_grasps}",
        f"device={options.device}",
        f"seed={options.seed}",
    ]
    if options.observation_path is not None:
        grasp_inference_cmd.append(f"script.observation={options.observation_path}")
    else:
        if options.ycb_root_raw is None or options.object_id is None:
            msg = "Provide script.observation or both script.ycb_root and script.object_id"
            raise ValueError(msg)
        grasp_inference_cmd += [
            f"script.ycb_root={options.ycb_root_raw}",
            f"script.object_id={options.object_id}",
        ]
    logger.info(">>> {}", " ".join(grasp_inference_cmd))
    subprocess.run(  # noqa: S603  # fixed internal CLI, no shell
        grasp_inference_cmd, cwd=paths.root, env=workflow_env, check=True, capture_output=False,
    )


def _run_simulation_stage(options: SimpleNamespace, paths: SimpleNamespace, workflow_env: dict[str, str]) -> None:
    """Run the MuJoCo grasp simulation stage when scene inputs are provided."""
    if options.robot_xml_path is not None and options.ycb_root_mjcf is not None and options.object_id is not None:
        sim_cmd: list[str] = [
            sys.executable,
            "scripts/run_simulation.py",
            f"script.grasps={paths.grasps.as_posix()}",
            f"script.object_id={options.object_id}",
            f"script.ycb_root={options.ycb_root_mjcf}",
            f"script.robot_xml={options.robot_xml_path}",
            f"script.output={paths.simulation.as_posix()}",
            f"num_steps={options.num_simulation_steps}",
            f"robot.gripper.close_command=[{options.gripper_close_command}]",
            f"script.grasp_pose_format={options.grasp_pose_format}",
        ]
        if options.table_xml_path is not None:
            sim_cmd.append(f"script.table_xml={options.table_xml_path}")
        logger.info(">>> {}", " ".join(sim_cmd))
        subprocess.run(  # noqa: S603  # fixed internal CLI, no shell
            sim_cmd, cwd=paths.root, env=workflow_env, check=True, capture_output=False,
        )
    else:
        logger.info("skipping MuJoCo simulation stage: robot-xml / ycb-mjcf / object-id not fully provided")


def _resolve_object_point_cloud(options: SimpleNamespace) -> Path:
    """Return the object point cloud, sampling it from the YCB mesh when needed."""
    if options.observation_path is not None:
        return options.observation_path
    if options.ycb_root_raw is not None and options.object_id is not None:
        object_pc_path = options.output_dir / options.object_point_cloud_name
        rng = np.random.default_rng(options.seed)
        mesh_path = resolve_ycb_object_id(options.ycb_root_raw, options.object_id)
        object_pc = sample_point_cloud_from_mesh(
            mesh_path, options.num_grasps * options.point_cloud_sample_multiplier, rng,
        )
        np.save(object_pc_path, object_pc.astype(np.float32))
        return object_pc_path
    msg = "Cannot locate an object point cloud for evaluation"
    raise ValueError(msg)


def _resolve_gripper_point_cloud(gripper_point_cloud_path: Path, root: Path) -> Path:
    """Resolve the gripper point cloud against the repo root and require that it exists."""
    if not gripper_point_cloud_path.is_absolute():
        gripper_point_cloud_path = root / gripper_point_cloud_path
    if not gripper_point_cloud_path.is_file():
        msg = (
            f"Gripper point cloud not found at {gripper_point_cloud_path}; "
            "run scripts/prepare_observations.py first"
        )
        raise FileNotFoundError(msg)
    return gripper_point_cloud_path


def _run_evaluation_stage(
    options: SimpleNamespace,
    paths: SimpleNamespace,
    object_pc_path: Path,
    workflow_env: dict[str, str],
) -> None:
    """Run the analytical evaluation stage."""
    gripper_point_cloud_path = _resolve_gripper_point_cloud(options.gripper_point_cloud_path, paths.root)
    eval_cmd: list[str] = [
        sys.executable,
        "scripts/evaluate.py",
        f"script.grasps={paths.grasps.as_posix()}",
        f"script.object_id={options.eval_object_key}",
        f"script.object_point_cloud={object_pc_path.as_posix()}",
        f"script.gripper_point_cloud={gripper_point_cloud_path.as_posix()}",
        f"script.report={paths.evaluation.as_posix()}",
        f"metrics.friction_coefficient={options.friction_coefficient}",
        f"metrics.lift_height_threshold={options.lift_height_threshold}",
        f"metrics.collision_clearance={options.contact_clearance}",
        f"metrics.wrench_regularization={options.wrench_regularization}",
    ]
    logger.info(">>> {}", " ".join(eval_cmd))
    subprocess.run(  # noqa: S603  # fixed internal CLI, no shell
        eval_cmd, cwd=paths.root, env=workflow_env, check=True, capture_output=False,
    )


def _run_rl_rollout_stage(options: SimpleNamespace, paths: SimpleNamespace, workflow_env: dict[str, str]) -> None:
    """Run the optional RL policy rollout stage."""
    if options.rl_policy_checkpoint_path is None:
        return
    if (
        options.robot_xml_path is None
        or options.ycb_root_mjcf is None
        or options.object_id is None
        or options.observation_dim is None
        or options.action_dim is None
        or options.rl_episodes is None
        or options.rl_max_steps is None
    ):
        msg = ("RL rollout stage requires script.robot_xml, script.ycb_mjcf, "
            "script.object_id, script.observation_dim, script.action_dim, "
            "script.rl_episodes, script.rl_max_steps")
        raise ValueError(
            msg,
        )
    rl_cmd: list[str] = [
        sys.executable,
        "scripts/run_rl_evaluation.py",
        f"script.policy_checkpoint={options.rl_policy_checkpoint_path}",
        f"script.robot_xml={options.robot_xml_path}",
        f"script.ycb_root={options.ycb_root_mjcf}",
        f"script.object_id={options.object_id}",
        f"script.observation_dim={options.observation_dim}",
        f"script.action_dim={options.action_dim}",
        f"script.output={paths.rl_rollout.as_posix()}",
        f"evaluation.episodes={options.rl_episodes}",
        f"evaluation.max_steps={options.rl_max_steps}",
        f"device={options.device}",
        f"seed={options.seed}",
    ]
    if options.table_xml_path is not None:
        rl_cmd.append(f"script.table_xml={options.table_xml_path}")
    logger.info(">>> {}", " ".join(rl_cmd))
    subprocess.run(  # noqa: S603  # fixed internal CLI, no shell
        rl_cmd, cwd=paths.root, env=workflow_env, check=True, capture_output=False,
    )


def _read_analytical_summary(eval_path: Path, summary: dict[str, float]) -> None:
    """Merge analytical evaluation metrics from the report into ``summary``."""
    if not eval_path.is_file():
        return
    try:
        for record in read_jsonl_records(eval_path):
            if record.get("record_type") != "summary":
                continue
            for key in ("success_rate", "collision_free_rate", "force_closure_rate"):
                if key in record:
                    summary[f"analytical_{key}"] = float(record[key])
            if "object_success_rate" in record:
                summary["analytical_object_success_rate"] = float(record["object_success_rate"])
            break
    except (OSError, ValueError):
        pass


def _read_simulation_summary(sim_path: Path, eval_object_key: str, summary: dict[str, float]) -> None:
    """Merge simulated grasp success metrics into ``summary``."""
    if not sim_path.is_file():
        return
    try:
        outcomes = [
            record for record in read_jsonl_records(sim_path) if record.get("record_type") == "grasp_outcome"
        ]
        if outcomes:
            n = len(outcomes)
            successes = sum(1 for outcome in outcomes if bool(outcome.get("success")))
            summary["simulated_success_rate"] = float(successes / n)
            summary["simulated_object_success_rate"] = aggregate_grasp_success_rate({eval_object_key: successes > 0})
    except (OSError, ValueError):
        pass


def _read_rl_summary(rl_path: Path, summary: dict[str, float]) -> None:
    """Merge RL rollout return statistics into ``summary``."""
    if not rl_path.is_file():
        return
    try:
        episodes = [record for record in read_jsonl_records(rl_path) if record.get("record_type") == "episode"]
        if episodes:
            mean_return = sum(float(record["summary"]["return_total"]) for record in episodes) / len(episodes)
            mean_len = sum(float(record["summary"]["length"]) for record in episodes) / len(episodes)
            summary["rl_mean_return"] = float(mean_return)
            summary["rl_mean_length"] = float(mean_len)
    except (OSError, ValueError, KeyError, TypeError):
        pass


def _log_workflow_summary(options: SimpleNamespace, paths: SimpleNamespace) -> None:
    """Aggregate the stage report metrics and log them."""
    summary: dict[str, float] = {}
    _read_analytical_summary(paths.evaluation, summary)
    _read_simulation_summary(paths.simulation, options.eval_object_key, summary)
    _read_rl_summary(paths.rl_rollout, summary)

    logger.info("workflow summary:")
    for key, value in summary.items():
        logger.info("  {} = {:.4f}", key, value)


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/run_workflow")
def main(cfg: DictConfig) -> None:
    """Run the configured end-to-end grasping workflow."""
    yaml_config = FlattenedYAMLConfig(cfg)
    method = str(
        yaml_config.value(
            "method", "evaluation", "method", value_type=object, script_or=True, default=METHOD,
        ),
    )
    checkpoint = yaml_config.value(
        "checkpoint", "model", "checkpoint", value_type=Path, script_or=True, default=CHECKPOINT_PATH,
    )
    if checkpoint is None:
        msg = "model.checkpoint path is required"
        raise ValueError(msg)

    output_dir = yaml_config.value(
        "output_dir", "artifacts", "root", value_type=Path, script_or=True, default=OUTPUT_DIR,
    )
    close_default = yaml_config.value(
        "close_command", "robot", "gripper", "close_command", value_type=list[float], script_or=True,
    ) or [GRIPPER_CLOSE_COMMAND]
    intermediate = yaml_config.get_path("workflow", "intermediate")
    if not isinstance(intermediate, dict):
        msg = "workflow.intermediate must be a mapping"
        raise TypeError(msg)
    rl_report = yaml_config.value(
        "rl_rollout_report",
        "workflow",
        "rl_rollout_report",
        value_type=Path,
        script_or=True,
        default=Path(RL_ROLLOUT_REPORT_NAME),
    )

    inference = dict(  # noqa: C408  # keyword grouping keeps options readable
        checkpoint_path=checkpoint,
        method=method,
        feature_dim=yaml_config.value(
            "feature_dim", "architecture", "feature_dim", value_type=int, script_or=True, default=FEATURE_DIM,
        ),
        num_steps=yaml_config.value(
            "num_steps", "model", "inference_steps", value_type=int, script_or=True, default=NUM_STEPS,
        ),
        num_grasps=yaml_config.value(
            "num_grasps", "architecture", "num_grasps", value_type=int, script_or=True, default=NUM_GRASPS,
        ),
        device=str(yaml_config.value("device", "device", value_type=object, script_or=True, default=DEVICE)),
        seed=yaml_config.value("seed", "seed", value_type=int, script_or=True, default=SEED),
    )
    artifact = dict(  # noqa: C408
        output_dir=output_dir,
        gripper_point_cloud_path=yaml_config.value(
            "gripper_point_cloud",
            "workflow",
            "gripper_point_cloud",
            value_type=Path,
            script_or=True,
            default=GRIPPER_POINT_CLOUD_PATH,
        ),
        eval_object_key=str(
            yaml_config.value(
                "eval_object_key",
                "workflow",
                "eval_object_key",
                value_type=object,
                script_or=True,
                default=EVAL_OBJECT_KEY,
            ),
        ),
        rl_rollout_report_name=rl_report.name,
        grasp_candidates_name=str(intermediate.get("grasp_candidates", GRASP_CANDIDATES_NAME)),
        simulation_outcomes_name=str(intermediate.get("simulation_outcomes", SIMULATION_OUTCOMES_NAME)),
        analytical_evaluation_name=str(intermediate.get("analytical_evaluation", ANALYTICAL_EVALUATION_NAME)),
        object_point_cloud_name=str(
            yaml_config.value(
                "object_point_cloud",
                "workflow",
                "object_point_cloud",
                value_type=object,
                script_or=True,
                default=OBJECT_POINT_CLOUD_NAME,
            ),
        ),
        point_cloud_sample_multiplier=yaml_config.value(
            "point_cloud_sample_multiplier",
            "workflow",
            "point_cloud_sample_multiplier",
            value_type=int,
            script_or=True,
            default=POINT_CLOUD_SAMPLE_MULTIPLIER,
        ),
        pycache_dir=str(
            yaml_config.value(
                "pycache_dir", "workflow", "pycache_dir", value_type=object, script_or=True, default=PYCACHE_DIR,
            ),
        ),
    )
    scene = dict(  # noqa: C408
        observation_path=(
            Path(observation_raw)
            if isinstance(
                observation_raw := yaml_config.value(
                    "observation", "script", "observation", value_type=object, default=OBSERVATION_PATH, script_or=True,
                ),
                str,
            )
            else observation_raw
        ),
        ycb_root_raw=yaml_config.value(
            "ycb_root", "paths", "ycb_root", value_type=Path, script_or=True, default=YCB_ROOT_RAW,
        ),
        ycb_root_mjcf=yaml_config.value(
            "ycb_mjcf", "paths", "ycb_mjcf", value_type=Path, script_or=True, default=YCB_ROOT_MJCF,
        ),
        object_id=yaml_config.value("object_id", value_type=object, script_or=True, default=OBJECT_ID),
        robot_xml_path=yaml_config.value(
            "robot_xml", "robot", "description", value_type=Path, script_or=True, default=ROBOT_XML_PATH,
        ),
        table_xml_path=yaml_config.value(
            "table_xml", "env", "table_xml", value_type=Path, script_or=True, default=TABLE_XML_PATH,
        ),
    )
    rl = dict(  # noqa: C408
        observation_dim=yaml_config.value(
            "observation_dim", value_type=object, script_or=True, default=OBSERVATION_DIM,
        ),
        action_dim=yaml_config.value("action_dim", value_type=object, script_or=True, default=ACTION_DIM),
        rl_policy_checkpoint_path=yaml_config.value(
            "rl_policy_checkpoint", value_type=object, script_or=True, default=RL_POLICY_CHECKPOINT_PATH,
        ),
        rl_episodes=yaml_config.value("rl_episodes", value_type=object, script_or=True, default=RL_EPISODES),
        rl_max_steps=yaml_config.value("rl_max_steps", value_type=object, script_or=True, default=RL_MAX_STEPS),
    )
    simulation = dict(  # noqa: C408
        num_simulation_steps=yaml_config.value(
            "num_steps", "num_steps", value_type=int, script_or=True, default=NUM_SIMULATION_STEPS,
        ),
        gripper_close_command=close_default[0],
        friction_coefficient=yaml_config.value(
            "friction_coefficient",
            "metrics",
            "friction_coefficient",
            value_type=float,
            script_or=True,
            default=FRICTION_COEFFICIENT,
        ),
        lift_height_threshold=yaml_config.value(
            "lift_height_threshold",
            "metrics",
            "lift_height_threshold",
            value_type=float,
            script_or=True,
            default=LIFT_HEIGHT_THRESHOLD,
        ),
        contact_clearance=yaml_config.value(
            "contact_clearance",
            "metrics",
            "collision_clearance",
            value_type=float,
            script_or=True,
            default=CONTACT_CLEARANCE,
        ),
        wrench_regularization=yaml_config.value(
            "wrench_regularization",
            "metrics",
            "wrench_regularization",
            value_type=float,
            script_or=True,
            default=WRENCH_REGULARIZATION,
        ),
        grasp_pose_format=str(
            yaml_config.value(
                "grasp_pose_format",
                "workflow",
                "grasp_pose_format",
                value_type=object,
                script_or=True,
                default=GRASP_POSE_FORMAT,
            ),
        ),
    )
    root = Path(__file__).resolve().parents[1]
    validation = SimpleNamespace(**simulation)
    paths_config = SimpleNamespace(**artifact, **inference)
    grasp = SimpleNamespace(**inference, **scene)
    simulation_stage = SimpleNamespace(**scene, **simulation)
    point_cloud = SimpleNamespace(**inference, **artifact, **scene)
    evaluation = SimpleNamespace(**artifact, **simulation)
    rl_stage = SimpleNamespace(**inference, **scene, **rl)

    _validate_workflow(validation)
    workflow_env = _build_workflow_env(root, artifact["pycache_dir"])
    paths = _create_output_paths(paths_config, root)
    _run_grasp_inference_stage(grasp, paths, workflow_env)
    _run_simulation_stage(simulation_stage, paths, workflow_env)
    object_pc_path = _resolve_object_point_cloud(point_cloud)
    _run_evaluation_stage(evaluation, paths, object_pc_path, workflow_env)
    _run_rl_rollout_stage(rl_stage, paths, workflow_env)
    _log_workflow_summary(
        SimpleNamespace(eval_object_key=artifact["eval_object_key"]),
        paths,
    )

if __name__ == "__main__":
    main()
