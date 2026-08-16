from grasping_ai import (
    FlattenedYAMLConfig,
    init_mlflow,
    setup_logging,
)

# %% [markdown]
# # Flow-Matching Grasp Evaluation
#
# **Objective.** Evaluate generative flow-matching grasp poses on analytical and simulation metrics.
#
# **Inputs.** Checkpoints, observations, and report paths from `configs/`.
#
# **Outputs.** JSONL reports under `configs/data/default.yaml` `paths.reports` and optional MLflow metrics.

# %% [markdown]
# ## 1. Environment

# %%
import subprocess
import sys
from pathlib import Path

try:
    import google.colab  # noqa: F401

    in_colab = True
except ImportError:
    google.colab = None  # pragma: no cover
    in_colab = False

if in_colab:
    repo_root = Path("/content/vr-rlft-side-project")
    if not repo_root.is_dir():
        subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                "dev",
                "--depth",
                "1",
                "https://github.com/ntiensiit/vr-rlft-side-project.git",
                str(repo_root),
            ],
            check=True,
        )
    sys.path.insert(0, str(repo_root / "notebooks"))
else:
    for base in (Path.cwd(), Path.cwd().parent):
        if (base / "notebooks" / "bootstrap.py").is_file():
            sys.path.insert(0, str(base / "notebooks"))
            break

from bootstrap import bootstrap_notebook

root = bootstrap_notebook()

# %% [markdown]
# ## 2. Experiment configuration

# %%
import json
import numpy as np

try:
    import mlflow
except ImportError:
    mlflow = None  # pragma: no cover

from notebook_helpers import (
    config_dir_relative,
    load_notebook_config,
    configure_seeds_and_device,
    notebook_experiment,
    notebook_download_ycb,
    selected_object_id,
    print_runtime_banner,
    run_dataset_preparation,
    run_generative_evaluation,
    lift_success_rate,
    object_ids_from_config,
)

CONFIG_DIR = root / "configs"
config_dir_arg = config_dir_relative(CONFIG_DIR, root)
FLOW_CONFIG_NAME = "evaluation/flow"

cfg = load_notebook_config(CONFIG_DIR, "config", overrides=["evaluation=flow", "training=flow", "model=flow"])
yaml_config = FlattenedYAMLConfig(cfg)
all_object_ids = object_ids_from_config(cfg)
object_id = selected_object_id(cfg)
seed, device = configure_seeds_and_device(cfg)
experiment = notebook_experiment(cfg)

exports_dir = yaml_config.value( "paths", "exports", value_type=Path)
reports_dir = yaml_config.value( "paths", "reports", value_type=Path)
exports_dir.mkdir(parents=True, exist_ok=True)
reports_dir.mkdir(parents=True, exist_ok=True)

print_runtime_banner(
    experiment=experiment,
    config_names={"flow": FLOW_CONFIG_NAME},
    root=root,
    device=device,
    seed=seed,
)

# %% [markdown]
# ## 3. Data and checkpoint validation

# %%
paths = run_dataset_preparation(
    root,
    cfg=cfg,
    config_dir_arg=config_dir_arg,
    config_name="config",
    object_ids=all_object_ids,
    download_ycb=notebook_download_ycb(cfg),
)
observations_dir = paths["observations_dir"]
observation_path = observations_dir / f"{object_id}.npy"
gripper_point_cloud = np.load(observations_dir / "gripper.npy")
object_point_cloud = np.load(observation_path)

checkpoint_path = yaml_config.value( "flow", "checkpoint", value_type=Path)
exists = checkpoint_path.is_file()
print("flow", {"path": str(checkpoint_path), "exists": exists})
if not exists:
    raise FileNotFoundError(f"Missing flow checkpoint: {checkpoint_path}")

close_default = yaml_config.get_path( "robot", "gripper", "close_command") or [0.0]
gripper_close_command = np.asarray(close_default, dtype=np.float64)
num_simulation_steps = int(yaml_config.get_path( "num_steps"))
friction_coefficient = yaml_config.value( "metrics", "friction_coefficient", value_type=float, default=0.5)
lift_height_threshold = yaml_config.value( "metrics", "lift_height_threshold", value_type=float, default=0.05)
contact_clearance = yaml_config.value( "metrics", "collision_clearance", value_type=float, default=0.005)
wrench_regularization = yaml_config.value( "metrics", "wrench_regularization", value_type=float, default=1.0)
flow_filter_collisions = yaml_config.value( "evaluation", "filter_collisions", value_type=bool, default=False)

eval_common = {
    "object_id": object_id,
    "object_point_cloud": object_point_cloud,
    "gripper_point_cloud": gripper_point_cloud,
    "observation_path": observation_path,
    "reports_dir": reports_dir,
    "device": device,
    "seed": seed,
    "friction_coefficient": friction_coefficient,
    "lift_height_threshold": lift_height_threshold,
    "contact_clearance": contact_clearance,
    "wrench_regularization": wrench_regularization,
    "num_simulation_steps": num_simulation_steps,
    "gripper_close_command": gripper_close_command,
}

# %% [markdown]
# ## 4. Flow inference and evaluation

# %%

setup_logging(module_name="evaluate_flow")
use_mlflow = init_mlflow(cfg)

flow_grasp_path = exports_dir / f"flow_grasp_poses_{object_id}.npy"
flow_aggregated, flow_sim_outcomes, flow_report_path, flow_sim_path = run_generative_evaluation(
    method="flow",
    cfg=cfg,
    checkpoint_path=checkpoint_path,
    grasp_path=flow_grasp_path,
    filter_collisions=flow_filter_collisions,
    **eval_common,
)

# %% [markdown]
# ## 5. Results summary

# %%
comparison = {
    "object_id": object_id,
    "flow_aggregated": flow_aggregated,
    "flow_sim_lift_success_rate": lift_success_rate(flow_sim_outcomes),
    "reports": {
        "flow_analytical": str(flow_report_path),
        "flow_simulation": str(flow_sim_path),
    },
}
print(json.dumps(comparison, indent=2, default=str))

if use_mlflow:
    with mlflow.start_run(run_name=experiment):
        mlflow.log_param("object_id", object_id)
        for key, val in flow_aggregated.items():
            if isinstance(val, (int, float)):
                mlflow.log_metric(f"flow_{key}", val)
        flow_rate = lift_success_rate(flow_sim_outcomes)
        if flow_rate is not None:
            mlflow.log_metric("flow_sim_lift_success_rate", flow_rate)
