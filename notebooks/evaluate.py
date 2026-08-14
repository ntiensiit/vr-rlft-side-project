# %% [markdown]
# # Diffusion, Flow, and RL Evaluation
#
# **Objective.** Compare generative grasp models (diffusion vs flow) and an RL policy on shared analytical and simulation metrics.
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
#
# Composes diffusion (default), flow, and RL evaluation config groups from `configs/evaluation/`.

# %%
import json

import numpy as np

from grasping_ai.config.yaml_loader import config_bool, config_float, config_get, config_int, config_path
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
DIFFUSION_CONFIG_NAME = "evaluation/diffusion"
FLOW_CONFIG_NAME = "evaluation/flow"
RL_CONFIG_NAME = "evaluation/rl"

cfg = load_notebook_config(CONFIG_DIR, DIFFUSION_CONFIG_NAME)
flow_cfg = load_notebook_config(CONFIG_DIR, FLOW_CONFIG_NAME)
rl_cfg = load_notebook_config(CONFIG_DIR, RL_CONFIG_NAME)
all_object_ids = object_ids_from_config(cfg)
object_id = selected_object_id(cfg)
seed, device = configure_seeds_and_device(cfg)
experiment = notebook_experiment(cfg)

exports_dir = config_path(cfg, "paths", "exports")
reports_dir = config_path(cfg, "paths", "reports")
exports_dir.mkdir(parents=True, exist_ok=True)
reports_dir.mkdir(parents=True, exist_ok=True)

print_runtime_banner(
    experiment=experiment,
    config_names={
        "diffusion": DIFFUSION_CONFIG_NAME,
        "flow": FLOW_CONFIG_NAME,
        "rl": RL_CONFIG_NAME,
    },
    root=root,
    device=device,
    seed=seed,
    evaluation_object=object_id,
    exports_dir=exports_dir,
    reports_dir=reports_dir,
)

# %% [markdown]
# ## 3. Data and checkpoint validation

# %%
paths = run_dataset_preparation(
    root,
    cfg=cfg,
    config_dir_arg=config_dir_arg,
    config_name=DIFFUSION_CONFIG_NAME,
    object_ids=all_object_ids,
    download_ycb=notebook_download_ycb(cfg),
)
observations_dir = paths["observations_dir"]
observation_path = observations_dir / f"{object_id}.npy"
gripper_point_cloud = np.load(observations_dir / "gripper.npy")
object_point_cloud = np.load(observation_path)

checkpoints = {
    "diffusion": config_path(cfg, "diffusion", "checkpoint"),
    "flow": config_path(flow_cfg, "flow", "checkpoint"),
    "rl": config_path(cfg, "rl", "checkpoint"),
}
for name, path in checkpoints.items():
    exists = path.is_file()
    print(name, {"path": str(path), "exists": exists})
    if not exists:
        raise FileNotFoundError(f"Missing {name} checkpoint: {path}")

close_default = config_get(cfg, "robot", "gripper", "close_command") or [0.0]
gripper_close_command = np.asarray(close_default, dtype=np.float64)
num_simulation_steps = int(config_get(cfg, "num_steps"))
friction_coefficient = config_float(cfg, "metrics", "friction_coefficient", default=0.5)
lift_height_threshold = config_float(cfg, "metrics", "lift_height_threshold", default=0.05)
contact_clearance = config_float(cfg, "metrics", "collision_clearance", default=0.005)
wrench_regularization = config_float(cfg, "metrics", "wrench_regularization", default=1.0)
diffusion_filter_collisions = config_bool(cfg, "evaluation", "filter_collisions", default=False)
flow_filter_collisions = config_bool(flow_cfg, "evaluation", "filter_collisions", default=False)
rl_episodes = config_int(rl_cfg, "evaluation", "episodes", default=5)
rl_max_steps = config_int(rl_cfg, "evaluation", "max_steps", default=100)
rl_stochastic = config_bool(rl_cfg, "evaluation", "stochastic", default=False)
rl_exploration_noise = config_float(rl_cfg, "evaluation", "exploration_noise", default=0.1)

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
# ## 4. Diffusion inference and evaluation

# %%
from grasping_ai.utils.logging_utils import init_mlflow, setup_logging

setup_logging(module_name="evaluate")
use_mlflow = init_mlflow(cfg)

diffusion_grasp_path = exports_dir / f"diffusion_grasp_poses_{object_id}.npy"
diffusion_aggregated, diffusion_sim_outcomes, diffusion_report_path, diffusion_sim_path = run_generative_evaluation(
    method="diffusion",
    cfg=cfg,
    checkpoint_path=checkpoints["diffusion"],
    grasp_path=diffusion_grasp_path,
    filter_collisions=diffusion_filter_collisions,
    **eval_common,
)

# %% [markdown]
# ## 5. Flow inference and evaluation

# %%
flow_grasp_path = exports_dir / f"flow_grasp_poses_{object_id}.npy"
flow_aggregated, flow_sim_outcomes, flow_report_path, flow_sim_path = run_generative_evaluation(
    method="flow",
    cfg=flow_cfg,
    checkpoint_path=checkpoints["flow"],
    grasp_path=flow_grasp_path,
    filter_collisions=flow_filter_collisions,
    **eval_common,
)

# %% [markdown]
# ## 6. RL policy rollouts

# %%
from run_rl_evaluation import run_rl_evaluation_main

rl_output_path = config_path(rl_cfg, "evaluation", "rollout_report")
run_rl_evaluation_main(
    policy_checkpoint_path=checkpoints["rl"],
    robot_xml_path=config_path(cfg, "robot", "description"),
    ycb_root=config_path(cfg, "paths", "ycb_mjcf"),
    object_id=object_id,
    observation_dim=int(config_get(cfg, "rl", "observation_dim")),
    action_dim=int(config_get(cfg, "rl", "action_dim")),
    output_path=rl_output_path,
    episodes=rl_episodes,
    max_steps=rl_max_steps,
    device=device,
    seed=seed,
    table_xml_path=None,
    observation_dim_from_env=False,
    action_dim_from_env=False,
    stochastic=rl_stochastic,
    exploration_noise=rl_exploration_noise,
)

# %% [markdown]
# ## 7. Results summary

# %%
comparison = {
    "object_id": object_id,
    "diffusion_analytical": diffusion_aggregated,
    "flow_analytical": flow_aggregated,
    "diffusion_sim_lift_success_rate": lift_success_rate(diffusion_sim_outcomes),
    "flow_sim_lift_success_rate": lift_success_rate(flow_sim_outcomes),
    "reports": {
        "diffusion_analytical": str(diffusion_report_path),
        "flow_analytical": str(flow_report_path),
        "diffusion_simulation": str(diffusion_sim_path),
        "flow_simulation": str(flow_sim_path),
        "rl_rollout": str(rl_output_path),
    },
}
print(json.dumps(comparison, indent=2, default=str))

if use_mlflow:
    import mlflow

    with mlflow.start_run(run_name=experiment):
        mlflow.log_param("object_id", object_id)
        for key, val in diffusion_aggregated.items():
            if isinstance(val, (int, float)):
                mlflow.log_metric(f"diffusion_{key}", val)
        for key, val in flow_aggregated.items():
            if isinstance(val, (int, float)):
                mlflow.log_metric(f"flow_{key}", val)
        diffusion_rate = lift_success_rate(diffusion_sim_outcomes)
        flow_rate = lift_success_rate(flow_sim_outcomes)
        if diffusion_rate is not None:
            mlflow.log_metric("diffusion_sim_lift_success_rate", diffusion_rate)
        if flow_rate is not None:
            mlflow.log_metric("flow_sim_lift_success_rate", flow_rate)
