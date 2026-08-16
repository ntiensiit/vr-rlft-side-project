from grasping_ai import FlattenedYAMLConfig

# %% [markdown]
# # RL Policy Grasp Evaluation
#
# **Objective.** Evaluate the trained PPO RL grasp policy on rollout metrics.
#
# **Inputs.** Checkpoints, observations, and report paths from `configs/`.
#
# **Outputs.** JSONL rollout reports under `paths.reports` and optional MLflow metrics.

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

from notebook_helpers import (
    config_dir_relative,
    load_notebook_config,
    configure_seeds_and_device,
    notebook_experiment,
    selected_object_id,
    print_runtime_banner,
    object_ids_from_config,
)

CONFIG_DIR = root / "configs"
config_dir_arg = config_dir_relative(CONFIG_DIR, root)
RL_CONFIG_NAME = "evaluation/rl"

cfg = load_notebook_config(CONFIG_DIR, "config", overrides=["evaluation=rl", "training=diffusion"])
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
    config_names={"rl": RL_CONFIG_NAME},
    root=root,
    device=device,
    seed=seed,
)

# %% [markdown]
# ## 3. Checkpoint validation

# %%
checkpoint_path = yaml_config.value( "rl", "checkpoint", value_type=Path)
exists = checkpoint_path.is_file()
print("rl", {"path": str(checkpoint_path), "exists": exists})
if not exists:
    raise FileNotFoundError(f"Missing RL checkpoint: {checkpoint_path}")

rl_episodes = yaml_config.value( "evaluation", "episodes", value_type=int, default=5)
rl_max_steps = yaml_config.value( "evaluation", "max_steps", value_type=int, default=100)
rl_stochastic = yaml_config.value( "evaluation", "stochastic", value_type=bool, default=False)
rl_exploration_noise = yaml_config.value( "evaluation", "exploration_noise", value_type=float, default=0.1)

# %% [markdown]
# ## 4. RL policy rollouts

# %%
from run_rl_evaluation import run_rl_evaluation_main

rl_output_path = yaml_config.value( "evaluation", "rollout_report", value_type=Path)
run_rl_evaluation_main(
    policy_checkpoint_path=checkpoint_path,
    robot_xml_path=yaml_config.value( "robot", "description", value_type=Path),
    ycb_root=yaml_config.value( "paths", "ycb_mjcf", value_type=Path),
    object_id=object_id,
    observation_dim=int(yaml_config.get_path( "rl", "observation_dim")),
    action_dim=int(yaml_config.get_path( "rl", "action_dim")),
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
# ## 5. Results summary

# %%
comparison = {
    "object_id": object_id,
    "reports": {
        "rl_rollout": str(rl_output_path),
    },
}
print(json.dumps(comparison, indent=2, default=str))
