from grasping_ai import (
    FlattenedYAMLConfig,
    run_rl_training_pipeline,
    setup_logging,
)

# %% [markdown]
# # RL Grasp Policy Training
#
# **Objective.** Train a PPO policy (Stable-Baselines3) in the Gymnasium-compatible MuJoCo grasping environment.
#
# **Outputs.** Policy checkpoint and TensorBoard paths from `configs/rl/default.yaml`.
#
# **Prerequisites.** Python ≥ 3.12, MuJoCo assets (robot + YCB MJCF). GPU optional.

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
#
# RL hyperparameters and paths come from `configs/rl/default.yaml` and `configs/base.yaml`.

# %%

from notebook_helpers import (
    config_dir_relative,
    load_notebook_config,
    configure_seeds_and_device,
    notebook_experiment,
    notebook_download_ycb,
    notebook_object_index,
    print_runtime_banner,
    run_dataset_preparation,
    print_checkpoint_summary,
    object_ids_from_config,
)

CONFIG_DIR = root / "configs"
CONFIG_NAME = "rl/rl_train"
config_dir_arg = config_dir_relative(CONFIG_DIR, root)

cfg = load_notebook_config(CONFIG_DIR, CONFIG_NAME)
yaml_config = FlattenedYAMLConfig(cfg)
all_object_ids = object_ids_from_config(cfg)
training_object_ids = [all_object_ids[notebook_object_index(cfg)]]
seed, device = configure_seeds_and_device(cfg)
learning_rate = yaml_config.value( "rl", "learning_rate", value_type=float, default=3e-4)
num_updates = yaml_config.value( "rl", "num_updates", value_type=int, default=10)
experiment = notebook_experiment(cfg)

print_runtime_banner(
    experiment=experiment,
    config_names=CONFIG_NAME,
    root=root,
    device=device,
    seed=seed,
    training_object=training_object_ids[0],
    rl={"learning_rate": learning_rate, "num_updates": num_updates},
)

# %% [markdown]
# ## 3. Data and simulation assets

# %%
paths = run_dataset_preparation(
    root,
    cfg=cfg,
    config_dir_arg=config_dir_arg,
    config_name=CONFIG_NAME,
    object_ids=all_object_ids,
    download_ycb=notebook_download_ycb(cfg),
)
robot_xml = yaml_config.value( "robot", "description", value_type=Path)
print("robot_xml:", robot_xml)
print("ycb_mjcf:", paths["mjcf_root"])

# %% [markdown]
# ## 4. Training

# %%
setup_logging(module_name="train_rl")
policy_checkpoint = yaml_config.value( "rl", "checkpoint", value_type=Path)
policy_checkpoint.parent.mkdir(parents=True, exist_ok=True)

run_rl_training_pipeline(
    robot_xml_path=robot_xml,
    ycb_root=paths["mjcf_root"],
    object_ids=training_object_ids,
    policy_checkpoint_path=policy_checkpoint,
    observation_dim=int(yaml_config.get_path( "rl", "observation_dim")),
    action_dim=int(yaml_config.get_path( "rl", "action_dim")),
    hidden_dim=int(yaml_config.get_path( "rl", "hidden_dim")),
    learning_rate=learning_rate,
    num_updates=num_updates,
    gamma=float(yaml_config.get_path( "rl", "gamma")),
    device=device,
    seed=seed,
    experiment_log_dir=yaml_config.value( "rl", "tensorboard", value_type=Path),
)

# %% [markdown]
# ## 5. Results

# %%
print_checkpoint_summary(policy_checkpoint, yaml_config.value( "rl", "tensorboard", value_type=Path))
