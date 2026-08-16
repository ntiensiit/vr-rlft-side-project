# %% [markdown]
# # Diffusion Grasp Training
#
# **Objective.** Train a conditional score-based diffusion model that maps object point clouds to 9D grasp poses.
#
# **Outputs.** Checkpoint and TensorBoard paths from `configs/model/diffusion.yaml` and `configs/training/diffusion.yaml`.
#
# **Prerequisites.** Python ≥ 3.12, GPU runtime recommended on Colab.

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
# Edit notebook flags in ``configs/*/…yaml`` under the ``notebook`` key.

# %%
from grasping_ai.config.config import config_value
from grasping_ai.pipelines.train_diffusion import run_diffusion_training_pipeline
from grasping_ai.utils.logging_utils import init_mlflow, setup_logging
from notebook_helpers import (
    build_supervised_train_kwargs,
    config_dir_relative,
    load_notebook_config,
    configure_seeds_and_device,
    notebook_experiment,
    notebook_download_ycb,
    print_runtime_banner,
    run_dataset_preparation,
    run_with_optional_mlflow,
    supervised_hyperparameters,
    supervised_mlflow_log_params,
    print_checkpoint_summary,
    object_ids_from_config,
)

CONFIG_DIR = root / "configs"
CONFIG_NAME = "training/diffusion"
config_dir_arg = config_dir_relative(CONFIG_DIR, root)

cfg = load_notebook_config(CONFIG_DIR, CONFIG_NAME)
object_ids = object_ids_from_config(cfg)
seed, device = configure_seeds_and_device(cfg)
learning_rate, num_epochs, batch_size = supervised_hyperparameters(cfg)
experiment = notebook_experiment(cfg)

print_runtime_banner(
    experiment=experiment,
    config_names=CONFIG_NAME,
    root=root,
    device=device,
    seed=seed,
    supervised={"learning_rate": learning_rate, "num_epochs": num_epochs, "batch_size": batch_size},
)

# %% [markdown]
# ## 3. Data preparation
#
# Uses the same script entry points as `scripts/run_artifacts.py`.

# %%
paths = run_dataset_preparation(
    root,
    cfg=cfg,
    config_dir_arg=config_dir_arg,
    config_name=CONFIG_NAME,
    object_ids=object_ids,
    download_ycb=notebook_download_ycb(cfg),
)
dataset_root = paths["dataset_root"]
print("dataset_root:", dataset_root)
print("records:", sorted(p.name for p in dataset_root.glob("*.npz")))

# %% [markdown]
# ## 4. Training

# %%
setup_logging(module_name="train_diffusion")
use_mlflow = init_mlflow(cfg)
checkpoint_path = config_value(cfg, "diffusion", "checkpoint", value_type=Path)
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

train_kwargs = build_supervised_train_kwargs(
    cfg,
    model_key="diffusion",
    dataset_root=dataset_root,
    checkpoint_path=checkpoint_path,
    device=device,
    seed=seed,
    learning_rate=learning_rate,
    num_epochs=num_epochs,
    batch_size=batch_size,
)

run_with_optional_mlflow(
    use_mlflow,
    experiment,
    supervised_mlflow_log_params(
        cfg,
        device=device,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
    ),
    run_diffusion_training_pipeline,
    **train_kwargs,
)

# %% [markdown]
# ## 5. Results

# %%
print_checkpoint_summary(checkpoint_path, config_value(cfg, "diffusion", "tensorboard", value_type=Path))
