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

_colab_root = Path("/content/vr-rlft-side-project")
_notebooks = None
for _base in (_colab_root, Path.cwd(), Path.cwd().parent):
    _candidate = _base / "notebooks" if (_base / "notebooks" / "bootstrap.py").is_file() else _base
    if (_candidate / "bootstrap.py").is_file():
        _notebooks = _candidate
        break
if _notebooks is None:
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            "dev",
            "--depth",
            "1",
            "https://github.com/ntiensiit/vr-rlft-side-project.git",
            str(_colab_root),
        ],
        check=True,
    )
    _notebooks = _colab_root / "notebooks"
sys.path.insert(0, str(_notebooks))

from bootstrap import bootstrap_notebook

root = bootstrap_notebook()

# %% [markdown]
# ## 2. Experiment configuration
#
# RL hyperparameters and paths come from `configs/rl/default.yaml` and `configs/base.yaml`.

# %%
from notebook_helpers import load_notebook_context, prepare_context_dataset, run_rl_training_notebook

ctx = load_notebook_context(root, "training/rl_train")

# %% [markdown]
# ## 3. Data and simulation assets

# %%
paths = prepare_context_dataset(ctx)

# %% [markdown]
# ## 4. Training

# %%
run_rl_training_notebook(ctx, paths)
