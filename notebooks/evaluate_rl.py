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

# %%
from notebook_helpers import load_notebook_context, run_rl_eval_notebook

ctx = load_notebook_context(
    root,
    "config",
    overrides=["evaluation=rl", "training=diffusion"],
    config_labels={"rl": "evaluation/rl"},
)

# %% [markdown]
# ## 3. RL policy rollouts

# %%
run_rl_eval_notebook(ctx)
