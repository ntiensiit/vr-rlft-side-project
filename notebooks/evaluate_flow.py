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
from notebook_helpers import load_notebook_context, prepare_context_dataset, run_generative_eval_notebook

ctx = load_notebook_context(
    root,
    "config",
    overrides=["evaluation=flow", "training=flow", "model=flow"],
    config_labels={"flow": "evaluation/flow"},
)

# %% [markdown]
# ## 3. Data and checkpoint validation

# %%
paths = prepare_context_dataset(ctx)

# %% [markdown]
# ## 4. Flow inference and evaluation

# %%
run_generative_eval_notebook(ctx, method="flow", dataset_paths=paths)
