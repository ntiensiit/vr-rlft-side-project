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
# Edit notebook flags in ``configs/*/…yaml`` under the ``notebook`` key.

# %%
from grasping_ai.pipelines.train_diffusion import run_diffusion_training_pipeline
from notebook_helpers import load_notebook_context, prepare_context_dataset, run_supervised_notebook

ctx = load_notebook_context(root, "training/diffusion")

# %% [markdown]
# ## 3. Data preparation
#
# Uses the same script entry points as `scripts/run_artifacts.py`.

# %%
paths = prepare_context_dataset(ctx)

# %% [markdown]
# ## 4. Training

# %%
run_supervised_notebook(
    ctx,
    method="diffusion",
    pipeline=run_diffusion_training_pipeline,
    dataset_root=paths["dataset_root"],
)
