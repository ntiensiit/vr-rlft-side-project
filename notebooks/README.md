# Research notebooks (Google Colab)

Percent-format notebooks (`# %%` / `# %% [markdown]`) for interactive diffusion, flow, and RL experiments.

Shared setup, config loading, data prep, and training helpers live in [`notebook_helpers.py`](notebook_helpers.py); [`bootstrap.py`](bootstrap.py) handles Colab clone and import-path setup for the first cell.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| [`train_diffusion.py`](train_diffusion.py) | Supervised score-based diffusion grasp training |
| [`train_flow.py`](train_flow.py) | Supervised flow-matching grasp training |
| [`train_rl.py`](train_rl.py) | PPO RL policy training in MuJoCo |
| [`evaluate.py`](evaluate.py) | Diffusion/flow inference, analytical metrics, simulation, RL rollouts |

Notebooks call existing `scripts/` entry points for data preparation and `grasping_ai.pipelines.*` for training and evaluation.

## Configuration

Full Hydra compositions live in the matching config group files. Each notebook sets only `CONFIG_NAME`:

| Notebook | Config entrypoint |
| --- | --- |
| `train_diffusion.py` | `configs/training/diffusion.yaml` |
| `train_flow.py` | `configs/training/flow.yaml` |
| `train_rl.py` | `configs/rl/rl_train.yaml` |
| `evaluate.py` | `evaluation/diffusion`, `evaluation/flow`, `evaluation/rl` |

Data-prep subprocesses receive the same `--config-name` so scripts load an identical composed tree.

Edit the ``notebook`` block in each entrypoint YAML for notebook-only flags (`download_ycb`, `augment`, `object_index`, `experiment`, `mount_drive`). Set ``notebook.mount_drive: true`` on Colab to persist ``data/`` and ``artifacts/`` under ``MyDrive/<drive_storage_dir>/`` across sessions. Hyperparameters and paths come from the composed config. Override at runtime with Hydra, e.g. ``load_project_yaml_config(CONFIG_DIR, config_name=CONFIG_NAME, overrides=["notebook.augment=true"])``.

## Colab quick start

1. **Runtime → Change runtime type → GPU** (optional, recommended for training).
2. Run cells top-to-bottom; do not skip the environment section.
3. Edit ``notebook.*`` keys in the matching config YAML before long jobs. Enable ``notebook.mount_drive`` on Colab to store datasets and artifacts on Google Drive.
4. Artifacts follow ``paths.output_dir`` in the composed config (default ``artifacts/``, or Drive-backed when ``mount_drive`` is enabled).

## Local VS Code / Cursor

Open any notebook `.py` file with the Jupyter extension. The environment cell detects a local checkout and skips cloning.

## Script equivalents

| Notebook | CLI |
| --- | --- |
| `train_diffusion.py` | `python scripts/train_diffusion.py --config-name training/diffusion` |
| `train_flow.py` | `python scripts/train_flow.py --config-name training/flow` |
| `train_rl.py` | `python scripts/train_rl.py --config-name rl/rl_train` |
| `evaluate.py` | combine inference/eval scripts with matching `--config-name` |

Legacy Colab notebooks are archived under [`archive/`](archive/README.md).
