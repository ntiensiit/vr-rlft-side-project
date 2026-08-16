# Archived notebooks

Legacy Colab-style notebooks predated the maintained script/pipeline entry points and are kept for historical reference only.

## Active notebooks (Google Colab)

Research notebooks live in the parent `notebooks/` directory. Each follows a standard layout (environment → configuration → data via `scripts/` → experiment via `grasping_ai.pipelines.*` → results).

| Notebook | CLI equivalent |
| --- | --- |
| `../train_diffusion.py` | `scripts/train_diffusion.py` |
| `../train_flow.py` | `scripts/train_flow.py` |
| `../train_rl.py` | `scripts/train_rl.py` |
| `../evaluate_diffusion.py`, `../evaluate_flow.py` | `scripts/run_grasp_inference.py`, `evaluate.py`, `run_simulation.py` |
| `../evaluate_rl.py` | `scripts/run_rl_evaluation.py` |

See [`../README.md`](../README.md) for Colab quick start.

## Legacy sources (removed 2026-08-13)

| Archived notebook | Replacement |
| --- | --- |
| `equivariant_diffusion_grasp.py` | `scripts/train_diffusion.py`, `scripts/prepare_data.py` |
| `kinematic_flow_grasp.py` | `scripts/train_flow.py` |
| `rl_grasp_policy.py` | `scripts/train_rl.py` |
