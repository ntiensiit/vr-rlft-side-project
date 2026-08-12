# Grasping

The grasping subsystem maps object point clouds to candidate grasp poses
and evaluates them offline (analytical) and online (MuJoCo).

## Active generation paths

| Path | Training entry point           | Inference entry point                  |
| ---- | ------------------------------ | -------------------------------------- |
| Diffusion score matching | `scripts/train.py`     | `scripts/generate_grasps.py` or `scripts/run_grasp_inference.py --method diffusion` |
| Flow matching            | `scripts/train_flow.py` | `scripts/run_grasp_inference.py --method flow` |

## Evaluation paths

| Path | Entry point                              | Semantics |
| ---- | ---------------------------------------- | --------- |
| Offline analytical | `scripts/evaluate.py`           | `grasp_success` = collision-free ∧ force-closure (analytical proxy, **not** a physical lift). |
| MuJoCo simulation  | `scripts/run_simulation.py`      | `success` = contact ≥ 1 ∧ simulated lift ∧ bounded object velocity. |

For full architectural detail, see `docs/architecture.md`.