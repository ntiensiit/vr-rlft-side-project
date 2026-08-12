# ADR-0005 — Runtime workflow integration scripts

## Status

Accepted (2026-08-12). Updated 2026-08-13 for inference deduplication and grasp I/O.

## Context

Library modules, training pipelines, and an artifact-chain verifier
(`scripts/run_artifacts.py`) were in place, but standalone runtime CLIs for
single-object grasp generation, simulation, evaluation, RL rollout, and composed
end-to-end workflows were missing.

`scripts/run_artifacts.py` remains the **CPU-friendly artifact-chain smoke test**
(reproducible supervised + RL chain, `artifacts/manifest.json`). The runtime
workflow scripts below are **entry points** for single-object or composed runs.

## Decision

Implement the following scripts under `scripts/` using `PYTHONPATH=src` (or
equivalent) like existing CLIs.

| Script | Role |
| --- | --- |
| `generate_grasps.py` | Multi-object artifact-chain inference; pickled dict output via `write_generated_grasps`. |
| `run_grasp_inference.py` | Single-object runtime inference; plain `(num_grasps, 4, 4)` `.npy` (no pickle). |
| `run_simulation.py` | MuJoCo grasp simulation; optional `--grasp-pose-format` (default `world`); rejects unsupported formats with clear `ValueError`. |
| `evaluate.py` | Analytical grasp evaluation; accepts plain array or dict-on-disk via `load_generated_grasps`. |
| `run_rl_evaluation.py` | Deterministic RL policy rollout; same `MuJoCoGraspingEnv` as `train_rl.py`; clips to actuator bounds; optional `--observation-dim-from-env` / `--action-dim-from-env`; `--stochastic` uses `select_action`. |
| `run_workflow.py` | Orchestrates inference → simulation → evaluation (optional RL rollout); writes artifacts under `--output-dir`. |
| `print_model_info.py` | Prints checkpoint metadata (`feature_dim`, `hidden_dim`, `num_layers`, RL dims) via `training/checkpoint_io.read_model_checkpoint_metadata`. |

Shared inference logic lives in `inference/grasp_inference_runtime.py`
(checkpoint load, point-cloud resolve, single-object generate). Both inference
CLIs are retained with distinct output contracts (see table above).

### Contract highlights

- **Grasp inference:** two entry points, one shared runtime module.
  - Artifact chain: multi-object dict on disk (`generate_grasps.py`).
  - Runtime workflow: single-object plain array (`run_grasp_inference.py`).
- **Grasp I/O:** `load_generated_grasps` / `write_generated_grasps_array` in
  `pipelines/generate_grasps.py` reconcile both formats for evaluation and
  extraction scripts.
- **Simulation** accepts world-frame poses only; the format flag documents the
  contract for future callers.
- **Workflow orchestrator** subprocesses the individual scripts (same pattern as
  `run_artifacts.py`) rather than importing pipeline internals directly.

## Rationale

- Separates **verification** (artifact chain, small hyperparameters) from
  **runtime** (user-supplied checkpoints, single-object focus).
- Plain numpy grasp arrays avoid pickle security/friction in downstream tools.
- `print_model_info.py` reduces misconfiguration before expensive inference runs.

## Consequences

- Runtime workflow integration is complete; inference deduplication closed (2026-08-13).
- Do not merge the two inference CLIs without updating artifact-chain and workflow
  callers; extend shared runtime helpers instead.
- New runtime stages should extend `run_workflow.py` or add a sibling script;
  keep `run_artifacts.py` as the regression gate unless the artifact chain
  contract changes.
- Checkpoint metadata discovery is centralized in `checkpoint_io.py`; CLI is
  optional sugar.

## Follow-up review triggers

Revisit when:

- flow becomes the default artifact-chain path (see ADR-0002);
- multi-object workflow orchestration is required;
- checkpoint schema gains fields that `print_model_info` must surface.
