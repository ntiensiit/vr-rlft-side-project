# ADR-0008 — Hydra configuration composition

## Status

Accepted (2026-08-14)

## Context

CLI scripts loaded flat YAML files (`base.yaml`, `data.yaml`, `model.yaml`, …)
through a custom deep-merge implementation in
`grasping_ai.config.yaml_loader`. Each script selected its own layer subset,
and runtime overrides required explicit argparse flags per parameter.

Experiment matrices (swap model type, sweep hyperparameters) needed a
standard override syntax and a single composable config entrypoint.

## Decision

1. Adopt **Hydra** (`hydra-core`) as the configuration composition engine.
2. Restructure `configs/` into Hydra **config groups** with
   `# @package _global_` so keys remain at the top level (no nesting under
   group names).
3. Add `configs/config.yaml` as the sole entrypoint:

   ```yaml
   defaults:
     - base
     - model: default
     - data: default
     - gripper: default
     - env: default
     - training: default
     - evaluation: diffusion
     - _self_
   ```

4. Rewrite `load_project_yaml_config` to compose via Hydra when
   `config.yaml` is present; retain the legacy flat-file merge path for
   partial test directories without `config.yaml`.
5. Add `parse_config_overrides_from_argv` so scripts accept Hydra-style
   overrides (`seed=100`, `supervised.learning_rate=0.0005`) without
   per-flag argparse entries.
6. Keep `config_get`, `config_path`, and related accessors on plain dicts so
   pipeline code is unchanged.

## Rationale

- Hydra provides hierarchical defaults, CLI overrides, and config groups
  without replacing the project's procedural script structure.
- `@package _global_` preserves existing key paths (`cfg["diffusion"]`, etc.).
- Compose API works in tests without `@hydra.main` decorators on every script.
- MLflow was not required; Hydra addresses composition and overrides only.

## Consequences

- Flat group files (`configs/data.yaml`, etc.) are replaced by
  `configs/<group>/default.yaml`.
- Scripts continue calling `load_project_yaml_config`; no `@hydra.main` wrapper
  on each entrypoint.
- New experiment variants add files such as `configs/model/flow.yaml` and
  override with `model=flow`.
- `pyyaml` remains for direct file reads in unit tests of merge helpers.

## Follow-up review triggers

Revisit when:

- structured dataclass configs are needed for type-safe access;
- a hosted experiment registry (W&B artifacts, ADR-0007) must link to Hydra
  run IDs automatically;
- config groups grow large enough to split by environment (dev/prod).

## Follow-up (2026-08-14)

Unified config layout to match the ADR-0008 group names and remove split keys:

- `configs/gripper/` holds robot MJCF, gripper, and IK settings (top-level key `robot:`).
- `configs/env/` holds MuJoCo timestep and rollout length (`dt`, `num_steps`).
- All `paths` consolidated under `configs/data/default.yaml` with `${paths.input_dir}` interpolation; removed from `base.yaml`.
- All `rl` settings consolidated under `configs/rl/default.yaml` (removed from `model/` and `training/`).
- `configs/model/flow.yaml` exports aligned with diffusion export paths.
- CLI scripts call `load_project_yaml_config(config_dir)` without redundant layer lists when `config.yaml` is present.
- `configs/gripper/default.yaml` split into `configs/gripper/franka_emika_panda.yaml`; `default.yaml` defaults to Franka.
- `configs/model/default.yaml` split into `configs/model/diffusion.yaml` and `configs/model/flow.yaml`; `default.yaml` defaults to diffusion.
- Config deduplication: shared `configs/model/grasp.yaml`; artifact paths use `${paths.checkpoints|exports|reports|tensorboard}`; synthetic `gripper_width`, friction/collision, and RL `lift_height_threshold` interpolate from gripper/metrics; object-specific export filenames use `${objects.ids.0}`; removed unused `diffusion.train_steps`.
- `configs/training/default.yaml` split into `configs/training/diffusion.yaml` and `configs/training/flow.yaml`; `default.yaml` defaults to diffusion.
- `configs/evaluation/default.yaml` holds shared metrics/limits only; `diffusion.yaml`, `flow.yaml`, and `rl.yaml` compose `default` for shared keys and serve as full notebook/CLI entrypoints.
- Training presets merged into `configs/training/diffusion.yaml` and `configs/training/flow.yaml`; evaluation presets merged into `configs/evaluation/diffusion.yaml`, `flow.yaml`, and `rl.yaml`. Load via `config_name="group/name"` (e.g. `training/diffusion`, `evaluation/flow`).
