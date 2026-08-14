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
     - robot: default
     - simulation: default
     - training: default
     - evaluation: default
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
