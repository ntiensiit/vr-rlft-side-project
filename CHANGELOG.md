# Changelog

All notable changes to this repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(development pre-release).

**Rationale:** [`docs/adr/`](docs/adr/README.md). This file is the user-visible summary only.

## [Unreleased]

### Added

- ADRs [0001](docs/adr/001-phase4-canonicalization-vs-equivariant.md)–[0005](docs/adr/005-runtime-workflow-integration.md); CI; artifact-chain script (`run_artifacts.py`) and runtime workflow scripts — [ADR-0005](docs/adr/005-runtime-workflow-integration.md).
- Flow training/inference (`FlowGeneratorModel`, `scripts/train_flow.py`).
- Shared modules from dedup/refactor: `grasp_vector`, `training_pairs`, `grasp_sampling`, `checkpoint_io`, `supervised_training`, `grasp_sampling_batch`, `grasp_inference_runtime` — [ADR-0004](docs/adr/004-dead-helper-wiring-and-refactoring.md), [ADR-0005](docs/adr/005-runtime-workflow-integration.md).
- Training flags `--augment` / `--resume`; unified grasp I/O (`load_generated_grasps`); `tests/test_grasp_io_runtime.py`.

### Changed

- Analytical metric `lift_success` → `grasp_success`; 9D grasp representation in configs.
- Dead helpers wired; duplicate training/checkpoint/SE(3) paths consolidated — [ADR-0004](docs/adr/004-dead-helper-wiring-and-refactoring.md).
- Dual inference CLIs share runtime helpers; artifact and runtime grasp formats reconciled — [ADR-0005](docs/adr/005-runtime-workflow-integration.md).
- Simulation pre-grasp open phase; docs/README for CLI contracts and YAML-as-docs-only.

### Removed

- Empty repo placeholders; obsolete notebooks (`notebooks/archive/README.md`); dead wrappers and test-only public exports from refactor audit — [ADR-0004](docs/adr/004-dead-helper-wiring-and-refactoring.md).

### Fixed

- Training dataloader re-instantiation; RL env object propagation; simulation judges; artifact-chain object identity; flow encoder checkpoint contract — [ADR-0003](docs/adr/003-flow-checkpoint-joint-encoder.md).
- GitHub CI pytest step on Linux: install Open3D/Mesa/EGL system libraries, use xvfb, fix hatch package install, sync dev dependency group explicitly; per-file coverage runner to avoid headless segfault (exit 139).

## [0.1.0] - Earlier implementation phases

Phase 1–10: geometry, MuJoCo/YCB simulation, dataset contract, diffusion grasp
model, Gymnasium + SB3 RL, analytical evaluation, experiment tracking, CLI
orchestration. See `git log` for per-commit history.
