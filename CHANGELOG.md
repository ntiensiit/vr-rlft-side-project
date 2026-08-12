# Changelog

All notable changes to this repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(development pre-release).

**Rationale and caller maps:** see [`docs/adr/`](docs/adr/README.md) (canonical
“why”). This file records the user-visible “what” only.

## [Unreleased]

### Added

- ADR directory and records [0001](docs/adr/001-phase4-canonicalization-vs-equivariant.md)–[0005](docs/adr/005-runtime-workflow-integration.md).
- `FlowGeneratorModel`, `load_flow_model_checkpoint`, `scripts/train_flow.py`.
- Artifact and data helper scripts: `run_artifacts.py`, `prepare_ycb_mjcf.py`, `prepare_observations.py`, `extract_object_grasps.py`.
- Runtime workflow scripts: `run_grasp_inference.py`, `run_rl_evaluation.py`, `run_workflow.py`, `print_model_info.py` — see [ADR-0005](docs/adr/005-runtime-workflow-integration.md).
- CI (`.github/workflows/ci.yml`), slow artifact-chain test, synthetic-data `--required-objects`, RL action clipping, deploy/docs scaffolding.
- Shared modules from refactoring dedup: `data/grasp_vector.py`, `data/training_pairs.py`, `inference/grasp_sampling.py`, `training/checkpoint_io.py` — see [ADR-0004](docs/adr/004-dead-helper-wiring-and-refactoring.md).

### Changed

- Phase 3 `lift_success` → `grasp_success` in analytical evaluation (distinct from simulated lift).
- `configs/model.yaml`: `grasp_dim` 9, `grasp_representation: 6d_rotation_columns`.
- Docs/USAGE, notebooks (OBSOLETE banners), MJCF absolute mesh paths.
- Dead helpers wired to production; loss builders, scene XML temp dir, `__init__.py` exports — [ADR-0004](docs/adr/004-dead-helper-wiring-and-refactoring.md).
- `run_simulation.py` `--grasp-pose-format` — [ADR-0005](docs/adr/005-runtime-workflow-integration.md).
- Dependencies: `theseus` (YCB alias resolution), expanded `pytransform3d` in IK/transforms — [ADR-0004](docs/adr/004-dead-helper-wiring-and-refactoring.md).

### Removed

- Empty placeholders: `.env.example`, `data/README.md`, `experiments/README.md`, `logs/README.md`.

### Fixed

- Training dataloader re-instantiation per epoch (`train.py` / `run_training_loop`).
- RL env object propagation and single-object constraint.
- Simulation grasp-success judges (`lift_height_threshold`, stability thresholds).
- Artifact chain object-identity mismatch in evaluation step.
- Flow train/inference encoder mismatch — [ADR-0003](docs/adr/003-flow-checkpoint-joint-encoder.md).

## [0.1.0] - Earlier implementation phases

Phase 1–10: geometry, MuJoCo/YCB simulation, dataset contract, diffusion grasp
model, Gymnasium + SB3 RL, analytical evaluation, experiment tracking, CLI
orchestration. See `git log` for per-commit history.
