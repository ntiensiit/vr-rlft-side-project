# Changelog

All notable changes to this repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(development pre-release).

**Rationale:** [`docs/adr/`](docs/adr/README.md). This file is the user-visible summary only.

## [Unreleased]

### Added

- Google Colab–ready research notebooks (`notebooks/README.md`) for diffusion, flow, RL training, and combined evaluation; data prep reuses `scripts/` entry points.
- ADRs [0001](docs/adr/001-phase4-canonicalization-vs-equivariant.md)–[0005](docs/adr/005-runtime-workflow-integration.md), [0007](docs/adr/007-wandb-artifact-versioning.md)–[0009](docs/adr/009-panda-contact-frame.md); CI; artifact-chain script (`run_artifacts.py`) and runtime workflow scripts — [ADR-0005](docs/adr/005-runtime-workflow-integration.md).
- Hydra config composition (`hydra-core`, `configs/project.yaml`, `configs/scripts/`) — [ADR-0008](docs/adr/008-hydra-configuration.md).
- Panda sim fidelity: contact-to-hand transform, width-to-joint mapping, fingertip friction — [ADR-0009](docs/adr/009-panda-contact-frame.md).
- `scripts/visualize_robot.py` passive MuJoCo viewer (MuJoCo built-in viewer UI).
- Flow training/inference (`FlowGeneratorModel`, `scripts/train_flow.py`).
- Shared modules from dedup/refactor: `grasp_vector`, `training_pairs`, `grasp_sampling`, `checkpoint_io`, `supervised_training`, `grasp_sampling_batch`, `grasp_inference_runtime`, `mlp`, `supervised_training_script` — [ADR-0004](docs/adr/004-dead-helper-wiring-and-refactoring.md), [ADR-0005](docs/adr/005-runtime-workflow-integration.md).
- Training keys `training.augment` / `training.resume`; unified grasp I/O (`load_generated_grasps`); `tests/test_grasp_io_runtime.py`.
- Per-script Hydra entrypoints under `configs/scripts/`; library composition root `configs/project.yaml`; config groups `artifacts/`, `workflow/`, `notebook/`, `object/`.
- RL checkpoint schema fields `format_version`, `architecture`, and explicit policy dims; checkpoint kind detection in `read_model_checkpoint_metadata`.

### Changed

- All CLI scripts use `@hydra.main` (`config_path=../configs`, `config_name=scripts/<script>`). Argparse flags and `grasping_ai.config.yaml_loader` are gone; overrides are Hydra (`seed=42`, `script.object_id=…`).
- `FlattenedYAMLConfig` is the typed accessor for composed Hydra trees (`script_or` looks up `script.*` then domain keys).
- Grasp dataset records migrated from pickled `.npy` dicts to pickle-free `.npz` archives.
- `configs/gripper/` split into `franka_emika_panda.yaml`; `default.yaml` aliases Franka — [ADR-0008](docs/adr/008-hydra-configuration.md).
- `configs/model/` split into `diffusion.yaml` and `flow.yaml`; `default.yaml` aliases diffusion; shared `grasp.yaml` for common grasp representation — [ADR-0008](docs/adr/008-hydra-configuration.md).
- Config deduplication: artifact paths via `${paths.*}` / `${artifacts.*}`; synthetic and RL keys interpolate from gripper/metrics; object export filenames use `${objects.ids.0}` — [ADR-0008](docs/adr/008-hydra-configuration.md).
- Training and evaluation notebook entrypoints live in `configs/training/{diffusion,flow}.yaml` and `configs/evaluation/{diffusion,flow,rl}.yaml`.
- `configs/training/` split into `diffusion.yaml` and `flow.yaml`; `default.yaml` aliases diffusion; shared `supervised.yaml` hyperparameters.
- `configs/evaluation/default.yaml` holds shared metrics/limits; method variants `diffusion.yaml`, `flow.yaml`, `rl.yaml`.
- Analytical metric `lift_success` → `grasp_success`; 9D grasp representation in configs.
- Dead helpers wired; duplicate training/checkpoint/SE(3) paths consolidated — [ADR-0004](docs/adr/004-dead-helper-wiring-and-refactoring.md).
- Dual inference CLIs share runtime helpers; artifact and runtime grasp formats reconciled — [ADR-0005](docs/adr/005-runtime-workflow-integration.md).
- Diffusion training entry points renamed (`scripts/train_diffusion.py`, `pipelines/train_diffusion.py`); artifact outputs use method-specific filenames (`diffusion_grasp_generator.pt`, `flow_grasp_generator.pt`, `rl_grasp_policy.pt`, `{method}_grasp_candidates.npy`, etc.).
- Pipeline imports and package re-exports slimmed; supervised CLI helpers moved to `pipelines/supervised_training_script.py`.
- Franka Emika Panda replaces toy 2-DOF arm in `deploy/robot.xml`; robot viewing is consolidated in `pipelines/visualize_robot.py`.

### Removed

- Empty repo placeholders; obsolete notebooks (`notebooks/archive/README.md`); dead wrappers and test-only public exports from refactor audit — [ADR-0004](docs/adr/004-dead-helper-wiring-and-refactoring.md).
- `run_simulation.py --render`; `pipelines/control_robot.py`, `grasping_ai.control` package, `GraspCommandPlayer`, and unused table-pick automation.
- Argparse CLIs, custom `--config-dir` loader, and UDP keyboard-topic robot TUI (viewer uses MuJoCo's built-in UI).
- `scripts/_supervised_training.py` (replaced by `pipelines/supervised_training_script.py`).

### Fixed

- Training dataloader re-instantiation; RL env object propagation; simulation judges; artifact-chain object identity; flow encoder checkpoint contract — [ADR-0003](docs/adr/003-flow-checkpoint-joint-encoder.md).
- GitHub CI pytest on Linux: Mesa/EGL/xvfb, hatch package path, Open3D-before-NumPy in data-prep scripts; pin `torch`/`torchvision` to CPU wheels (no `triton`) so `Adam` → `_disable_dynamo` does not segfault on headless runners. CI also uninstalls `triton` if present.
- Code-quality pass: batched torch SE(3) inversion; fragile-pattern fixes; TypedDict export; generative-model `condition()` parity; shared MLP builders; `SupervisedTrainingStep`; MLflow logging helpers; antipodal grasp dedup; dynamic SB3 policy weight export; configurable RL PPO hyperparameters; `GraspPoseGenerator` protocol; package-wide `from __future__ import annotations`; RL `PolicyNetwork` typing cleanup; float32 grasp transforms in supervised training pairs; encoder typed as `SE3EquivariantPointNet`; I/O boundary validators replacing boundary `cast()` calls; shared numerics constants and path validation helpers.

## [0.1.0] - Earlier implementation phases

Phase 1–10: geometry, MuJoCo/YCB simulation, dataset contract, diffusion grasp
model, Gymnasium + SB3 RL, analytical evaluation, experiment tracking, CLI
orchestration. See `git log` for per-commit history.
