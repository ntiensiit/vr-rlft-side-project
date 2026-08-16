# Changelog

All notable changes to this repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(development pre-release).

**Rationale:** [`docs/adr/`](docs/adr/README.md). This file is the user-visible summary only.

## [Unreleased]

### Added

- Google Colab–ready research notebooks (`notebooks/README.md`) for diffusion, flow, RL training, and combined evaluation; data prep reuses `scripts/` entry points.
- ADRs [0001](docs/adr/001-phase4-canonicalization-vs-equivariant.md)–[0008](docs/adr/008-hydra-configuration.md); CI; artifact-chain script (`run_artifacts.py`) and runtime workflow scripts — [ADR-0005](docs/adr/005-runtime-workflow-integration.md).
- Hydra config composition (`hydra-core`, `configs/config.yaml`, config groups) — [ADR-0008](docs/adr/008-hydra-configuration.md).
- Panda sim fidelity: contact-to-hand transform, width-to-joint mapping, fingertip friction — [ADR-0009](docs/adr/009-panda-contact-frame.md).
- `scripts/visualize_robot.py` passive MuJoCo viewer.
- Flow training/inference (`FlowGeneratorModel`, `scripts/train_flow.py`).
- Shared modules from dedup/refactor: `grasp_vector`, `training_pairs`, `grasp_sampling`, `checkpoint_io`, `supervised_training`, `grasp_sampling_batch`, `grasp_inference_runtime` — [ADR-0004](docs/adr/004-dead-helper-wiring-and-refactoring.md), [ADR-0005](docs/adr/005-runtime-workflow-integration.md).
- Training flags `--augment` / `--resume`; unified grasp I/O (`load_generated_grasps`); `tests/test_grasp_io_runtime.py`.

### Changed

- `configs/gripper/` split into `franka_emika_panda.yaml`; `default.yaml` aliases Franka — [ADR-0008](docs/adr/008-hydra-configuration.md).
- `configs/model/` split into `diffusion.yaml` and `flow.yaml`; `default.yaml` aliases diffusion; shared `grasp.yaml` for common grasp representation — [ADR-0008](docs/adr/008-hydra-configuration.md).
- Config deduplication: artifact paths via `${paths.*}`; synthetic and RL keys interpolate from gripper/metrics; object export filenames use `${objects.ids.0}` — [ADR-0008](docs/adr/008-hydra-configuration.md).
- Training and evaluation notebook entrypoints merged into `configs/training/{diffusion,flow}.yaml` and `configs/evaluation/{diffusion,flow,rl}.yaml`; load via `--config-name group/name` — [ADR-0008](docs/adr/008-hydra-configuration.md).
- `configs/training/` split into `diffusion.yaml` and `flow.yaml`; `default.yaml` aliases diffusion — [ADR-0008](docs/adr/008-hydra-configuration.md).
- `configs/evaluation/default.yaml` holds shared metrics/limits; method variants `diffusion.yaml`, `flow.yaml`, `rl.yaml` — [ADR-0008](docs/adr/008-hydra-configuration.md).
- Analytical metric `lift_success` → `grasp_success`; 9D grasp representation in configs.
- Dead helpers wired; duplicate training/checkpoint/SE(3) paths consolidated — [ADR-0004](docs/adr/004-dead-helper-wiring-and-refactoring.md).
- Dual inference CLIs share runtime helpers; artifact and runtime grasp formats reconciled — [ADR-0005](docs/adr/005-runtime-workflow-integration.md).
- Diffusion training entry points renamed (`scripts/train_diffusion.py`, `pipelines/train_diffusion.py`); artifact outputs use method-specific filenames (`diffusion_grasp_generator.pt`, `flow_grasp_generator.pt`, `rl_grasp_policy.pt`, `{method}_grasp_candidates.npy`, etc.).
- `configs/` restructured as Hydra config groups (`config.yaml` entrypoint, `<group>/default.yaml`); `compose_config` composes via Hydra with CLI override support (`seed=100`) — [ADR-0008](docs/adr/008-hydra-configuration.md).
- Flat config files (`configs/data.yaml`, `configs/model.yaml`, etc.) replaced by group defaults under `configs/<group>/default.yaml`.
- Franka Emika Panda replaces toy 2-DOF arm in `deploy/robot.xml`; robot viewing is consolidated in `pipelines/visualize_robot.py`.

### Removed

- Empty repo placeholders; obsolete notebooks (`notebooks/archive/README.md`); dead wrappers and test-only public exports from refactor audit — [ADR-0004](docs/adr/004-dead-helper-wiring-and-refactoring.md).
- `run_simulation.py --render`; `pipelines/control_robot.py`, `grasping_ai.control` package, `GraspCommandPlayer`, and unused table-pick automation.

### Fixed

- Training dataloader re-instantiation; RL env object propagation; simulation judges; artifact-chain object identity; flow encoder checkpoint contract — [ADR-0003](docs/adr/003-flow-checkpoint-joint-encoder.md).
- GitHub CI pytest on Linux: Mesa/EGL/xvfb, hatch package path, Open3D-before-NumPy in data-prep scripts; pin `torch`/`torchvision` to CPU wheels (no `triton`) so `Adam` → `_disable_dynamo` does not segfault on headless runners.
- Code-quality pass: batched torch SE(3) inversion; fragile-pattern fixes; TypedDict export; generative-model `condition()` parity; shared MLP builders; `SupervisedTrainingStep`; MLflow logging helpers; antipodal grasp dedup; dynamic SB3 policy weight export; configurable RL PPO hyperparameters; `GraspPoseGenerator` protocol; package-wide `from __future__ import annotations`; RL `PolicyNetwork` typing cleanup; float32 grasp transforms in supervised training pairs; encoder typed as `SE3EquivariantPointNet`; I/O boundary validators replacing boundary `cast()` calls; shared numerics constants and path validation helpers.

## [0.1.0] - Earlier implementation phases

Phase 1–10: geometry, MuJoCo/YCB simulation, dataset contract, diffusion grasp
model, Gymnasium + SB3 RL, analytical evaluation, experiment tracking, CLI
orchestration. See `git log` for per-commit history.
