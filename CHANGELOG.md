# Changelog

All notable changes to this repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(development pre-release).

## [Unreleased]

See the ADR directory for the canonical record of architectural decisions,
scope changes, and design rationale that drove these changes:

- [`docs/adr/001-phase4-canonicalization-vs-equivariant.md`](docs/adr/001-phase4-canonicalization-vs-equivariant.md)
  — Phase 4 representation: canonicalization + invariant features (trivial
  feature action) accepted; genuine equivariance requires encoder replacement.
- [`docs/adr/002-flow-training-optional-vs-required.md`](docs/adr/002-flow-training-optional-vs-required.md)
  — Flow-matching training is an optional research extension; the default
  artifact chain trains the diffusion path via `scripts/run_artifacts.py`.
- [`docs/adr/003-flow-checkpoint-joint-encoder.md`](docs/adr/003-flow-checkpoint-joint-encoder.md)
  — Flow path mirrors the diffusion `GraspGeneratorModel` pattern: encoder
  and flow field live on one `FlowGeneratorModel`, jointly optimized,
  persisted together in the checkpoint.

### Added

- ADR directory `docs/adr/` with index and convention (`docs/adr/README.md`).
- `FlowGeneratorModel` (`src/grasping_ai/models/flow.py`) owns encoder + flow field jointly.
- `load_flow_model_checkpoint` (`src/grasping_ai/pipelines/train_flow.py`) reconstructs the model from a joint checkpoint.
- `scripts/run_artifacts.py` — reproducible supervised + RL artifact runner; writes `artifacts/manifest.json`; keeps `003_cracker_box` end-to-end; declares all three YCB objects as required.
- `scripts/train_flow.py` — CLI for `run_flow_training_pipeline`.
- `scripts/prepare_ycb_mjcf.py`, `scripts/prepare_observations.py`, `scripts/extract_object_grasps.py` — helper artifact scripts.
- CI workflow `.github/workflows/ci.yml` (Ruff, Mypy, `pytest -m "not slow" --cov --cov-fail-under=80`).
- Artifact-chain CI test `tests/test_artifact_chain.py` (`@pytest.mark.slow`).
- Synthetic-data fail-fast: `generate_synthetic_dataset(required_objects=...)` and `--required-objects` CLI flag.
- `--output-index` filename honored (`save_grasp_dataset_index(..., filename=...)`).
- `build_rl_policy_runner(action_low=, action_high=)` clips actions to actuator bounds.
- `docs/architecture.md`, `docs/dataset.md`, `docs/experiments.md`, `docs/grasping.md`, `docs/simulation.md`, `deploy/{Dockerfile,docker-compose.yaml,run.sh}`.
- `slow` test marker (conftest.py + pyproject.toml) and ADR-maintenance rule (`.agents/rules/adr-maintenance.md`).
- Joint-encoder flow regression tests: `test_flow_checkpoint_persists_encoder_and_flow_field`, `test_load_flow_model_checkpoint_reproduces_trained_state`, `test_flow_training_optimizes_encoder_and_flow_field`.

### Changed

- Phase 3 analytical `lift_success` -> `grasp_success` rename in `src/grasping_ai/pipelines/evaluate.py` (and its `aggregate_*` helper). Distinct from the simulated lift signal.
- `configs/model.yaml::grasp_dim` corrected from `7` to `9`; new `grasp_representation: 6d_rotation_columns` (translation 3 + first two rotation columns 6).
- Phase 8 / Phase 9 docs annotate the old custom RL trainer as historical.
- `docs/USAGE.md` synchronized with current SB3/Gymnasium architecture (removed `deploy/gripper.xml`, replaced stale dims with shipped-robot `21/4`, added helper scripts, redirected workflow to `scripts/run_artifacts.py`).
- MJCF wrappers use absolute mesh paths in `<mesh file=...>` (regenerable from source, not relocatable; rationale documented in `scripts/prepare_ycb_mjcf.py`).
- Notebooks `equivariant_diffusion_grasp.py`, `kinematic_flow_grasp.py`, `rl_grasp_policy.py` carry `OBSOLETE` banners pointing to maintained entry points.

### Removed

- Empty placeholder files `.env.example`, `data/README.md`, `experiments/README.md`, `logs/README.md`.

### Fixed

- Supervised training iterator bug (`src/grasping_ai/pipelines/train.py` passes reusable dataloader; `run_training_loop` refreshes per epoch).
- RL environment object propagation (`object_ids[0]` -> `MuJoCoGraspingEnv`; `RewardConfig` enabled; `ValueError` when `len(object_ids) > 1`).
- Simulation grasp-success contract (`simulate_grasp.py` uses `build_lift_outcome_judge` / `build_stability_judge` with `lift_height_threshold`, `max_linear_velocity`, `max_angular_velocity`).
- Artifact chain object-identity mismatch (`scripts/run_artifacts.py` evaluation step now uses `--object-id object_0` / cracker-box point cloud).
- Flow train/inference inconsistency (encoder + flow field now jointly trained and persisted; see ADR-0003).

### Verified

- `scripts/run_artifacts.py` end-to-end run on a clean tree (`data/processed/` and `artifacts/` removed): exit code 0, 44.9s, 9 commands, 13 retained artifacts covering supervised training, RL training, MuJoCo simulation, analytical evaluation, and policy_runner inference. The chain keeps the same object identity end-to-end (`003_cracker_box` / `object_0`). Result captured in `artifacts/verification_log.md` (gitignored). `tests/test_artifact_chain.py` (slow marker) exercises the same flow as part of the local verification gate.
- Three-stage verification distinction recorded for the project:
  - **Software pipeline correctness — VERIFIED.** The artifact chain
    runs end-to-end on a clean tree.
  - **Learning pipeline execution — VERIFIED mechanically.** Training,
    checkpoint persistence, and inference all execute; the flow path's
    joint encoder/checkpoint contract is verified structurally
    (`tests/test_phase4_flow_training.py`).
  - **Robotics / research outcome — NOT VERIFIED.** The recorded
    verification run produced 8/8 IK failures and 0/0 physical grasp
    successes because the shipped 2-DOF robot cannot reach the
    diffusion model's outputs and the model itself is undertrained.
    This is a research-stage condition, not a pipeline defect. See
    `hot-fix-checklist.md` Priority 13 for the recorded research-stage
    items.

## [0.1.0] - Earlier implementation phases

Phase 1-10 architecture: geometry/SE(3) primitives, MuJoCo/YCB simulation,
dataset contract, synthetic grasp generation, diffusion grasp model,
Gymnasium + SB3 PPO RL, analytical grasp evaluation, experiment tracking,
and CLI orchestration. See `git log` for the per-commit history.