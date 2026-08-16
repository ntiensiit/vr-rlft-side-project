# Phase 8: Experiment Tracking and Reproducibility

> **Historical design record.** This document captures the Phase 8 plan from the skeleton era. Current architecture, CLIs, and contracts live in [architecture.md](architecture.md), [USAGE.md](USAGE.md), and the repository [README.md](../README.md).

## 1. Phase Overview

**Phase name:** Phase 8: Experiment Tracking and Reproducibility

**Objective:** Add explicit, local, file-based experiment tracking and reproducible execution controls to the supervised training, RL training, and evaluation reporting paths without introducing global configuration state, new abstraction layers, helper utilities, or physical-robot dependencies.

**Why this phase is necessary based on verified repository state:**

The repository already contains runnable training and evaluation pipelines, but observability and reproducibility are missing or inconsistent.

Verified evidence:

- `src/grasping_ai/training/trainer.py::run_training_loop` only prints loss using `print`. It does not persist metrics, hyperparameters, or run metadata.
- `src/grasping_ai/training/rl_trainer.py::run_rl_training_loop` accepts `log_every` but does not use it. It discards the metrics dictionary returned by the update step. *(historical: module removed in the Phase 9 SB3 migration; the SB3 path logs via TensorBoard)*
- `pyproject.toml` declares `tensorboard` and `wandb` dependencies, but inspected training, RL, pipeline, and script files do not use either dependency.
- `src/grasping_ai/pipelines/train.py` hardcodes a local dataloader shuffle through `random.seed(42)` inside `TrainingDataloader.__iter__`, mutating global Python random state.
- `src/grasping_ai/pipelines/train.py` and `src/grasping_ai/pipelines/train_rl.py` do not expose seed arguments.
- `scripts/train.py` and `scripts/train_rl.py` do not expose experiment tracking directories or seed arguments.
- `src/grasping_ai/training/rl_trainer.py::build_rl_training_step` uses a hardcoded discount factor of `0.99`, while `src/grasping_ai/pipelines/train_rl.py::run_rl_training_pipeline` accepts and validates a `gamma` argument but does not pass it into the training step. This is a reproducibility and metadata-consistency defect. *(historical: module removed in the Phase 9 SB3 migration; the SB3 PPO path passes gamma directly to the algorithm)*
- `src/grasping_ai/pipelines/evaluate.py::write_evaluation_report` writes JSON metrics but does not emit tracking artifacts.

**Expected outcome:**

After this phase:

- Supervised training can write TensorBoard event artifacts for loss and run metadata.
- RL training can write TensorBoard event artifacts for loss, policy loss, mean reward, and run metadata.
- Evaluation reports can optionally write aggregate metrics to TensorBoard event artifacts.
- Training pipelines accept explicit seed arguments.
- When a seed is provided, supervised training and RL training use deterministic local RNG behavior for repository-controlled randomness.
- Checkpoints include an explicit seed field when a seed is provided.
- Existing pipeline behavior remains available when tracking is disabled.
- No new utility modules, helper functions, helper classes, global variables, or global configuration systems are introduced.

## 2. Verified Current State

### Verified implemented

- **Supervised training step and loop**
  - File: `src/grasping_ai/training/trainer.py`
  - `build_adam_optimizer`, `build_training_step`, `run_training_loop`, `save_training_checkpoint`, and `load_training_checkpoint` are implemented.
  - `build_training_step` returns a metrics dictionary containing `loss`.
  - `run_training_loop` prints loss every `log_every` steps and saves a checkpoint at the end.

- **RL training step and loop**
  - File: `src/grasping_ai/training/rl_trainer.py` *(historical: module no longer exists; the active RL loop is SB3 PPO in `src/grasping_ai/pipelines/train_rl.py`)*
  - `build_rl_training_step`, `run_rl_training_loop`, `compute_discounted_returns`, and `compute_gae_advantages` are implemented.
  - `build_rl_training_step` returns metrics containing `loss`, `policy_loss`, and `mean_reward`.

- **Supervised training pipeline**
  - File: `src/grasping_ai/pipelines/train.py`
  - `run_training_pipeline` discovers dataset records, loads grasp samples, converts SE(3) targets to 9D vectors, constructs the model and optimizer, and calls the supervised training loop.

- **RL training pipeline**
  - File: `src/grasping_ai/pipelines/train_rl.py`
  - `run_rl_training_pipeline` builds a MuJoCo-backed rollout environment, constructs a policy network, and calls the RL training loop.

- **CLI scripts**
  - Files: `scripts/train.py`, `scripts/train_rl.py`, `scripts/evaluate.py`
  - Scripts parse CLI arguments and delegate to pipeline functions.

- **Evaluation report persistence**
  - File: `src/grasping_ai/pipelines/evaluate.py`
  - `write_evaluation_report` writes aggregate evaluation metrics as JSON.

### Partially implemented

- **Logging interval in RL training**
  - File: `src/grasping_ai/training/rl_trainer.py` *(historical: module removed in the SB3 PPO migration)*
  - `run_rl_training_loop` receives `log_every` but does not use it. The parameter exists but has no behavioral effect.

- **Reproducible supervised data shuffling**
  - File: `src/grasping_ai/pipelines/train.py`
  - The local `TrainingDataloader` shuffles indices, but does so by calling `random.seed(42)` inside `__iter__`, mutating global Python random state rather than using an explicitly seeded local generator.

### Missing

- **Experiment tracking integration**
  - No inspected source file imports or uses `tensorboard` or `wandb`.
  - No training or evaluation pipeline writes experiment artifacts except checkpoints and JSON evaluation reports.

- **Explicit seed propagation**
  - Supervised training pipeline, RL training pipeline, CLI scripts, training loops, and checkpoint serializers do not expose or persist an explicit seed argument.

- **Hyperparameter and run metadata logging**
  - Training pipelines do not record feature dimensions, hidden dimensions, learning rates, batch sizes, update counts, gamma values, device settings, dataset roots, or checkpoint destinations in a tracking system.

- **RL gamma propagation**
  - `run_rl_training_pipeline` accepts `gamma`, but `build_rl_training_step` uses a hardcoded `0.99` when computing discounted returns. The user-provided gamma is not used.

### Inconsistent

- **RL gamma argument is accepted but not honored**
  - File: `src/grasping_ai/pipelines/train_rl.py`
  - `gamma` is validated but never passed into `build_rl_training_step`.
  - File: `src/grasping_ai/training/rl_trainer.py` *(historical: module removed; gamma is now passed directly to SB3 PPO)*
  - `build_rl_training_step` calls `compute_discounted_returns(transitions, 0.99)`. *(historical: function from the removed module)*

- **Global RNG mutation inside dataloader**
  - File: `src/grasping_ai/pipelines/train.py`
  - The local dataloader uses `random.seed(42)` during iteration. This conflicts with the repository constraint against global mutable state and makes reproducibility implicit rather than explicit.

### Unused or dead code relevant to this phase

- **`wandb` dependency**
  - File: `pyproject.toml`
  - Declared but not used by inspected training or evaluation code.

- **`tensorboard` dependency**
  - File: `pyproject.toml`
  - Declared but not used by inspected training or evaluation code before this phase.

- **`compute_gae_advantages`**
  - File: `src/grasping_ai/training/rl_trainer.py` *(historical: module removed in the SB3 PPO migration)*
  - Implemented, but inspected RL training step does not use it.

- **`load_pretrained_encoder`**
  - File: `src/grasping_ai/pipelines/train.py`
  - Present but not used by the supervised training pipeline.

## 3. Phase Boundary

### Exact scope of what this phase modifies

This phase modifies only the following responsibilities:

1. TensorBoard-based metric logging in supervised training.
2. TensorBoard-based metric logging in RL training.
3. TensorBoard-based aggregate metric logging for evaluation reports.
4. Explicit seed arguments for supervised training and RL training.
5. Deterministic local RNG usage for repository-controlled data shuffling.
6. Propagation of the user-provided RL discount factor into the RL training step.
7. Optional seed persistence in training checkpoints.
8. CLI exposure of seed and experiment log directory arguments for training and evaluation scripts.

### Explicitly excluded responsibilities

This phase does not implement or modify:

- Weights and Biases integration.
- Hyperparameter search.
- Experiment comparison dashboards.
- Remote artifact storage.
- Training resumption from checkpoints.
- Dataset generation.
- Grasp generation model architecture.
- RL policy architecture.
- MuJoCo physics behavior.
- Reward design beyond preserving existing behavior.
- Force-closure correctness.
- Collision-checking correctness.
- Physical robot deployment.
- Sensor streaming.
- Real hardware integration.
- YAML configuration loading.
- Global configuration management.

### What must NOT be touched

The following must remain unchanged except where explicitly required by signature compatibility:

- Existing checkpoint keys:
  - `epoch`
  - `model_state_dict`
  - `optimizer_state_dict`
  - `feature_dim`
  - `hidden_dim`
  - `num_layers`
- Existing metrics dictionary keys:
  - Supervised: `loss`
  - RL: `loss`, `policy_loss`, `mean_reward`
- Existing data contract expected by `load_grasp_sample`.
- Existing SE(3)-to-9D conversion behavior in supervised training.
- Existing model forward signatures.
- Existing evaluation metric definitions.
- Existing MuJoCo stepping and contact-reporting behavior.
- Existing public function return types.

### MECE boundary definition

This phase owns **observability and reproducibility of existing local pipelines**.

It does not own:

- Data creation, which belongs to synthetic data generation.
- Algorithmic improvement of RL or generative models.
- Evaluation metric correctness beyond logging already-produced metrics.
- Simulation environment expansion.
- Deployment.

## 4. Architecture and Dependency Analysis

### Real dependency flow

Current training flow:

- `scripts/train.py`
  - calls `src/grasping_ai/pipelines/train.py::run_training_pipeline`
    - calls `src/grasping_ai/data/pointcloud_dataset.py` for dataset discovery and loading
    - constructs model components from `src/grasping_ai/models/diffusion.py`
    - constructs loss from `src/grasping_ai/training/losses.py`
    - constructs optimizer and training step from `src/grasping_ai/training/trainer.py`
    - calls `run_training_loop`
      - calls `save_training_checkpoint`

Current RL flow:

- `scripts/train_rl.py`
  - calls `src/grasping_ai/pipelines/train_rl.py::run_rl_training_pipeline`
    - calls `src/grasping_ai/simulation/mujoco_env.py` through environment construction
    - constructs policy from `src/grasping_ai/models/rl_policy.py`
    - constructs `stable_baselines3` PPO against `MuJoCoGraspingEnv`
    - trains the PPO through the Gymnasium environment
    - exports a legacy checkpoint via `src/grasping_ai/models/rl_policy.py::save_rl_policy_checkpoint`

Current evaluation flow:

- `scripts/evaluate.py`
  - calls `src/grasping_ai/pipelines/evaluate.py::evaluate_generated_grasps`
  - calls `aggregate_evaluation_results`
  - calls `write_evaluation_report`

### Input, processing, output chain added by this phase

For training:

- Input:
  - Existing pipeline arguments.
  - Optional explicit seed.
  - Optional experiment log directory.
- Processing:
  - If seed is provided, initialize repository-controlled deterministic RNG state.
  - If experiment log directory is provided, create a TensorBoard writer local to that directory.
  - Log run metadata.
  - Log training metrics during the loop.
  - Save seed into checkpoint when provided.
- Output:
  - Existing checkpoint.
  - Optional TensorBoard event files.
  - Existing stdout behavior for supervised training.

For evaluation:

- Input:
  - Existing evaluation arguments.
  - Optional experiment log directory.
- Processing:
  - Write JSON report.
  - If log directory is provided, write aggregate scalar metrics to TensorBoard.
- Output:
  - Existing JSON report.
  - Optional TensorBoard event files.

### Data contracts

#### Metrics dictionaries

Supervised training step metrics:

- Key: `loss`
- Type: `float`

RL training step metrics:

- Key: `loss`
- Type: `float`
- Key: `policy_loss`
- Type: `float`
- Key: `mean_reward`
- Type: `float`

These contracts must remain unchanged.

#### Experiment metadata

Metadata passed to tracking must be limited to simple scalar values or strings:

- ints
- floats
- strings
- booleans
- None

Metadata must not contain tensors, NumPy arrays, file handles, or arbitrary objects.

#### Checkpoint additions

Supervised checkpoint may gain:

- `seed`: int, present only when seed is provided.

RL checkpoint may gain:

- `seed`: int, present only when seed is provided.

Existing checkpoint consumers must continue to work because extra keys are ignored by existing loading paths.

### Cross-module interactions

- Training loops gain a local dependency on TensorBoard writer behavior.
- Pipelines gain explicit seed propagation.
- CLI scripts gain new optional arguments.
- Evaluation report persistence gains optional TensorBoard output.
- No model, perception, robotics, or simulation module behavior is changed except where RL gamma propagation requires calling an existing function with the correct argument.

## 5. Source Code Impact Analysis

### `src/grasping_ai/training/trainer.py`

**Current responsibility:** Supervised optimizer construction, training step construction, training loop execution, checkpoint save/load.

**Exact functions affected:**

- `build_training_step`
- `run_training_loop`
- `save_training_checkpoint`

**Current behavior:**

- `build_training_step` uses global Torch RNG for diffusion step sampling and noise sampling.
- `run_training_loop` prints loss and saves checkpoint.
- `save_training_checkpoint` writes model, optimizer, epoch, and architecture metadata.

**Required change:**

- Add optional seed parameter to `build_training_step`.
- When seed is provided, use a local Torch generator for diffusion step sampling and noise sampling.
- Add optional experiment log directory and metadata parameters to `run_training_loop`.
- When log directory is provided, create a TensorBoard writer, log metadata, and log scalar metrics.
- Preserve existing print behavior.
- Add optional seed persistence to `save_training_checkpoint`.

**Reason for change:**

- Supervised training currently lacks metric persistence and reproducible noise sampling.

**Dependency impact:**

- Uses existing Torch and TensorBoard dependencies.
- No new package dependency.

**Regression risk:**

- Medium.
- Seeded noise sampling can change numerical results relative to unseeded prior runs.
- Optional parameters must preserve existing call sites.

### `src/grasping_ai/training/rl_trainer.py` (historical)

*The module no longer exists. Its intended responsibilities are fulfilled by the SB3 PPO path in `src/grasping_ai/pipelines/train_rl.py`.*

**Current responsibility:** RL update step construction, RL training loop, return computation.

**Exact functions affected:**

- `build_rl_training_step`
- `run_rl_training_loop`

**Current behavior:**

- `build_rl_training_step` hardcodes gamma as `0.99`.
- `run_rl_training_loop` ignores `log_every` and discards metrics.
- RL checkpoint contains only epoch, model state, and optimizer state.

**Required change:**

- Add explicit gamma parameter to `build_rl_training_step`.
- Use the provided gamma in `compute_discounted_returns`.
- Add optional experiment log directory and metadata parameters to `run_rl_training_loop`.
- Use `log_every` to log metrics when tracking is enabled.
- Add optional seed persistence to the RL checkpoint.

**Reason for change:**

- RL training currently lacks metric persistence and does not honor the pipeline-provided gamma argument.

**Dependency impact:**

- Uses existing TensorBoard dependency.
- No new package dependency.

**Regression risk:**

- Medium.
- Gamma propagation changes numerical behavior if callers previously supplied a gamma other than `0.99`.
- This is intentional because the previous behavior ignored the documented pipeline argument.

### `src/grasping_ai/pipelines/train.py`

**Current responsibility:** End-to-end supervised training orchestration.

**Exact functions affected:**

- `run_training_pipeline`
- Local `TrainingDataloader.__iter__`
- Script-facing call chain through `train_main` in `scripts/train.py`

**Current behavior:**

- No seed argument.
- No experiment log directory argument.
- Local dataloader mutates global Python random state using `random.seed(42)`.
- No tracking metadata is emitted.

**Required change:**

- Add optional seed argument.
- Add optional experiment log directory argument.
- Replace global `random.seed(42)` with a locally seeded Python random generator.
- Pass seed into supervised training step construction.
- Pass experiment log directory and metadata into `run_training_loop`.
- Construct experiment metadata from existing pipeline arguments.

**Reason for change:**

- Supervised pipeline must support explicit reproducibility and tracking.

**Dependency impact:**

- Depends on updated `trainer.py` signatures.
- No new module dependency.

**Regression risk:**

- Medium.
- Dataloader shuffle order changes when no seed is provided.
- When a seed is provided, model initialization and noise sampling become deterministic.

### `src/grasping_ai/pipelines/train_rl.py`

**Current responsibility:** End-to-end RL training orchestration.

**Exact functions affected:**

- `run_rl_training_pipeline`
- `collect_rl_rollout` only indirectly through unchanged behavior.

**Current behavior:**

- Accepts gamma but does not pass it to RL training step.
- No seed argument.
- No experiment log directory argument.
- No tracking metadata emitted.

**Required change:**

- Add optional seed argument.
- Add optional experiment log directory argument.
- Pass gamma into `build_rl_training_step`.
- Pass experiment log directory and metadata into `run_rl_training_loop`.
- Construct experiment metadata from existing pipeline arguments and hardcoded values that affect training, such as rollout step count, clip ratio, and entropy coefficient.

**Reason for change:**

- RL pipeline must support reproducible execution and truthful experiment metadata.

**Dependency impact:**

- The dependency on `rl_trainer.py` is removed because the module no longer exists; the pipeline now depends on `stable-baselines3`.

**Regression risk:**

- Medium.
- Gamma propagation can change training outcomes for non-default gamma values.
- Seeding can change policy initialization.

### `scripts/train.py`

**Current responsibility:** CLI entry point for supervised training.

**Exact function affected:**

- `train_main`
- Argument parser block in `__main__`

**Current behavior:**

- No seed argument.
- No experiment log directory argument.

**Required change:**

- Add optional `--seed` argument.
- Add optional `--experiment-log-dir` argument.
- Pass both arguments through to `run_training_pipeline`.

**Reason for change:**

- Users must be able to request reproducible runs and tracking artifacts from the CLI.

**Dependency impact:**

- Depends on updated pipeline signature.

**Regression risk:**

- Low if new arguments are optional.

### `scripts/train_rl.py`

**Current responsibility:** CLI entry point for RL training.

**Exact function affected:**

- `train_rl_main`
- Argument parser block in `__main__`

**Current behavior:**

- No seed argument.
- No experiment log directory argument.

**Required change:**

- Add optional `--seed` argument.
- Add optional `--experiment-log-dir` argument.
- Pass both arguments through to `run_rl_training_pipeline`.

**Reason for change:**

- Users must be able to request reproducible RL runs and tracking artifacts from the CLI.

**Dependency impact:**

- Depends on updated pipeline signature.

**Regression risk:**

- Low if new arguments are optional.

### `src/grasping_ai/pipelines/evaluate.py`

**Current responsibility:** Evaluation computation and report writing.

**Exact function affected:**

- `write_evaluation_report`

**Current behavior:**

- Writes JSON report only.

**Required change:**

- Add optional experiment log directory argument.
- If provided, write aggregate scalar metrics to TensorBoard.
- Preserve JSON report behavior.

**Reason for change:**

- Evaluation metrics should be observable alongside training artifacts.

**Dependency impact:**

- Uses existing TensorBoard dependency.

**Regression risk:**

- Low because the new argument is optional.

### `scripts/evaluate.py`

**Current responsibility:** CLI entry point for evaluation.

**Exact function affected:**

- `evaluate_main`
- Argument parser block in `__main__`

**Current behavior:**

- No experiment log directory argument.

**Required change:**

- Add optional `--experiment-log-dir` argument.
- Pass it to `write_evaluation_report`.

**Reason for change:**

- Enables evaluation tracking without changing evaluation mathematics.

**Dependency impact:**

- Depends on updated `write_evaluation_report` signature.

**Regression risk:**

- Low.

## 6. New Source Code Units

No new source modules are required.

No new helper functions are required.

No new utility classes are required.

No new abstraction layers are required.

The phase must be implemented exclusively by extending existing functions and existing call sites.

The only external runtime object introduced is a TensorBoard writer instance created locally inside an existing function when tracking is enabled. This uses an existing dependency-provided class and does not introduce a new repository-defined class.

## 7. Existing Code Modification Rules

### Preserve all invariants

- Training loops must still produce checkpoints.
- Supervised training must still return no value.
- RL training must still return no value.
- Evaluation report writing must still produce JSON.
- Existing stdout behavior in supervised training must remain.
- Existing checkpoint loading behavior must remain compatible.

### Preserve public interfaces

All existing required parameters must keep their current order.

New parameters must be optional and appended to existing signatures.

Existing call sites must continue to work without modification unless they are explicitly part of this phase.

### Preserve tensor and data contracts

- Metrics dictionaries must keep existing keys.
- Checkpoints must keep existing keys.
- Dataset records must continue to match the `load_grasp_sample` contract.
- SE(3)-to-9D target conversion must remain unchanged.

### Preserve numerical behavior unless explicitly required

Numerical behavior changes are permitted only for:

1. Deterministic RNG usage when an explicit seed is provided.
2. RL gamma propagation, because the current behavior ignores the pipeline-provided gamma.

No other numerical behavior may be changed.

### No unrelated refactoring

Do not rename variables, restructure pipelines, split files, extract helpers, or modify unrelated logic.

## 8. Detailed Behavioral Design

### Supervised training behavior

`run_training_pipeline` must accept optional seed and experiment log directory arguments.

When a seed is provided:

- Python's local random generator for dataloader shuffling must be seeded with that seed.
- Torch noise and diffusion step sampling inside the training step must use a seeded local generator.
- The seed must be included in checkpoint metadata.
- The seed must be included in experiment metadata.

When no seed is provided:

- The pipeline must not invent a hidden default seed.
- Dataloader shuffling must use a local random generator, not global `random.seed`.
- Torch sampling may use default global Torch RNG behavior.
- No seed key is written to the checkpoint.

When an experiment log directory is provided:

- The training loop must create a TensorBoard writer rooted at that directory.
- The writer must log run metadata before training starts.
- The writer must log scalar training loss at the existing logging interval.
- The writer must be flushed and closed before the function returns.
- If writer creation fails, the failure must propagate naturally. Tracking failure must not be silently ignored.

When no experiment log directory is provided:

- No TensorBoard writer is created.
- Existing behavior remains unchanged.

### RL training behavior

`run_rl_training_pipeline` must accept optional seed and experiment log directory arguments.

The pipeline must pass its existing `gamma` argument into `build_rl_training_step`.

`build_rl_training_step` must use the provided gamma when computing discounted returns.

When an experiment log directory is provided:

- The RL training loop must create a TensorBoard writer rooted at that directory.
- The writer must log run metadata before training starts.
- The writer must log scalar metrics from each update step when the update index satisfies the logging interval.
- The writer must be flushed and closed before the function returns.

When no experiment log directory is provided:

- No TensorBoard writer is created.
- The RL loop must still perform updates and save a checkpoint.

The RL checkpoint must include a seed key when a seed is provided.

### Evaluation behavior

`write_evaluation_report` must continue writing the JSON report.

If an experiment log directory is provided:

- The function must create a TensorBoard writer rooted at that directory.
- It must log each numeric aggregate evaluation metric as a scalar.
- It must flush and close the writer.
- Non-numeric metric values must be skipped rather than coerced unsafely.

If no experiment log directory is provided:

- JSON-only behavior remains unchanged.

### Edge cases

- `experiment_log_dir` exists: writer may append to existing TensorBoard directory.
- `experiment_log_dir` parent does not exist: TensorBoard writer behavior or explicit directory creation must ensure the path can be written.
- `log_every` is zero or negative: logging must be disabled safely instead of raising a modulo error.
- Metrics dictionary contains non-finite values: tracking must not crash; scalar logging may skip non-finite values.
- Evaluation results dictionary is empty: JSON report remains valid and no TensorBoard scalars are required.
- Seed is negative or large: accepted as an integer seed unless existing library seeding rejects it; rejection must propagate naturally.
- Training stops early because RL rollout iterator is exhausted: checkpoint and tracking must still finalize cleanly.

### Failure cases

- Missing dataset root in supervised training must still raise the existing filesystem or value error.
- Missing robot XML in RL training must still raise the existing filesystem error.
- Invalid checkpoint path behavior must remain unchanged.
- TensorBoard writer failure must raise an explicit error rather than silently disabling tracking.
- Invalid metadata values must not be silently transformed into tensors or arrays.

## 9. Cross-Phase Impact

### Impact on Phase 1: Foundation and Math Primitives

- No math primitives are modified.
- Package import behavior must remain stable.
- TensorBoard imports should be lazy inside functions where practical to avoid increasing base import cost.

Required backward compatibility:

- `grasping_ai` remains importable without requiring tracking directories or seeds.

### Impact on Phase 2: Simulation and Robotics Core

- MuJoCo environment behavior is unchanged.
- Rollout collection remains unchanged except that RL gamma propagation affects training returns, not simulation stepping.

Required backward compatibility:

- Simulation stepping, contact reporting, reset behavior, and body pose reading remain unchanged.

### Impact on Phase 3: Data Pipeline and Perception

- Dataset loading remains unchanged.
- The supervised pipeline's dataloader shuffle becomes explicitly seeded or locally random.

Required backward compatibility:

- `load_grasp_sample` and `iterate_grasp_dataset` contracts remain unchanged.

### Impact on Phase 4: Generative Grasp Model

- Model architecture is unchanged.
- Diffusion training noise sampling becomes optionally deterministic.

Required regression tests:

- Existing Phase 4 tests must still pass.
- Supervised training must still produce finite loss and a checkpoint.

### Impact on Phase 5: Reinforcement Learning Policy

- Policy network architecture is unchanged.
- RL update step becomes gamma-correct and observable.

Required regression tests:

- Existing Phase 5 tests must still pass.
- RL update step must still produce finite metrics.

### Impact on Phase 6: End-to-End Orchestration and Evaluation

- Evaluation mathematics is unchanged.
- Evaluation reporting gains optional tracking output.

Required regression tests:

- Existing evaluation JSON behavior must remain unchanged when tracking is disabled.

## 10. Regression Protection

### Unit tests

Required unit tests must cover:

- Supervised training step with seed produces finite loss.
- Supervised training loop writes TensorBoard artifacts when log directory is provided.
- Supervised training loop does not write TensorBoard artifacts when log directory is absent.
- RL training step honors gamma without crashing and produces finite metrics.
- RL training loop writes TensorBoard artifacts when log directory is provided.
- RL training loop does not write TensorBoard artifacts when log directory is absent.
- Evaluation report writes JSON and optional TensorBoard artifacts.
- Checkpoint save includes seed only when seed is provided.
- Existing checkpoint load remains compatible with checkpoints containing seed metadata.

### Integration tests

Required integration tests must cover:

- End-to-end supervised training with a tiny synthetic dataset, explicit seed, and tracking enabled.
- End-to-end supervised training with tracking disabled.
- RL training loop with a synthetic rollout iterator and tracking enabled.
- Evaluation report generation with aggregate metrics and tracking enabled.

### Simulation tests

No new MuJoCo simulation tests are required unless RL training loop tests require a fake rollout iterator. Full MuJoCo integration tests should rely on existing Phase 2 and Phase 5 tests.

### Numerical validation tests

Required numerical validation:

- Same explicit seed produces identical supervised training loss across repeated runs on CPU for a tiny model and tiny dataset, within deterministic CPU tolerance.
- Different seeds are not required to produce different results, but deterministic same-seed behavior is required.
- RL discounted returns change when gamma changes, confirming gamma propagation.

### Explicit protection of previous phases

The full existing test suite must be run before committing this phase. In particular:

- Phase 1 package import tests.
- Phase 3 dataset loading tests.
- Phase 4 generative training tests.
- Phase 5 RL policy tests.
- Phase 6 evaluation tests.

## 11. New Test Suite

### `tests/unit/test_phase8_training_tracking.py`

**Purpose:** Validate supervised training tracking and seed handling.

**Input strategy:**

- Tiny Torch module.
- Tiny tensor batch.
- Temporary checkpoint path.
- Temporary TensorBoard log directory.

**Expected output:**

- Checkpoint file exists.
- TensorBoard directory contains at least one event file when tracking is enabled.
- No event file is created when tracking is disabled.
- Checkpoint contains seed key only when seed is provided.

**Failure condition:**

- Missing checkpoint.
- Missing event file when tracking enabled.
- Unexpected event file when tracking disabled.
- Missing or unexpected seed key.

**Determinism requirement:**

- Tests must use fixed seeds and CPU tensors.

**CI suitability:**

- Fast, local, no network, no GPU, no dataset assets.

### `tests/unit/test_phase8_rl_tracking.py`

**Purpose:** Validate RL tracking, gamma propagation, and seed persistence.

**Input strategy:**

- Tiny policy network.
- Synthetic transition list.
- Fake rollout iterator.
- Temporary checkpoint path.
- Temporary TensorBoard log directory.

**Expected output:**

- RL checkpoint exists.
- TensorBoard event file exists when tracking enabled.
- Metrics dictionary keys remain unchanged.
- Checkpoint contains seed key only when seed is provided.
- Gamma parameter affects discounted returns.

**Failure condition:**

- Missing checkpoint.
- Missing event file when tracking enabled.
- Changed metric keys.
- Gamma value ignored.

**Determinism requirement:**

- Fixed seeds for model initialization where needed.
- CPU-only tensors.

**CI suitability:**

- Fast and local.

### `tests/unit/test_phase8_evaluation_tracking.py`

**Purpose:** Validate evaluation report tracking without changing metric computation.

**Input strategy:**

- Temporary JSON report path.
- Temporary TensorBoard log directory.
- Small dictionary of aggregate metrics.

**Expected output:**

- JSON report exists and contains the same metrics.
- TensorBoard event file exists when tracking enabled.
- No event file exists when tracking disabled.

**Failure condition:**

- JSON report missing.
- Metric values altered.
- Event file missing when tracking enabled.

**Determinism requirement:**

- Static metric dictionary.

**CI suitability:**

- Fast and local.

### `tests/integration/test_phase8_supervised_reproducibility.py`

**Purpose:** Validate that explicit seed produces reproducible supervised training artifacts.

**Input strategy:**

- Tiny synthetic dataset record saved as `.npy`.
- Tiny model dimensions.
- One epoch.
- Batch size one.
- Explicit seed.
- Two temporary checkpoint destinations.

**Expected output:**

- Both runs complete.
- Model state dictionaries are identical or equal within deterministic CPU tolerance.
- Checkpoints include the same seed.

**Failure condition:**

- Different model weights for identical seed and input.
- Missing checkpoint.
- Tracking or seeding arguments ignored.

**Determinism requirement:**

- CPU-only.
- Fixed seed.
- No background threads affecting file output.

**CI suitability:**

- Tiny dimensions keep runtime short.

## 12. Test Matrix

| Code unit | Affected phase | Test | Regression coverage | Commit gate |
|---|---:|---|---|---|
| `build_training_step` | Phase 4, Phase 8 | `tests/unit/test_phase8_training_tracking.py` | Supervised loss behavior, seeded noise sampling | Unit tests pass |
| `run_training_loop` | Phase 4, Phase 8 | `tests/unit/test_phase8_training_tracking.py` | Checkpoint creation, logging behavior | Unit tests pass |
| `save_training_checkpoint` | Phase 4, Phase 8 | `tests/unit/test_phase8_training_tracking.py` | Existing checkpoint keys, optional seed key | Unit tests pass |
| `run_training_pipeline` | Phase 4, Phase 8 | `tests/integration/test_phase8_supervised_reproducibility.py` | Dataset loading, model construction, reproducibility | Integration tests pass |
| `build_rl_training_step` | Phase 5, Phase 8 | `tests/unit/test_phase8_rl_tracking.py` | RL metrics, gamma propagation | Unit tests pass |
| `run_rl_training_loop` | Phase 5, Phase 8 | `tests/unit/test_phase8_rl_tracking.py` | RL checkpoint, logging interval | Unit tests pass |
| `run_rl_training_pipeline` | Phase 5, Phase 8 | Existing RL integration tests plus Phase 8 tests | Gamma argument usage, metadata propagation | Existing and new tests pass |
| `write_evaluation_report` | Phase 6, Phase 8 | `tests/unit/test_phase8_evaluation_tracking.py` | JSON report compatibility | Unit tests pass |
| `scripts/train.py` | Phase 4, Phase 8 | Manual local validation and integration tests | CLI backward compatibility | Optional arguments do not break existing calls |
| `scripts/train_rl.py` | Phase 5, Phase 8 | Manual local validation and integration tests | CLI backward compatibility | Optional arguments do not break existing calls |
| `scripts/evaluate.py` | Phase 6, Phase 8 | `tests/unit/test_phase8_evaluation_tracking.py` | CLI evaluation compatibility | Optional argument does not break existing calls |

## 13. Numerical and Robotics Validation

### Tensor shape validation

- Supervised training metrics remain scalar floats.
- RL metrics remain scalar floats.
- Checkpoint metadata does not alter model tensor shapes.
- Evaluation metrics remain scalar values.

### Determinism constraints

- Same explicit seed must produce same supervised training result on CPU for tiny model and dataset.
- Dataloader shuffle must not depend on global Python random mutation.
- Torch diffusion step sampling and noise sampling must use the seeded generator when seed is provided.
- TensorBoard event file names may include timestamps and are not required to be deterministic. Existence and metric determinism are the validation targets.

### RL-specific validation

- Discounted returns must reflect the provided gamma.
- Policy output shape must remain `(B, action_dim)`.
- Rollout transition contract remains `(obs, action, reward, next_obs, done)`.

### MuJoCo simulation

- This phase does not modify MuJoCo stepping.
- Existing simulation determinism constraints from Phase 2 remain applicable.
- No new simulation randomness is introduced.

## 14. Data and Interface Contracts

### Training metrics

Supervised:

- `loss`: float

RL:

- `loss`: float
- `policy_loss`: float
- `mean_reward`: float

### Experiment metadata

Metadata must be a flat dictionary with scalar or string values.

Expected supervised metadata fields:

- dataset root
- checkpoint path
- feature dimension
- hidden dimension
- number of layers
- learning rate
- number of epochs
- batch size
- device
- seed, if provided

Expected RL metadata fields:

- robot XML path
- YCB root
- object identifier
- policy checkpoint path
- observation dimension
- action dimension
- hidden dimension
- learning rate
- number of updates
- gamma
- device
- seed, if provided
- rollout step count
- clip ratio
- entropy coefficient

### Persistence formats

- Checkpoints remain Torch save files.
- Evaluation reports remain JSON files.
- Tracking artifacts are TensorBoard event files.

### Units and coordinate systems

This phase does not change physical units, coordinate systems, or spatial representations.

## 15. Configuration and Dependency Impact

### Configuration files

No YAML configuration file changes are required.

No YAML parsing is introduced.

No global configuration loader is introduced.

### Dependency justification

- `tensorboard` is already declared in `pyproject.toml`.
- `torch` is already declared and provides the TensorBoard writer integration.
- `wandb` remains unused in this phase.

### Environment constraints

- Local filesystem write access is required for checkpoints, JSON reports, and TensorBoard artifacts.
- No network access is required.
- No GPU is required.
- No physical robot hardware is required.

## 16. Reproducibility and Local Validation

### How to run locally

Supervised training with tracking:

- Run `scripts/train.py` with an existing tiny synthetic dataset, a checkpoint path, tiny model dimensions, explicit seed, and an experiment log directory.

RL training with tracking:

- Run `scripts/train_rl.py` with a local robot XML, local YCB root, object ID, policy checkpoint path, tiny dimensions, explicit seed, and an experiment log directory.

Evaluation with tracking:

- Run `scripts/evaluate.py` with generated grasps, point clouds, report path, and an experiment log directory.

### Deterministic setup

- Use CPU device.
- Provide the same explicit seed.
- Use the same dataset records.
- Use the same CLI arguments.
- Avoid GPU nondeterminism.

### Synthetic data usage

Tests must use tiny synthetic `.npy` records rather than full YCB assets.

### No physical robot dependency

All validation paths must run offline in local simulation or pure numeric mode.

## 17. Implementation Order

### Step 1: Extend supervised trainer

Modify:

- `build_training_step`
- `run_training_loop`
- `save_training_checkpoint`

Then run existing supervised tests.

### Step 2: Extend supervised pipeline and script

Modify:

- `run_training_pipeline`
- `TrainingDataloader`
- `scripts/train.py`

Then run existing supervised pipeline tests and new supervised tracking tests.

### Step 3: Extend RL trainer (historical: module removed in the SB3 migration; this step no longer applies)

Modify:

- `build_rl_training_step`
- `run_rl_training_loop`

Then run existing RL tests.

### Step 4: Extend RL pipeline and script

Modify:

- `run_rl_training_pipeline`
- `scripts/train_rl.py`

Then run existing RL pipeline tests and new RL tracking tests.

### Step 5: Extend evaluation reporting

Modify:

- `write_evaluation_report`
- `scripts/evaluate.py`

Then run existing evaluation tests and new evaluation tracking tests.

### Step 6: Add reproducibility integration test

Add:

- `tests/integration/test_phase8_supervised_reproducibility.py`

Then run the full test suite.

At every step, the system must remain testable with tracking disabled.

## 18. Commit Gates

A commit for this phase is allowed only if:

1. All existing tests pass.
2. All new Phase 8 tests pass.
3. No regression is observed in Phase 1 through Phase 6 tests.
4. No helper functions are introduced.
5. No utility modules are introduced.
6. No new repository-defined classes are introduced.
7. No global variables or global constants are introduced.
8. No global configuration state is introduced.
9. Every modified function has an updated Google-style docstring.
10. Every new test function has a Google-style docstring.
11. TensorBoard tracking is optional and disabled by default when no log directory is provided.
12. Existing checkpoint loading remains compatible.
13. Existing CLI invocations remain valid.
14. Static checks required by the repository, including linting and type checking, pass.

## 19. Definition of Done

This phase is complete when all of the following are true:

- Supervised training can write TensorBoard artifacts when an experiment log directory is provided.
- RL training can write TensorBoard artifacts when an experiment log directory is provided.
- Evaluation reporting can write TensorBoard artifacts when an experiment log directory is provided.
- Supervised training accepts an explicit seed and produces deterministic CPU results for identical inputs and seed.
- RL training accepts an explicit seed and persists it when provided.
- The RL training step uses the gamma value supplied by the RL pipeline.
- The supervised dataloader no longer mutates global Python random state.
- Checkpoints contain a seed field when a seed is provided.
- Existing checkpoints without a seed field still load correctly.
- Existing behavior is preserved when tracking arguments are omitted.
- All tests in the repository pass.
- No forbidden constructs are introduced.
- Local validation requires no network, GPU, dataset download, or physical robot.

## 20. Risks and Failure Modes

### Technical risk: TensorBoard writer lifecycle

**Risk:** Writer may not flush or close on failure, leaving incomplete artifacts.

**Mitigation:** Design requires writer finalization in all normal and failure exit paths. Tests must validate artifact creation in successful paths.

### Technical risk: Global RNG seeding conflicts with no-global-state constraint

**Risk:** Torch model initialization determinism may require process-level seeding.

**Mitigation:** Seeding is explicit, optional, and confined to pipeline entry. No repository module stores mutable global state. Dataloader randomness must use a local generator.

### Numerical risk: Seeding changes training outcomes

**Risk:** Existing tests or expected losses may differ after seeded runs.

**Detection:** Run full regression suite. Compare only stable properties such as checkpoint existence, finite loss, and deterministic same-seed equality.

### Numerical risk: Gamma propagation changes RL behavior

**Risk:** RL runs with gamma not equal to `0.99` will now behave differently.

**Mitigation:** This is required because the previous behavior ignored the pipeline argument. Tests must verify gamma propagation.

### Integration risk: Checkpoint metadata breaks loaders

**Risk:** Additional checkpoint keys may break older loaders.

**Mitigation:** Add only optional scalar `seed` key. Existing loaders ignore unknown keys. Regression tests must load new checkpoints with existing loading functions.

### Regression risk: Existing scripts break due to new arguments

**Risk:** CLI parsing changes may reject old commands.

**Mitigation:** New arguments must be optional. Existing argument order and required flags must remain unchanged.

### Reproducibility risk: GPU nondeterminism

**Risk:** Same seed may not reproduce results on GPU.

**Mitigation:** Validation requirement is CPU-only deterministic local validation. Documentation and tests must use CPU.

## 21. Out of Scope

The following are explicitly out of scope for this phase:

- Weights and Biases integration.
- Remote experiment tracking.
- Hyperparameter optimization.
- Automatic run naming.
- Experiment comparison tooling.
- Checkpoint resumption.
- Dataset generation.
- Synthetic grasp generation.
- RL exploration noise redesign.
- Reward function changes.
- MuJoCo environment changes.
- Force-closure metric corrections.
- Collision-checking corrections.
- Evaluation contact provider correctness.
- Physical robot deployment.
- YAML configuration parsing.
- Global configuration objects.
- New utility modules.
- New helper functions.
- New helper classes.

## 22. Design Review Checklist

- Repository verified against inspected training, RL, pipeline, script, evaluation, and dependency files.
- Phase boundary restricted to tracking and reproducibility.
- No new helper functions introduced.
- No new utility modules introduced.
- No new repository-defined classes introduced.
- No global variables introduced.
- No global constants introduced.
- No global configuration state introduced.
- Existing checkpoint contracts preserved.
- Existing metrics contracts preserved.
- Existing CLI behavior preserved through optional arguments.
- TensorBoard tracking optional and local.
- Seed handling explicit.
- RL gamma inconsistency resolved.
- Test suite covers unit, integration, tracking, and reproducibility concerns.
- Local validation requires no physical robot, network, or GPU.