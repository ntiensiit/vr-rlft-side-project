# Phase 5 — Reinforcement Learning Policy

> **Historical design record.** This document captures the Phase 5 plan from the skeleton era. Current architecture, CLIs, and contracts live in [architecture.md](architecture.md), [USAGE.md](USAGE.md), and the repository [README.md](../README.md).

## 1. Phase Objective

Phase 5 establishes the minimum reinforcement learning capability required to train a grasping or interaction policy against the Phase 2 simulation core.

The phase must deliver only the following capabilities:

1. A verified reinforcement learning training pipeline that can initialize a simulation environment, train a policy, and save a policy checkpoint.
2. A verified policy behavior compatible with the environment observation and action contracts.
3. A verified reward behavior sufficient for reinforcement learning updates.
4. A stable checkpoint contract that later evaluation or inference phases can consume.
5. Tests that prove dependency availability, pipeline initialization, dimension validation, minimal training execution, reward finiteness, and checkpoint creation without requiring GPUs, external datasets, or real robot hardware.

This phase is intentionally conservative. The repository is a skeleton, and most source files inside `src/grasping_ai/training/`, `src/grasping_ai/models/`, and `src/grasping_ai/simulation/` were not fully verifiable in the available snapshot. However, `scripts/train_rl.py` was verified as the executable training entrypoint.

Where an exact file path or symbol cannot be verified, it is marked `Unverified`.

The implementation must not create new classes, helper utilities, global state, or abstraction layers. If the repository does not already contain the necessary skeleton units and the phase cannot be completed using existing modules and existing third-party behavior already declared as dependencies, the phase must be paused for a scope decision.

---

## 2. Verified Repository Context

The following facts were verified from the repository snapshot available during analysis.

### Verified script facts

- `scripts/train_rl.py` exists.
- `scripts/train_rl.py` imports:

  `python scripts/train_rl.py`

- `scripts/train_rl.py` defines a thin CLI entry point that parses arguments and calls `run_rl_training_pipeline`.
- The verified CLI arguments are:
  - `--robot-xml`
  - `--ycb-root`
  - `--object-ids`
  - `--policy-checkpoint`
  - `--observation-dim`
  - `--action-dim`
  - `--hidden-dim`
  - `--learning-rate`
  - `--num-updates`
  - `--gamma`
  - `--device`

This confirms that the RL training entry point already exists and that the missing behavior is primarily in the pipeline function and its downstream dependencies.

### Verified packaging facts

- `pyproject.toml` exists.
- The build backend is `hatchling.build`.
- The wheel target uses `packages = ["src"]`.
- Python requirement is `>=3.12`.
- Runtime dependencies already include:
  - `stable-baselines3`
  - `gymnasium`
  - `torch`
  - `pytransform3d`
  - `scipy`
- Development dependencies already include:
  - `mujoco`
  - `pytest`
  - `pytest-cov`
  - `ruff`
  - `mypy`

### Verified project documentation facts

- `README.md` states that the repository is a Python source-code skeleton and that function bodies raise `NotImplementedError`.
- `README.md` describes:
  - `src/grasping_ai/models/` as containing RL policy models.
  - `src/grasping_ai/training/` as containing supervised and RL training loops.
  - `src/grasping_ai/pipelines/` as end-to-end orchestration.
- `README.md` defines the RL training flow as:

  `simulation -> pipelines.train_rl (SB3 PPO + Gymnasium) -> exported legacy checkpoint -> inference.policy_runner`
  *(historical README flow: `simulation -> inference.policy_runner -> training.rl_trainer -> checkpoints`; `training.rl_trainer` no longer exists)*

- `docs/PROJECT.md` states that the exact RL algorithm has not been fixed.
- `docs/USAGE.md` describes `configs/training.yaml` as containing reinforcement learning hyperparameters.

### Non-verified or partially verified areas

The following are not verified in the available snapshot:

- Exact contents of `src/grasping_ai/pipelines/train_rl.py`.
- Exact contents of `src/grasping_ai/training/`.
- Exact contents of `src/grasping_ai/models/`.
- Exact contents of `src/grasping_ai/simulation/`.
- Existing test files, fixtures, and test conventions.
- Exact YAML key names inside `configs/training.yaml`.
- Whether a custom RL policy skeleton exists.
- Whether a custom reward function skeleton exists.
- Whether an existing checkpoint-loading behavior exists for later inference.

Because of this, any source-unit design inside those directories is marked `Unverified` where exact paths or symbols cannot be confirmed.

---

## 3. Scope

Phase 5 includes only the following work.

### In scope

1. Implementation of the existing RL training pipeline function imported by `scripts/train_rl.py`.
   - Validate explicit CLI-provided arguments.
   - Initialize the Phase 2 simulation environment.
   - Validate observation and action dimensions.
   - Train a policy using the existing Stable-Baselines3 dependency.
   - Save a policy checkpoint.

2. Implementation or completion of the existing RL policy behavior, if a repository-specific policy skeleton exists.
   - If a custom policy skeleton exists, implement its existing behavior.
   - If no custom policy skeleton exists, use existing Stable-Baselines3 policy behavior without defining a new policy class.

3. Implementation or completion of the existing reward behavior.
   - Reward behavior should be implemented inside the existing environment step or reward unit where possible.
   - Reward output must be a finite scalar.
   - Reward semantics must remain simple and explicit.

4. Preservation of static training configuration.
   - Do not remove `configs/training.yaml`.
   - Do not introduce global configuration loading.
   - Do not introduce module-level configuration state.

5. Tests for the Phase 5 contract.
   - Dependency availability.
   - Pipeline initialization.
   - Dimension validation.
   - Minimal training execution.
   - Reward finiteness.
   - Checkpoint creation.
   - Regression of Phase 1 and Phase 2 behavior.

### Conditional scope

The RL implementation is conditional on locating existing skeleton units.

If `src/grasping_ai/pipelines/train_rl.py` does not contain `run_rl_training_pipeline`, the phase must be paused because the verified script import contract is broken.

If no simulation environment initialization behavior exists from Phase 2, Phase 5 must not create a new environment class. The correct action is to stop and request a scope decision.

If no reward behavior exists and the environment skeleton does not expose enough state to implement a reward, Phase 5 must not invent a new reward abstraction. The correct action is to stop and request a scope decision.

---

## 4. Out of Scope

The following are explicitly out of scope for Phase 5.

1. Supervised grasp-generation training.
   - Diffusion model training.
   - Flow model training.
   - Grasp pose dataset training.

2. Grasp generation inference.
   - Sampling grasp poses from generative models.
   - Loading generative checkpoints.
   - Ranking or filtering grasp candidates.

3. Full evaluation pipeline.
   - Force-closure evaluation.
   - Lift-success reporting.
   - Collision reporting.
   - Aggregate benchmark reports.

4. Simulation expansion.
   - Domain randomization.
   - Parallel environments.
   - Sensor noise simulation.
   - Actuator delay simulation.
   - New physics engines.
   - New robot models.

5. Real-robot integration.
   - Hardware drivers.
   - Robot communication.
   - Safety controllers.
   - Calibration pipelines.
   - Deployment workflows.

6. Advanced RL infrastructure.
   - Distributed RL.
   - Replay buffer abstractions.
   - Custom callback classes.
   - Custom logger classes.
   - Experiment tracking integration.
   - Hyperparameter sweeps.

7. New architecture.
   - New RL algorithm classes.
   - New policy wrapper classes.
   - New environment wrapper classes.
   - New helper functions.
   - New utility modules.
   - New global configuration managers.

---

## 5. Existing Architecture and Patterns

Phase 5 must preserve the following verified or documented repository patterns.

### Pattern 1: Thin CLI scripts delegate to pipeline functions

`scripts/train_rl.py` demonstrates the existing pattern:

- The script parses CLI arguments.
- The script calls `run_rl_training_pipeline`.
- The script does not contain training logic.

Phase 5 must preserve this pattern. The implementation must occur primarily in `run_rl_training_pipeline` or in lower-level modules that it calls.

### Pattern 2: Skeleton functions raise `NotImplementedError`

The README states that the repository contains skeleton code with `NotImplementedError` bodies. Phase 5 should replace `NotImplementedError` only in existing functions, methods, or module execution blocks that belong to the phase scope.

Phase 5 must not create new reusable helper functions to avoid implementing required behavior directly in the appropriate existing skeleton unit.

### Pattern 3: Pipelines orchestrate lower-level modules

The documented architecture places pipelines above simulation, models, training, and inference.

Phase 5 pipeline code may orchestrate:

- Simulation environment initialization.
- Policy construction.
- Training execution.
- Checkpoint saving.

Lower-level modules must not import the pipeline.

### Pattern 4: AI models do not directly depend on robot hardware

`docs/PROJECT.md` states that AI models must not directly depend on robot hardware.

Phase 5 policy behavior must not import hardware, driver, calibration, or deployment modules. It may depend on the simulation environment interface because simulation is the verified training substrate.

### Pattern 5: Configuration remains static

The repository uses YAML configuration files, but no global configuration loader was verified.

Phase 5 must not introduce a global configuration object, module-level config cache, or singleton configuration manager. RL hyperparameters should continue to be accepted explicitly through the verified CLI pipeline arguments.

---

## 6. Implementation Dependencies

Phase 5 requires the following dependencies.

### Existing dependencies that must remain available

| Dependency | Reason |
| --- | --- |
| `stable-baselines3` | Existing RL training dependency. |
| `gymnasium` | Existing environment interface dependency. |
| `torch` | Existing tensor and model dependency. |
| `mujoco` | Development dependency required by the Phase 2 simulation core. |
| `pytest` | Development dependency for Phase 5 tests. |

### Dependencies explicitly not required

| Dependency | Reason for exclusion |
| --- | --- |
| New RL library | Stable-Baselines3 already exists. |
| New experiment tracking library | Phase 5 does not require dashboards or remote logging. |
| New configuration library | Global configuration is out of scope. |
| New visualization library | Phase 5 does not require policy visualization. |

No new third-party dependency may be introduced.

---

## 7. Source-Unit Change Matrix

Rows marked `Unverified` are required phase targets, but the exact file path or symbol must be resolved from the repository before implementation. They are not speculative new files.

| File | Symbol | Change Type | Current Responsibility | Proposed Change | Risk | Required Tests |
| --- | --- | --- | --- | --- | --- | --- |
| `src/grasping_ai/pipelines/train_rl.py` | `run_rl_training_pipeline` | Modify | Imported by `scripts/train_rl.py`; expected to orchestrate RL training. | Implement argument validation, environment initialization, policy training, and checkpoint saving. | High | Pipeline initialization, training, and checkpoint tests. |
| Unverified existing module under `src/grasping_ai/training/` | Unverified existing RL training function | Modify if present | Expected to contain RL training loop behavior. | Complete RL training behavior or connect pipeline behavior to the existing training function. | High | Minimal training tests. |
| Unverified existing module under `src/grasping_ai/models/` | Unverified existing RL policy symbol | Modify if present | Expected to contain RL policy behavior. | Implement existing policy behavior or connect it to Stable-Baselines3 policy behavior without defining a new class. | High | Policy compatibility tests if a custom policy exists. |
| Unverified existing module under `src/grasping_ai/simulation/` | Unverified existing step or reward behavior | Modify if reward is not already meaningful | Expected to return environment step results, including reward. | Implement finite scalar reward behavior using existing simulation state. | High | Reward finiteness and step-contract tests. |
| Unverified existing module under `src/grasping_ai/training/` or `src/grasping_ai/inference/` | Unverified existing RL checkpoint saving/loading behavior | Modify if present | Expected to save or load RL policy checkpoints. | Implement checkpoint saving using the existing skeleton contract or Stable-Baselines3 native saving behavior. | Medium | Checkpoint creation and loadability tests. |
| `configs/training.yaml` | Documented RL training settings | Configuration preservation | Declares RL hyperparameter semantics according to documentation. | Preserve file and documented semantics. Do not introduce global loading. If exact keys are verified, keep them unchanged. | Low | Configuration existence test. |
| Existing or minimum required test file under `tests/unit/` | Phase 5 RL policy tests | Test-only change | No verified existing Phase 5 tests. | Add tests for dependency availability, pipeline initialization, dimension validation, minimal training, reward finiteness, and checkpoint creation. | Medium | All Phase 5 tests pass. |

`scripts/train_rl.py` is expected to remain unchanged unless the verified CLI contract is broken. It already delegates to `run_rl_training_pipeline`.

---

## 8. Detailed Source-Unit Design

### `src/grasping_ai/pipelines/train_rl.py::run_rl_training_pipeline`

#### Current behavior

The symbol is imported by `scripts/train_rl.py`. The file contents were not verified, but the repository skeleton status indicates that the function body is likely unimplemented.

#### Required behavior

`run_rl_training_pipeline` must orchestrate the full minimal RL training flow.

It must:

1. Accept the arguments already passed by `scripts/train_rl.py`.
2. Validate those arguments.
3. Initialize the Phase 2 simulation environment.
4. Validate that environment observation and action dimensions match the expected dimensions.
5. Construct or configure an RL policy using existing repository or Stable-Baselines3 behavior.
6. Train the policy for the requested number of updates.
7. Save the policy checkpoint to the requested path.

#### Design

##### Input

The verified inputs are:

- Robot XML path.
- YCB root path.
- Object identifier list.
- Policy checkpoint path.
- Observation dimension.
- Action dimension.
- Hidden dimension.
- Learning rate.
- Number of updates.
- Gamma.
- Device identifier.

##### Output

The function may return nothing if the existing skeleton contract defines no return value. Its side effect is the creation of a policy checkpoint.

##### Control flow

1. Validate filesystem paths.
2. Validate numeric arguments.
3. Initialize the simulation environment using the existing Phase 2 environment behavior.
4. Verify that the environment observation space is compatible with the expected observation dimension.
5. Verify that the environment action space is compatible with the expected action dimension.
6. Construct or configure the policy using existing Stable-Baselines3 behavior or an existing repository policy skeleton.
7. Execute the minimal training loop for the requested number of updates.
8. Save the checkpoint.

##### State changes

Training creates local policy state and simulation state. Module-level state must not be introduced.

##### Existing dependencies

- Phase 2 simulation environment initialization behavior.
- Stable-Baselines3.
- Gymnasium.
- Torch.

##### Error behavior

- Missing robot XML path raises a filesystem error.
- Missing YCB root raises a filesystem error when object IDs require it.
- Invalid dimensions raise value errors.
- Unsupported device raises a value or runtime error.
- Environment initialization failure propagates naturally.

##### Edge cases

- Empty object ID list.
- CPU-only execution.
- Very small hidden dimension.
- One update only.
- Checkpoint parent directory does not exist.

##### Interaction with surrounding code

`scripts/train_rl.py` calls this function. Later evaluation or inference phases may load the checkpoint produced by this function.

##### Existing pattern being followed

Pipeline orchestration remains above lower-level modules and does not move logic into the CLI script.

---

### Unverified existing RL training function

#### Current behavior

The exact training symbol was not verified. The current behavior is expected to be `NotImplementedError` if the symbol exists.

#### Required behavior

If an existing RL training function exists under `src/grasping_ai/training/`, it must implement the actual training behavior used by the pipeline.

If no such function exists, the pipeline may implement the minimal training behavior directly, provided no new helper abstraction is created.

#### Design

##### Input

Expected inputs include:

- Initialized environment or environment factory result.
- Policy configuration.
- Learning rate.
- Gamma.
- Number of updates.
- Device identifier.

##### Output

The output may be a trained policy object, training metadata, or nothing if the existing skeleton contract defines no return value.

##### Control flow

1. Receive an initialized environment.
2. Configure the policy using existing Stable-Baselines3 behavior.
3. Perform the requested number of training updates.
4. Return or expose the trained policy for checkpoint saving.

##### State changes

Policy parameters change during training. Module-level state must not change.

##### Existing dependencies

- Stable-Baselines3.
- Gymnasium environment.
- Torch.

##### Error behavior

- Invalid learning rate raises a value error.
- Invalid gamma raises a value error.
- Non-finite training loss or policy output raises an explicit error.

##### Edge cases

- One update.
- Very small network width.
- CPU-only execution.

##### Interaction with surrounding code

The pipeline calls this behavior. It must not call the pipeline back.

##### Existing pattern being followed

Training behavior remains separate from CLI entry points.

---

### Unverified existing RL policy symbol

#### Current behavior

The exact policy symbol was not verified. README documentation states that `src/grasping_ai/models/` contains RL policy models.

#### Required behavior

If a repository-specific RL policy skeleton exists, Phase 5 must implement that existing policy behavior.

If no repository-specific policy skeleton exists, Phase 5 must use existing Stable-Baselines3 policy behavior and must not define a new policy class.

#### Design

##### Input

Expected policy inputs include:

- Observation tensor or observation vector.
- Policy state if required by the existing skeleton.

##### Output

Expected policy outputs include:

- Action.
- Action distribution parameters or log-probability if required by the existing training behavior.

##### Control flow

1. Validate observation shape.
2. Apply the existing policy network behavior.
3. Return action-related outputs.

##### State changes

Policy parameters are read during inference and updated during training. Module-level state must not be introduced.

##### Existing dependencies

- Torch.
- Stable-Baselines3 if the policy is delegated to existing library behavior.

##### Error behavior

- Invalid observation shape raises a value error.
- Non-finite observation raises a value error.

##### Edge cases

- Batch size one.
- CPU-only execution.
- Continuous action spaces.

##### Interaction with surrounding code

Training and later inference consume policy behavior. The policy must not depend on pipelines or CLI scripts.

##### Existing pattern being followed

Models remain low-level and do not depend on orchestration.

---

### Unverified existing step or reward behavior

#### Current behavior

The exact simulation step or reward symbol was not verified. Phase 2 established that the simulation step contract should return observation, reward, termination, truncation, and info if it follows Gymnasium behavior.

#### Required behavior

The environment step behavior must return a finite scalar reward suitable for RL training.

If Phase 2 currently returns a placeholder reward, Phase 5 may modify the existing reward behavior to return a meaningful finite scalar based on available simulation state.

#### Design

##### Input

The input is the current simulation state and the action being executed.

##### Output

The output reward must be a finite scalar.

##### Control flow

1. Read available simulation state after the action is applied.
2. Compute reward from explicit state fields.
3. Return the reward as part of the step result.

##### State changes

Simulation state changes normally during stepping. Module-level state must not change.

##### Existing dependencies

- Phase 2 simulation state.
- Gymnasium step contract.

##### Error behavior

- Non-finite reward must not be returned.
- Missing required state fields should either fall back to a finite placeholder reward or raise an explicit error if the existing skeleton contract requires those fields.

##### Edge cases

- Episode termination.
- Episode truncation.
- Object state unavailable.
- Gripper state unavailable.

##### Interaction with surrounding code

Stable-Baselines3 consumes the reward returned by the environment step.

##### Existing pattern being followed

Reward remains part of the environment contract rather than being injected through global state.

---

### Unverified existing RL checkpoint saving/loading behavior

#### Current behavior

The exact checkpoint symbol was not verified. The current behavior is expected to be `NotImplementedError` if the symbol exists.

#### Required behavior

The checkpoint behavior must save the trained RL policy to the requested checkpoint path.

If an existing repository checkpoint contract exists, it must be preserved. If no existing contract exists, the implementation may use Stable-Baselines3 native checkpoint saving behavior.

#### Design

##### Input

Expected inputs include:

- Trained policy or algorithm state.
- Checkpoint path.

##### Output

The output is a checkpoint artifact on disk.

##### Control flow

1. Validate checkpoint path.
2. Create parent directories if necessary.
3. Save the policy or algorithm state.
4. Ensure the artifact is discoverable by later loading behavior.

##### State changes

Filesystem side effect: writes checkpoint artifact. No module-level state.

##### Existing dependencies

- Stable-Baselines3 checkpoint saving behavior if used.
- Standard filesystem utilities.

##### Error behavior

- Invalid checkpoint path raises a filesystem or value error.
- Saving failure propagates naturally.

##### Edge cases

- Checkpoint path without extension.
- Existing checkpoint file overwrite.
- CPU-only execution.

##### Interaction with surrounding code

Later evaluation or inference phases may load the checkpoint.

##### Existing pattern being followed

Checkpointing is explicit and does not rely on global state.

---

### `configs/training.yaml`

#### Current behavior

Documentation states that this file contains training hyperparameters, including reinforcement learning hyperparameters. The exact key names were not verified.

#### Required behavior

Phase 5 must preserve the file and its documented semantics. Phase 5 must not introduce global configuration loading. RL hyperparameters should continue to be accepted explicitly through the verified CLI pipeline arguments.

#### Design

- Input: none.
- Output: none.
- Control flow: none.
- State changes: none.
- Existing dependencies: none.
- Error behavior: not applicable during Phase 5 because no runtime YAML parsing is introduced.
- Edge cases: missing keys should be handled by later pipeline validation if that pipeline chooses to consume the file.
- Interaction with surrounding code: later orchestration may read this file, but Phase 5 does not require it.
- Existing pattern being followed: static configuration without global state.

No YAML key may be renamed, removed, or added unless exact key verification proves that a documented key is missing.

---

## 9. Data Flow

Phase 5 introduces the following data flows.

### RL training data flow

`CLI arguments -> scripts/train_rl.py -> run_rl_training_pipeline -> Phase 2 simulation environment -> Stable-Baselines3 policy -> Policy update -> Checkpoint file`

Input source:

- CLI caller provides robot XML, YCB root, object IDs, checkpoint path, dimensions, hyperparameters, and device.

Transformation:

- Arguments are validated.
- Environment is initialized.
- Policy is configured.
- Rollouts are collected.
- Policy parameters are updated.

Validation:

- Filesystem paths.
- Observation dimension.
- Action dimension.
- Device compatibility.
- Finite reward.

Output:

- Policy checkpoint artifact.

Consumers:

- Later evaluation phases.
- Later inference phases.
- Potential policy execution in simulation.

### Environment interaction data flow

`Environment reset -> Observation -> Policy -> Action -> Environment step -> Reward, next observation, termination flags, info`

Input source:

- Simulation environment.

Transformation:

- Policy maps observation to action.
- Environment applies action and advances physics.

Validation:

- Observation shape.
- Action shape.
- Reward finiteness.

Output:

- Next observation.
- Scalar reward.
- Termination flag.
- Truncation flag.
- Info metadata.

Consumers:

- Stable-Baselines3 training loop.

---

## 10. Execution Flow

Phase 5 uses the verified script entry point.

### Verified execution path

`scripts/train_rl.py -> run_rl_training_pipeline -> Phase 2 simulation environment -> RL policy training -> Checkpoint file`

Phase 5 implements the missing behavior in `run_rl_training_pipeline` and any verified lower-level skeleton units.

### Direct validation path

For Phase 5 validation, tests may directly call `run_rl_training_pipeline` or located lower-level training units. This avoids depending on unimplemented downstream evaluation behavior.

### Error propagation

Errors must propagate naturally:

- Missing robot XML propagates filesystem errors.
- Missing YCB root propagates filesystem errors when required.
- Invalid dimensions propagate value errors.
- Simulation initialization failures propagate simulation errors.
- Checkpoint saving failures propagate filesystem errors.

Phase 5 must not swallow errors silently.

---

## 11. Configuration Changes

Phase 5 requires no YAML configuration additions.

### Existing configuration preserved

| File | Key area | Existing semantics | Required Phase 5 treatment | Consumed by |
| --- | --- | --- | --- | --- |
| `configs/training.yaml` | RL training hyperparameters | Documented training settings. | Preserve file and documented semantics. Do not introduce global loading. | Future orchestration if it explicitly consumes configuration. |

### Configuration changes not allowed

Phase 5 must not:

- Add a new YAML schema.
- Add environment-variable configuration.
- Add global config state.
- Add config parser utilities.
- Add default values in code that shadow configuration values unless the existing skeleton already defines such defaults.

Hyperparameters must remain explicit through the verified CLI pipeline arguments.

---

## 12. Test Strategy

Testing must use `pytest`, which is already a development dependency.

No existing test suite was verified. Therefore, the implementation engineer must first check whether a test directory and existing conventions are present.

If tests already exist, Phase 5 tests must follow the existing conventions for:

- File naming.
- Test function naming.
- Fixture usage.
- Assertion style.
- Parameterization.
- Mocking, if already used.

If no tests exist, Phase 5 may add the minimum required test file under a conventional unit-test location. This is a test-only addition.

### Testing principles

1. Tests must be deterministic enough for CI.
2. Tests must not require network access.
3. Tests must not require downloaded datasets.
4. Tests must not require YCB assets unless Phase 2 tests already require them.
5. Tests must not require GPUs.
6. Tests must not require real robot hardware.
7. Tests must not introduce global state.
8. Tests must not create helper functions or helper classes.
9. Tests must use tiny training durations and tiny network widths.
10. Tests must avoid long RL training runs.

### Asset strategy

Because external assets are not guaranteed, tests must use the smallest possible local simulation configuration.

The preferred approach is:

- Use the minimal simulation description established by Phase 2 tests.
- Use no external dataset.
- Avoid committing binary assets.
- Avoid creating shared helper fixture code unless an existing fixture already provides the same behavior.

If Phase 2 simulation cannot initialize without external YCB assets, Phase 5 validation is blocked until a minimal testable simulation contract is available.

---

## 13. Test Suite

The following tests are required. Rows that depend on unverified skeleton symbols are marked accordingly.

| Test | File | Target | Scenario | Expected Result | Regression Risk |
| --- | --- | --- | --- | --- | --- |
| `test_phase1_package_import_remains_stable` | Existing or minimum required Phase 5 unit test file | `src/grasping_ai/__init__.py` | Import `grasping_ai`. | Import succeeds. | High if Phase 5 changes package initialization. |
| `test_stable_baselines_dependency_available` | Existing or minimum required Phase 5 unit test file | `pyproject.toml` dependency set | Import Stable-Baselines3. | Import succeeds. | Medium if dependency configuration changes. |
| `test_training_config_file_exists` | Existing or minimum required Phase 5 unit test file | `configs/training.yaml` | Check file existence. | File exists. | Low. |
| `test_run_rl_pipeline_rejects_missing_robot_xml` | Existing or minimum required Phase 5 unit test file | `run_rl_training_pipeline` | Provide a missing robot XML path. | Raises filesystem or value error. | Medium. |
| `test_run_rl_pipeline_rejects_missing_ycb_root_when_objects_requested` | Existing or minimum required Phase 5 unit test file | `run_rl_training_pipeline` | Provide object IDs with a missing YCB root. | Raises filesystem or value error. | Medium. |
| `test_run_rl_pipeline_initializes_environment` | Existing or minimum required Phase 5 unit test file | `run_rl_training_pipeline` and Phase 2 environment | Initialize environment with minimal simulation inputs. | Environment initializes and exposes observation/action spaces. | High. |
| `test_run_rl_pipeline_validates_observation_dim` | Existing or minimum required Phase 5 unit test file | `run_rl_training_pipeline` | Provide an observation dimension incompatible with the environment. | Raises value error. | High. |
| `test_run_rl_pipeline_validates_action_dim` | Existing or minimum required Phase 5 unit test file | `run_rl_training_pipeline` | Provide an action dimension incompatible with the environment. | Raises value error. | High. |
| `test_env_step_reward_is_finite` | Existing or minimum required Phase 5 unit test file | Unverified environment step or reward behavior | Reset environment, step with a valid action. | Reward is finite. | High. |
| `test_run_rl_pipeline_performs_minimal_training_and_saves_checkpoint` | Existing or minimum required Phase 5 unit test file | `run_rl_training_pipeline` | Run one tiny training update and save checkpoint. | Training completes and checkpoint artifact exists. | High. |
| `test_rl_checkpoint_is_loadable_or_discoverable` | Existing or minimum required Phase 5 unit test file | Unverified checkpoint saving/loading behavior | Save a tiny checkpoint and load or locate it. | Checkpoint can be loaded or discovered by existing loading behavior. | High. |
| `test_rl_training_does_not_leak_global_state` | Existing or minimum required Phase 5 unit test file | Pipeline and environment units | Run initialization twice. | Second run does not depend on hidden state from the first run. | Medium. |

---

## 14. Regression Test Plan

Phase 5 can introduce regressions in package imports, simulation contracts, dependency resolution, and downstream checkpoint expectations.

### Unit-level regressions

| Existing behavior | Why it could break | Protecting test | Expected result |
| --- | --- | --- | --- |
| Package import succeeds. | RL modules may introduce import cycles or heavy imports. | `test_phase1_package_import_remains_stable` | Import succeeds. |
| Stable-Baselines3 remains importable. | Dependency changes may affect resolution. | `test_stable_baselines_dependency_available` | Import succeeds. |
| Phase 2 environment step contract remains stable. | Reward modification may accidentally change step outputs. | Phase 2 step tests and Phase 5 reward test. | Step returns observation, reward, termination, truncation, and info consistently. |

### Integration-level regressions

| Existing behavior | Why it could break | Protecting test | Expected result |
| --- | --- | --- | --- |
| `scripts/train_rl.py` can import the pipeline. | Pipeline implementation may introduce import errors. | Existing script import behavior or direct pipeline import test. | Import path remains valid. |
| Downstream evaluation can expect a policy checkpoint. | Checkpoint saving format may become unstable. | Checkpoint creation and loadability tests. | Checkpoint contract remains stable. |

### Pipeline-level regressions

No full evaluation pipeline behavior is verified yet. Therefore, no pipeline-level regression test can be specified without invention.

If an existing pipeline smoke test exists, it must be rerun. If no such test exists, Phase 5 must not create a broad pipeline smoke test.

---

## 15. Impact on Preceding Phases

### Phase 1 — Foundation & Math Primitives

| Item | Description |
| --- | --- |
| Existing contract | Phase 1 established package importability, dependency availability, static configuration preservation, and pure math behavior where verified. |
| What Phase 5 changes | Phase 5 adds RL training behavior that uses existing dependencies. |
| What remains compatible | Package layout, existing dependencies, static configuration files, and Phase 1 math behavior must remain unchanged. |
| Previous-phase tests to rerun | Phase 1 package import test and dependency tests. |
| Whether Phase 1 needs modification | No modification is expected unless dependency resolution fails. |

### Phase 2 — Simulation & Robotics Core

| Item | Description |
| --- | --- |
| Existing contract | Phase 2 established simulation initialization, reset, step, observation, action, and basic robotics behavior where verified. |
| What Phase 5 changes | Phase 5 may modify reward behavior inside the existing environment step unit. |
| What remains compatible | Observation space, action space, reset contract, and step return structure must remain compatible. |
| Previous-phase tests to rerun | Phase 2 environment initialization, reset, step, invalid action, and observation stability tests. |
| Whether Phase 2 needs modification | Only the existing reward behavior may need modification. No other Phase 2 modification is expected. |

### Phase 3 — Data Pipeline & Perception

| Item | Description |
| --- | --- |
| Existing contract | Phase 3 established dataset indexing, point-cloud loading, and perception preprocessing. |
| What Phase 5 changes | Phase 5 does not directly modify Phase 3 behavior. |
| What remains compatible | Phase 3 data behavior remains unchanged. |
| Previous-phase tests to rerun | Phase 3 tests if they are part of the broader relevant suite. |
| Whether Phase 3 needs modification | No modification is expected. |

### Phase 4 — Generative Grasp Model

| Item | Description |
| --- | --- |
| Existing contract | Phase 4 established generative grasp model training and inference behavior where verified. |
| What Phase 5 changes | Phase 5 does not directly modify Phase 4 behavior. |
| What remains compatible | Generative model behavior remains unchanged. |
| Previous-phase tests to rerun | Phase 4 tests if they are part of the broader relevant suite. |
| Whether Phase 4 needs modification | No modification is expected. |

---

## 16. Impact on Downstream Phases

### Downstream Phase 6 — End-to-End Orchestration & Evaluation

| Item | Description |
| --- | --- |
| Dependency | Phase 6 may depend on the RL policy checkpoint for policy execution or evaluation. |
| Interface relied on | Checkpoint path, observation dimension, action dimension, and environment compatibility. |
| New behavior available after Phase 5 | A trained RL policy checkpoint exists and can be used in simulation. |
| Constraints preserved | Checkpoint must be loadable or discoverable by existing inference behavior. Observation and action dimensions must remain stable. |
| Protecting tests | Checkpoint creation, checkpoint loadability, observation dimension, and action dimension tests. |

### Downstream inference or policy-runner behavior

| Item | Description |
| --- | --- |
| Dependency | README documents an inference policy runner in the RL flow. |
| Interface relied on | RL policy checkpoint and environment interface. |
| New behavior available after Phase 5 | A policy checkpoint is available for later inference. |
| Constraints preserved | The checkpoint contract must remain explicit and stable. |
| Protecting tests | Checkpoint round-trip or loadability test. |

---

## 17. Cross-Phase Contract

Phase 5 establishes the following contract.

### Inputs

1. Training pipeline:
   - Robot XML path.
   - YCB root path.
   - Object identifier list.
   - Policy checkpoint path.
   - Observation dimension.
   - Action dimension.
   - Hidden dimension.
   - Learning rate.
   - Number of updates.
   - Gamma.
   - Device identifier.

2. Environment interaction:
   - Observation compatible with the environment observation space.
   - Action compatible with the environment action space.

### Outputs

1. Training pipeline:
   - Policy checkpoint artifact.

2. Environment step:
   - Observation.
   - Finite scalar reward.
   - Termination flag.
   - Truncation flag.
   - Info metadata.

3. Policy:
   - Action output compatible with the environment action space.

### Expected behavior

1. Training pipeline validates explicit arguments.
2. Environment observation and action dimensions are checked before training.
3. Reward is finite.
4. Policy checkpoint is saved explicitly.
5. No global state is introduced.

### Invariants

1. No module-level mutable state.
2. No global configuration state.
3. No dependency on real robot hardware.
4. Observation and action contracts remain stable.
5. Checkpoint artifact is discoverable after training.

### Error behavior

1. Missing robot XML raises explicit filesystem or value errors.
2. Missing YCB root raises explicit filesystem or value errors when required.
3. Dimension mismatch raises value errors.
4. Unsupported device raises explicit errors.
5. Non-finite reward is not returned.

### Configuration assumptions

1. YAML files remain static.
2. Phase 5 does not parse YAML globally.
3. Hyperparameters are explicit through the pipeline arguments.

### Data assumptions

1. Simulation environment is available from Phase 2.
2. Environment observation is numeric.
3. Environment action is compatible with the policy.
4. No external dataset is required for RL training in Phase 5 tests.

### Performance assumptions

1. Single-environment training is sufficient for Phase 5.
2. Parallel environments are out of scope.
3. GPU execution is optional.
4. Tests must use tiny training durations.

---

## 18. Validation Before Commit

The implementation must not be considered ready for commit until the following sequence is complete.

1. Run Phase 5 dependency tests.
2. Run Phase 5 package import regression tests.
3. Run Phase 5 configuration existence tests.
4. Run pipeline argument validation tests.
5. Run environment initialization tests.
6. Run observation and action dimension validation tests.
7. Run environment reward finiteness tests.
8. Run minimal training execution tests.
9. Run checkpoint creation tests.
10. Run checkpoint loadability or discoverability tests.
11. Rerun Phase 1 tests.
12. Rerun Phase 2 environment tests.
13. Run the broader relevant test suite.
14. Verify that no unrelated module changed.
15. Verify that no new class was introduced.
16. Verify that no helper or utility function was introduced.
17. Verify that no global variable or constant was introduced.
18. Verify that no file-level source description was introduced.
19. Verify that every modified or newly introduced function or method has a Google-style docstring.
20. Verify that no unrelated refactoring is included.
21. Verify that no new dependency was introduced.
22. Verify that downstream contracts remain compatible.

The final commit gate is:

`Implementation -> Targeted Tests -> Regression Tests -> Integration Tests -> Full Relevant Suite -> Static/Structural Review -> Commit`

---

## 19. Commit Boundary

The Phase 5 commit must contain only the following.

### Allowed in the commit

1. Modifications to `run_rl_training_pipeline`.
2. Modifications to verified existing RL training skeleton units.
3. Modifications to verified existing RL policy skeleton units if present.
4. Modifications to verified existing environment reward behavior if required.
5. Modifications to verified existing checkpoint saving/loading behavior if present.
6. Required Phase 5 tests.
7. Minimal test-only simulation assets if required by the existing testing convention and no existing asset is available.

### Not allowed in the commit

1. New RL algorithm classes.
2. New policy wrapper classes.
3. New environment wrapper classes.
4. New helper functions.
5. New utility modules.
6. New global constants.
7. New global variables.
8. New configuration loaders.
9. Unrelated formatting changes.
10. Refactoring of scripts unrelated to Phase 5.
11. Supervised model training code.
12. Evaluation metric code.
13. Real-robot integration code.
14. Temporary debugging code.
15. Experimental code.

---

## 20. Implementation Risks

| Risk | Severity | Cause | Affected Component | Detection Method | Mitigation |
| --- | --- | --- | --- | --- | --- |
| Exact training skeleton units are unverified. | High | Repository snapshot did not expose full contents of training or pipeline modules. | RL training implementation. | Manual source inspection before implementation. | Modify only located existing units. If absent, pause for scope decision. |
| Exact RL policy skeleton is unverified. | High | README mentions RL policy, but source files were not verified. | Policy implementation. | Manual source inspection. | Use existing policy skeleton if present. Otherwise use existing Stable-Baselines3 behavior without defining a new class. |
| Reward state availability is unverified. | High | Phase 2 may expose only minimal state. | Reward behavior. | Environment step tests. | Implement finite reward from available state. If insufficient state, pause for scope decision. |
| Stable-Baselines3 algorithm selection is unverified. | Medium | Documentation states RL algorithm is not fixed. | Training pipeline. | Source inspection and design decision. | Use a minimal stable algorithm only if no existing skeleton selection exists. |
| Mapping `num_updates` to Stable-Baselines3 training duration is ambiguous. | Medium | CLI uses updates, while SB3 commonly uses timesteps. | Training duration. | Minimal training test. | Preserve existing skeleton mapping if present. Otherwise map to the closest supported SB3 behavior and document the assumption. |
| Checkpoint format may be unstable. | High | SB3 native checkpoints may differ from repository expectations. | Downstream inference. | Checkpoint loadability test. | Preserve existing checkpoint contract if present. Otherwise use SB3 native saving and document the contract. |
| Environment dimension mismatch. | High | CLI passes observation and action dimensions that may not match env spaces. | Training pipeline. | Dimension validation tests. | Validate dimensions before training and fail explicitly. |
| Creating new classes would violate constraints. | High | RL workflows often use wrappers or custom policies. | Phase 5 implementation. | Structural review. | Use existing classes and functions only. Do not subclass or wrap. |

---

## 21. Design Decisions

| Decision | Evidence | Alternatives Considered | Reason for Selection |
| --- | --- | --- | --- |
| Implement Phase 5 primarily inside `run_rl_training_pipeline`. | Verified import in `scripts/train_rl.py`. | Move logic into script or create a new trainer module. | Preserves thin CLI pattern and verified call site. |
| Use Stable-Baselines3 as the RL backend. | Existing dependency and README RL flow. | Implement custom RL loop from scratch. | Minimal change and uses existing dependency. |
| Use an existing built-in Stable-Baselines3 policy if no custom policy skeleton exists. | No custom policy source verified. | Create a new policy class. | Avoids violating no-new-class constraint. |
| Use explicit CLI-provided dimensions for validation. | Verified CLI arguments include observation and action dimensions. | Infer dimensions silently from environment only. | Prevents contract mismatch and preserves explicit interface. |
| Implement reward inside the existing environment step behavior where possible. | Gymnasium step contract returns reward. | Compute reward outside environment or wrap environment. | Avoids wrappers and preserves environment contract. |
| Do not parse YAML configuration in Phase 5. | No config loader verified; CLI arguments already explicit. | Add global config loader. | Avoids global state and unnecessary dependency. |
| Save checkpoint using existing skeleton behavior if present, otherwise Stable-Baselines3 native saving. | Checkpoint path is explicit in CLI. | Invent a new checkpoint format. | Minimal change and preserves downstream compatibility. |
| Keep training single-environment. | No parallel environment behavior verified. | Add vectorized environments. | Parallelization is optimization and out of scope. |

---

## 22. Explicitly Rejected Changes

The following changes were considered and rejected.

### Rejected: creating a new RL algorithm class

A custom algorithm class would allow precise control over updates, but it violates the no-new-class constraint and is unnecessary because Stable-Baselines3 already exists.

### Rejected: creating an environment wrapper class

Wrappers are common in Gymnasium workflows, but they violate the no-new-class constraint. Required behavior must be implemented inside the existing environment or pipeline units.

### Rejected: creating helper functions for reward calculation

Reward calculation is required, but creating separate helper utilities would violate the no-helper constraint. Required reward behavior must be implemented directly inside the appropriate existing environment unit.

### Rejected: adding a global configuration loader

A config loader would make `configs/training.yaml` easier to use, but Phase 5 does not require global configuration. It would introduce hidden state and unnecessary dependency risk.

### Rejected: adding experiment tracking integration

TensorBoard and W&B dependencies exist, but Phase 5 does not require experiment dashboards. Adding tracking would expand scope and introduce side effects.

### Rejected: adding parallel environments

Parallel environments are performance optimizations. They are not required for Phase 5 correctness and would complicate the environment contract.

### Rejected: adding domain randomization

Domain randomization is a sim-to-real or robustness concern. It is not required for the minimum RL training capability.

### Rejected: modifying `scripts/train_rl.py` unnecessarily

The script already delegates to the pipeline function. Modifying it would risk breaking the verified CLI contract.

---

## 23. Verification Evidence

| Claim | Evidence |
| --- | --- |
| `scripts/train_rl.py` exists. | Verified script content. |
| `scripts/train_rl.py` imports `run_rl_training_pipeline`. | Verified import statement. |
| `scripts/train_rl.py` passes explicit RL arguments. | Verified script argument list. |
| Stable-Baselines3 is a runtime dependency. | Verified `pyproject.toml`. |
| Gymnasium is a runtime dependency. | Verified `pyproject.toml`. |
| Torch is a runtime dependency. | Verified `pyproject.toml`. |
| MuJoCo is a development dependency. | Verified `pyproject.toml`. |
| RL policy models are intended under `src/grasping_ai/models/`. | README repository layout. |
| RL training loops are intended under `src/grasping_ai/training/`. | README repository layout. |
| RL training flow involves simulation, policy runner, trainer, and checkpoints. | README end-to-end workflow. |
| Exact training module contents are unverified. | Available snapshot did not expose full source under relevant directories. |
| Exact RL policy skeleton is unverified. | Available snapshot did not expose contents of `src/grasping_ai/models/`. |
| Exact reward implementation is unverified. | Available snapshot did not expose contents of `src/grasping_ai/simulation/`. |

---

## 24. Definition of Done

Phase 5 is considered complete only when:

- All required source units have been implemented.
- No unnecessary source units were modified.
- Existing architecture patterns are preserved.
- No helper/utility function was introduced.
- No helper/utility class was introduced.
- No new class was introduced.
- No global variable was introduced.
- No global constant was introduced.
- No file-level source description was introduced.
- Every modified/new function has a Google-style docstring.
- Targeted tests pass.
- Regression tests pass.
- Relevant integration tests pass.
- The broader relevant test suite passes.
- Preceding-phase behavior remains intact.
- Downstream contracts remain compatible.
- No unrelated refactoring is included.
- The implementation has been verified against the repository.
- The phase is safe to commit.

---

## 25. Implementation Checklist

### Source Changes

- [ ] Verified `src/grasping_ai/pipelines/train_rl.py` contains `run_rl_training_pipeline`.
- [ ] Implemented `run_rl_training_pipeline`.
- [ ] Verified existing RL training skeleton units if present.
- [ ] Implemented or connected existing RL training behavior.
- [ ] Verified existing RL policy skeleton units if present.
- [ ] Implemented existing RL policy behavior or used existing Stable-Baselines3 policy behavior.
- [ ] Verified existing environment reward behavior.
- [ ] Implemented finite reward behavior inside the existing environment step unit if required.
- [ ] Verified checkpoint saving/loading behavior.
- [ ] Implemented checkpoint saving behavior.
- [ ] Did not create new classes.
- [ ] Did not create helper functions.
- [ ] Did not create utility modules.
- [ ] Did not introduce global variables.
- [ ] Did not introduce global constants.
- [ ] Did not add file-level descriptions.
- [ ] Added Google-style docstrings to modified functions or methods.

### Tests

- [ ] Added package import regression test.
- [ ] Added Stable-Baselines3 dependency availability test.
- [ ] Added training configuration existence test.
- [ ] Added missing robot XML failure test.
- [ ] Added missing YCB root failure test when object IDs are provided.
- [ ] Added environment initialization test.
- [ ] Added observation dimension validation test.
- [ ] Added action dimension validation test.
- [ ] Added reward finiteness test.
- [ ] Added minimal training execution test.
- [ ] Added checkpoint creation test.
- [ ] Added checkpoint loadability or discoverability test.
- [ ] Added no-global-state test.

### Regression Verification

- [ ] Phase 1 package import remains successful.
- [ ] Phase 1 dependency tests remain successful.
- [ ] Phase 2 environment initialization remains successful.
- [ ] Phase 2 reset and step tests remain successful.
- [ ] Phase 2 observation and action contracts remain unchanged.
- [ ] No import cycles were introduced.
- [ ] No module-level side effects were introduced.

### Cross-Phase Verification

- [ ] Phase 6 can rely on a policy checkpoint artifact.
- [ ] Phase 6 can rely on stable observation and action dimensions.
- [ ] RL pipeline does not import real-robot modules.
- [ ] Policy behavior does not depend on pipelines or CLI scripts.
- [ ] Environment reward remains part of the environment step contract.

### Structural Constraints

- [ ] No new RL algorithm class exists.
- [ ] No new policy wrapper class exists.
- [ ] No new environment wrapper class exists.
- [ ] No helper abstraction exists.
- [ ] No utility abstraction exists.
- [ ] No global configuration abstraction exists.
- [ ] No unrelated refactoring exists.
- [ ] No experimental code exists.

### Commit Readiness

- [ ] Targeted Phase 5 tests pass.
- [ ] Regression tests pass.
- [ ] Broader relevant test suite passes.
- [ ] Static and structural review confirms forbidden constructs are absent.
- [ ] Commit contains only Phase 5 changes.