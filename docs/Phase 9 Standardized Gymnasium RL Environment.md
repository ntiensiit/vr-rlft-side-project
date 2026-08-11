# Phase 9: Standardized Gymnasium RL Environment

## 1. Phase Overview

**Phase name:** Phase 9: Standardized Gymnasium RL Environment

**Objective:** Replace the repository-local, ad hoc MuJoCo rollout path used by the RL training pipeline with a standardized, locally testable `gymnasium.Env` implementation and train the policy through the already-declared `stable-baselines3` dependency. The phase must preserve the existing inference contract used by `src/grasping_ai/inference/policy_runner.py` as much as possible.

**Why this phase is necessary based on verified repository state:**

The current repository contains a custom RL pipeline that bypasses the declared RL infrastructure.

Verified evidence:

- `pyproject.toml` declares `gymnasium>=1.3.0` and `stable-baselines3>=2.9.0`, but the inspected RL pipeline does not use either dependency.
- `src/grasping_ai/pipelines/train_rl.py` defines `build_rl_environment` and `collect_rl_rollout`, which manually step MuJoCo, construct observations from `qpos` and `qvel`, pad or truncate actions, and compute a hardcoded reward.
- `src/grasping_ai/training/rl_trainer.py` implements a custom policy update step rather than using a standard RL algorithm implementation.
- `src/grasping_ai/simulation/mujoco_env.py` provides a functional MuJoCo wrapper with `load_mujoco_model`, `create_simulation`, `reset_simulation`, and body/joint accessors, but it does not expose a Gymnasium-compatible `reset`, `step`, observation space, or action space.
- `src/grasping_ai/pipelines/train_rl.py::run_rl_training_pipeline` accepts `observation_dim` and `action_dim`, validates them against the MuJoCo model, builds a policy with `src/grasping_ai/models/rl_policy.py::build_policy_network`, and then uses the custom rollout loop.
- `src/grasping_ai/pipelines/train_rl.py::build_rl_environment` accepts `ycb_root` and `object_id`, but the inspected implementation loads only the robot XML and does not attach or use the YCB object. This is an existing inconsistency.
- `src/grasping_ai/inference/policy_runner.py::build_rl_policy_runner` expects a checkpoint containing a Torch state dict compatible with `build_policy_network`.

**Expected outcome:**

After this phase:

- The repository contains one Gymnasium-compatible MuJoCo environment class owned by the simulation module.
- The environment exposes explicit observation and action spaces.
- The environment reset and step behavior is locally testable without physical hardware.
- `run_rl_training_pipeline` uses the Gymnasium environment and `stable-baselines3` instead of the custom rollout generator and custom update step.
- The trained policy is exported to a checkpoint format that remains loadable by the existing `policy_runner` inference path.
- Existing MuJoCo functional primitives remain available and unchanged.
- No physical robot, network service, or external dataset is required.

## 2. Verified Current State

### Verified implemented

- **Functional MuJoCo simulation wrapper**
  - File: `src/grasping_ai/simulation/mujoco_env.py`
  - Implemented functions include:
    - `load_mujoco_model`
    - `create_simulation`
    - `reset_simulation`
    - `read_joint_positions`
    - `set_joint_positions`
    - `read_body_pose`
  - Behavior: loads an MJCF model, creates an opaque state dictionary containing MuJoCo model and data, provides stepping and contact reporting callables, and supports reset and pose queries.

- **RL policy network construction**
  - File: `src/grasping_ai/models/rl_policy.py`
  - `build_policy_network` constructs a Torch sequential policy mapping observation vectors to action vectors.
  - `build_value_network` exists but is not used by the current custom RL update path.
  - `select_action` exists but is not used by the inspected pipeline rollout path.

- **Custom RL training pipeline**
  - File: `src/grasping_ai/pipelines/train_rl.py`
  - `run_rl_training_pipeline` validates paths and dimensions, builds a MuJoCo simulation state, constructs a policy, constructs an optimizer, and trains through a custom rollout generator.
  - `build_rl_environment` loads a MuJoCo model from the robot XML and returns the simulation state, step callable, and contact reporter.
  - `collect_rl_rollout` manually resets the simulation, reads `qpos` and `qvel`, calls the policy, pads or truncates the action, steps MuJoCo, and computes a reward.

- **Custom RL update logic**
  - File: `src/grasping_ai/training/rl_trainer.py`
  - `build_rl_training_step`, `run_rl_training_loop`, `compute_discounted_returns`, and `compute_gae_advantages` are implemented.
  - The custom update step computes discounted returns and applies a clipped advantage-style loss.

- **RL policy inference**
  - File: `src/grasping_ai/inference/policy_runner.py`
  - `load_rl_policy_checkpoint` loads a Torch checkpoint.
  - `build_rl_policy_runner` reconstructs a policy using `build_policy_network` and expects a checkpoint field named `model_state_dict`.

- **CLI entry point**
  - File: `scripts/train_rl.py`
  - Parses CLI arguments and delegates to `run_rl_training_pipeline`.

### Partially implemented

- **RL environment abstraction**
  - File: `src/grasping_ai/pipelines/train_rl.py`
  - A closed-loop rollout environment exists only as procedural code inside `build_rl_environment` and `collect_rl_rollout`.
  - It does not expose standardized reset, step, observation space, action space, termination, or truncation semantics.

- **YCB object usage in RL environment**
  - File: `src/grasping_ai/pipelines/train_rl.py`
  - `run_rl_training_pipeline` validates `ycb_root` and `object_ids`, but `build_rl_environment` does not attach or use the object.
  - The object-related arguments are present but do not affect the environment state.

### Missing

- **Gymnasium environment implementation**
  - No inspected file defines a `gymnasium.Env` subclass.
  - No observation space or action space is defined.

- **stable-baselines3 integration**
  - No inspected pipeline imports or uses `stable_baselines3`.
  - No standard RL algorithm is used.

- **Standardized termination and truncation semantics**
  - The custom rollout uses a `done` flag when the next observation is non-finite, but it does not distinguish terminated and truncated states.

- **Checkpoint export compatibility for standard RL training**
  - If training were moved to `stable-baselines3`, the native checkpoint format would not match the existing `policy_runner` expectation unless an explicit export step is added.

### Inconsistent

- **Unused object arguments**
  - `run_rl_training_pipeline` accepts and validates `ycb_root` and `object_ids`, but the current environment construction does not attach or manipulate the YCB object.

- **Declared dependencies unused**
  - `gymnasium` and `stable-baselines3` are declared dependencies, but the active RL training path uses neither.

- **Custom RL update versus declared dependency strategy**
  - The repository declares `stable-baselines3`, but the pipeline implements a local update rule instead of using the dependency.

### Unused or dead code relevant to this phase

- **`select_action`**
  - File: `src/grasping_ai/models/rl_policy.py`
  - Implemented, but the inspected rollout path calls the policy directly rather than using this sampling function.

- **`compute_gae_advantages`**
  - File: `src/grasping_ai/training/rl_trainer.py`
  - Implemented, but the inspected custom update step does not use it.

- **`build_value_network`**
  - File: `src/grasping_ai/models/rl_policy.py`
  - Implemented, but the current custom training step does not use a value network.

## 3. Phase Boundary

### Exact scope of what this phase modifies

This phase owns the following work:

1. Add one Gymnasium-compatible MuJoCo environment class to the existing simulation module.
2. Define observation and action spaces for that environment using the existing MuJoCo model dimensions and actuator limits.
3. Implement deterministic local reset and step behavior on top of the existing functional MuJoCo wrapper.
4. Modify `run_rl_training_pipeline` to construct the Gymnasium environment and train with `stable-baselines3` instead of the custom rollout generator and custom update step.
5. Export the trained policy into a checkpoint format compatible with the existing `policy_runner` inference path.
6. Add optional explicit seed support to the RL pipeline and CLI if such support is not already present, solely for environment and algorithm initialization.
7. Add tests validating the Gymnasium environment, stable-baselines3 training path, and exported checkpoint compatibility.

### Explicitly excluded responsibilities

This phase does not implement:

- YCB object attachment to the RL scene.
- Grasp-specific reward design.
- Lift-success reward design.
- Contact-rich manipulation objectives.
- Changes to generative grasp models.
- Changes to supervised training.
- Changes to evaluation metrics.
- Changes to force closure or collision checking.
- Experiment tracking integration.
- Hyperparameter search.
- Real robot deployment.
- Hardware communication.
- Physical sensor integration.
- Rendering infrastructure.
- New configuration loaders.
- YAML parsing.

### What must NOT be touched

The following must remain unchanged unless directly required for signature compatibility:

- The functional MuJoCo primitives in `src/grasping_ai/simulation/mujoco_env.py`:
  - `load_mujoco_model`
  - `create_simulation`
  - `reset_simulation`
  - `read_joint_positions`
  - `set_joint_positions`
  - `read_body_pose`
- The existing policy network contract in `src/grasping_ai/models/rl_policy.py::build_policy_network`.
- The existing inference expectation that `build_rl_policy_runner` can reconstruct a policy from a Torch state dictionary under the `model_state_dict` key.
- The existing CLI contract required by `scripts/train_rl.py` unless new optional arguments are added.
- The existing data contracts for generated grasps, point clouds, or evaluation reports.

### MECE boundary definition

This phase owns **standardized local RL environment construction and standard algorithm integration**.

It does not own:

- The grasp generation model.
- The supervised training pipeline.
- The evaluation pipeline.
- The YCB object manipulation task design.
- Physical deployment.
- Experiment observability.

## 4. Architecture and Dependency Analysis

### Real dependency flow before this phase

Current RL execution flow:

- `scripts/train_rl.py`
  - calls `src/grasping_ai/pipelines/train_rl.py::run_rl_training_pipeline`
    - calls `build_rl_environment`
      - calls `src/grasping_ai/simulation/mujoco_env.py::load_mujoco_model`
      - calls `src/grasping_ai/simulation/mujoco_env.py::create_simulation`
    - validates observation and action dimensions from the MuJoCo model
    - calls `src/grasping_ai/models/rl_policy.py::build_policy_network`
    - calls `src/grasping_ai/training/rl_trainer.py::build_rl_training_step`
    - calls `src/grasping_ai/training/rl_trainer.py::run_rl_training_loop`
      - repeatedly calls `collect_rl_rollout`
        - directly steps MuJoCo
        - computes observations and rewards
    - saves a checkpoint

### Required dependency flow after this phase

New RL execution flow:

- `scripts/train_rl.py`
  - calls `src/grasping_ai/pipelines/train_rl.py::run_rl_training_pipeline`
    - validates paths and dimensions
    - constructs the Gymnasium environment class from `src/grasping_ai/simulation/mujoco_env.py`
      - environment internally uses existing `load_mujoco_model` and `create_simulation`
    - validates observation and action dimensions against Gymnasium spaces
    - constructs a `stable_baselines3` PPO algorithm instance
    - trains through the Gymnasium environment
    - exports a legacy-compatible policy checkpoint
    - writes checkpoint to the requested path

### Input, processing, output chain

#### Environment input

- Robot MJCF path.
- Optional reset seed.
- Action vector supplied by the RL algorithm.

#### Environment processing

- Load MuJoCo model through existing functional wrapper.
- Reset simulation state.
- Read `qpos` and `qvel` as observation.
- Apply action to MuJoCo control vector.
- Step simulation.
- Compute reward.
- Detect non-finite state and terminate safely.

#### Environment output

- Observation vector of shape `(nq + nv,)`.
- Scalar reward.
- Termination flag.
- Truncation flag.
- Info dictionary.

#### Training output

- Legacy-compatible policy checkpoint file.
- No required tracking artifacts.
- No required dataset artifacts.

### Data contracts

#### Observation

- Type: NumPy array.
- Dtype: `float32`.
- Shape: `(nq + nv,)`.
- Semantics: concatenation of MuJoCo `qpos` and `qvel`.
- Coordinate frame: MuJoCo world and joint state representation, unchanged.

#### Action

- Type: NumPy array.
- Dtype: `float32` at the Gymnasium space boundary.
- Shape: `(nu,)`.
- Semantics: MuJoCo actuator control commands.
- Bounds: derived from MuJoCo actuator control limits when available; otherwise finite default bounds.

#### Reward

- Scalar Python float.
- Preserves the existing local reward behavior:
  - negative quadratic action cost,
  - survival bonus when the next observation is finite,
  - clipping to a finite range.

#### Checkpoint

- Torch save file.
- Must contain a `model_state_dict` entry loadable by `build_policy_network`.
- Must remain readable by `src/grasping_ai/inference/policy_runner.py::load_rl_policy_checkpoint`.

### Cross-module interactions

- Simulation module gains a Gymnasium-facing class but must continue exposing the existing functional API.
- RL pipeline gains a dependency on `gymnasium` and `stable_baselines3` at execution time.
- Inference module remains unchanged but depends on the pipeline exporting a compatible checkpoint.
- Training module `rl_trainer.py` is no longer required by the standardized pipeline path but remains present for regression coverage of existing RL math.

## 5. Source Code Impact Analysis

### `src/grasping_ai/simulation/mujoco_env.py`

**Current responsibility:** Functional MuJoCo model loading, simulation stepping, reset, joint access, body pose access, contact reporting.

**Exact unit affected:** New Gymnasium environment class added to this module.

**Current behavior:** No Gymnasium environment exists.

**Required change:** Add one required environment class that wraps the existing functional simulation state and exposes `reset` and `step` semantics plus observation and action spaces.

**Reason for change:** `stable-baselines3` and Gymnasium require an environment object with standardized spaces and step/reset semantics. The existing functional API cannot satisfy this contract directly.

**Dependency impact:** Introduces use of the already-declared `gymnasium` dependency inside the simulation module. Does not require new packages.

**Regression risk:** Low to medium. Existing functional API remains unchanged, but module import now depends on Gymnasium.

### `src/grasping_ai/pipelines/train_rl.py`

**Current responsibility:** End-to-end RL training orchestration using a custom rollout path.

**Exact functions affected:**

- `run_rl_training_pipeline`
- `build_rl_environment`
- `collect_rl_rollout`

**Current behavior:**

- `run_rl_training_pipeline` uses `build_rl_environment` and `collect_rl_rollout`.
- `build_rl_environment` loads a robot-only MuJoCo model.
- `collect_rl_rollout` manually constructs observations, applies actions, and computes rewards.

**Required change:**

- Replace the active training path in `run_rl_training_pipeline` with the Gymnasium environment and `stable-baselines3`.
- Remove the active use of `build_rl_environment` and `collect_rl_rollout` from the pipeline.
- If repository tests still require those legacy functions, migrate their behavioral coverage to the new environment tests before removing them.
- Preserve path validation and dimension validation behavior.
- Export a legacy-compatible checkpoint after training.

**Reason for change:** The custom rollout path is non-standard and bypasses the declared Gymnasium and stable-baselines3 dependencies.

**Dependency impact:** Pipeline now depends on the new environment class and stable-baselines3. It no longer depends on the custom rollout path for end-to-end training.

**Regression risk:** Medium. RL training behavior changes because the algorithm implementation changes. Checkpoint export must protect inference compatibility.

### `scripts/train_rl.py`

**Current responsibility:** CLI entry point for RL training.

**Exact function affected:**

- `train_rl_main`
- Argument parser block in `__main__`

**Current behavior:** Passes existing arguments to `run_rl_training_pipeline`.

**Required change:** Add an optional seed argument only if the repository does not already provide one. Pass it to the pipeline. Preserve all existing required arguments.

**Reason for change:** Gymnasium reset and stable-baselines3 initialization support explicit seeding. Local deterministic testing requires an explicit seed path.

**Dependency impact:** Depends on updated pipeline signature.

**Regression risk:** Low if the seed argument is optional.

### `src/grasping_ai/inference/policy_runner.py`

**Current responsibility:** Load RL checkpoints and build a deterministic policy runner.

**Exact functions affected:** None required if checkpoint export remains compatible.

**Current behavior:** Loads a Torch checkpoint and expects `model_state_dict` compatible with `build_policy_network`.

**Required change:** No change is required if the new pipeline exports a compatible state dict. If export compatibility cannot be guaranteed, this file would need modification, but that should be treated as a failure of the preferred export strategy rather than the default design.

**Reason for change:** Preserve previous inference behavior.

**Dependency impact:** None if unchanged.

**Regression risk:** Low if unchanged.

### `src/grasping_ai/training/rl_trainer.py`

**Current responsibility:** Custom RL update step and rollout loop.

**Exact functions affected:** None required.

**Current behavior:** Implements a custom update step and training loop.

**Required change:** No modification is required for Phase 9. The standardized pipeline should no longer depend on this module for end-to-end training, but the module may remain for regression coverage of existing RL math.

**Reason for change:** Avoid unnecessary modification of Phase 5 math primitives while replacing the active training path.

**Dependency impact:** The active pipeline dependency on this module is removed or reduced.

**Regression risk:** Low for existing unit tests because the module remains unchanged.

### `src/grasping_ai/models/rl_policy.py`

**Current responsibility:** Policy and value network construction.

**Exact functions affected:** None.

**Current behavior:** `build_policy_network` constructs the policy network used by inference and the legacy pipeline.

**Required change:** None. The new pipeline must export a checkpoint compatible with this function.

**Reason for change:** Preserve inference compatibility.

**Dependency impact:** None.

**Regression risk:** Low.

## 6. New Source Code Units

### Required new class: Gymnasium MuJoCo environment

**Location:** `src/grasping_ai/simulation/mujoco_env.py`

**Responsibility:** Provide a standardized Gymnasium-compatible environment over the existing MuJoCo functional wrapper.

**Required methods:**

- Constructor:
  - Accepts the robot MJCF path.
  - Loads the MuJoCo model using existing `load_mujoco_model`.
  - Creates the simulation using existing `create_simulation`.
  - Builds observation and action spaces.
- Reset:
  - Accepts an optional seed and optional options mapping.
  - Resets the simulation using existing `reset_simulation`.
  - Returns the observation and an info dictionary.
- Step:
  - Accepts an action.
  - Validates finite action values.
  - Applies the action to the MuJoCo control vector.
  - Steps the simulation.
  - Returns observation, reward, terminated, truncated, and info.

**Inputs:**

- Robot MJCF path.
- Optional Gymnasium reset seed.
- Action vector.

**Outputs:**

- Observation vector.
- Scalar reward.
- Termination flag.
- Truncation flag.
- Info dictionary.

**Failure behavior:**

- Missing robot XML must raise a filesystem error through the existing loading path.
- Invalid MuJoCo model must raise a value error through the existing loading path.
- Non-finite action input must raise a value error.
- A model with no actuators must raise a value error because the RL environment requires a non-empty action space.

**Why existing code cannot be extended instead:**

The existing functional API cannot satisfy the Gymnasium contract because Gymnasium and stable-baselines3 require an object with observation space, action space, reset, and step methods. A class is therefore required by the external interface. This is not a helper class or utility class; it is the required environment abstraction for the phase.

### No other new source units

No new modules are required.

No helper functions are required.

No utility classes are required.

No new configuration objects are required.

## 7. Existing Code Modification Rules

### Preserve functional simulation invariants

The existing functional MuJoCo API must remain unchanged.

The new environment class must reuse the existing loading, simulation creation, and reset primitives rather than duplicating MuJoCo initialization logic.

### Preserve public inference contract

The trained policy checkpoint must remain loadable by `src/grasping_ai/inference/policy_runner.py` without requiring changes to that module.

If this cannot be achieved without modifying inference, the implementation must be considered blocked until the export strategy is corrected.

### Preserve tensor and data contracts

Observation vectors must remain one-dimensional NumPy arrays with dtype `float32`.

Action vectors must remain one-dimensional NumPy arrays at the environment boundary.

Checkpoint state dictionaries must remain compatible with Torch `load_state_dict` behavior.

### Preserve numerical behavior unless explicitly required

The environment reward should preserve the existing custom rollout reward behavior:

- negative quadratic action cost,
- survival bonus for finite next observation,
- finite clipping.

The replacement of the custom update step with stable-baselines3 necessarily changes RL optimization behavior. That change is explicitly required by this phase.

### No unrelated refactoring

Do not modify unrelated simulation, perception, data, generative-model, evaluation, or inference logic.

Do not extract helpers.

Do not rename existing public functions unless removing obsolete internal pipeline functions as part of the migration.

## 8. Detailed Behavioral Design

### Environment construction behavior

The environment constructor must:

1. Accept a robot MJCF path.
2. Validate that the path is a file using existing behavior from `load_mujoco_model`.
3. Load the MuJoCo model using `load_mujoco_model`.
4. Create simulation state using `create_simulation`.
5. Read MuJoCo model dimensions:
   - `nq` for generalized positions,
   - `nv` for generalized velocities,
   - `nu` for actuators.
6. Raise a value error if `nu` is zero.
7. Build an observation space with shape `(nq + nv,)` and dtype `float32`.
8. Build an action space with shape `(nu,)` and dtype `float32`.
9. Derive finite action bounds from actuator control ranges when those ranges are limited and finite.
10. Use finite default bounds for unlimited actuators.
11. Store the simulation state, step callable, and contact reporter needed by reset and step.

The constructor must not attach YCB objects. Object attachment is out of scope and would change observation dimensionality and simulation behavior.

### Reset behavior

Reset must:

1. Accept optional seed and options arguments according to the Gymnasium contract.
2. Use the Gymnasium base-class seeding behavior when a seed is provided.
3. Call the existing `reset_simulation` primitive.
4. Read current `qpos` and `qvel`.
5. Concatenate them into a `float32` observation vector.
6. Return the observation and an empty info dictionary.

Reset must not create files, mutate global repository state, or require network access.

### Step behavior

Step must:

1. Convert the supplied action to a NumPy array.
2. Flatten the action.
3. Reject non-finite action values with a value error.
4. If the action size is larger than `nu`, truncate it.
5. If the action size is smaller than `nu`, pad it with zeros.
6. Assign the resulting action to the MuJoCo control vector.
7. Step the simulation using the existing step callable.
8. Read the next `qpos` and `qvel`.
9. Construct the next observation as `float32`.
10. Compute reward as:
    - negative action cost proportional to squared action magnitude,
    - plus a survival bonus if the next observation is fully finite,
    - clipped to a finite range.
11. If the next observation is non-finite:
    - set terminated to true,
    - reset the simulation,
    - return the fresh finite initial observation rather than a non-finite observation.
12. Set truncated to false under normal local simulation conditions.
13. Return observation, reward, terminated, truncated, and an info dictionary.

Step must not silently ignore invalid action shapes. Padding and truncation are preserved only for size mismatch, not for non-finite values.

### Stable-baselines3 training behavior

`run_rl_training_pipeline` must:

1. Preserve existing validation for robot XML, YCB root, object identifiers, dimensions, hidden width, learning rate, update count, gamma, and device.
2. Construct the new Gymnasium environment.
3. Validate that `observation_dim` matches the environment observation space dimension.
4. Validate that `action_dim` matches the environment action space dimension.
5. Construct a stable-baselines3 PPO algorithm using:
    - the environment,
    - the supplied learning rate,
    - the supplied gamma,
    - a rollout length matching the existing local rollout size of 64 steps,
    - one optimization epoch per rollout to approximate the previous one-update-per-rollout structure,
    - a batch size equal to the rollout length,
    - the supplied device,
    - an optional explicit seed if provided,
    - a policy network configuration matching the existing hidden dimension and two hidden layers.
6. Train for a total number of environment steps equal to the requested number of updates multiplied by the rollout length.
7. Export a legacy-compatible policy checkpoint.
8. Write the checkpoint to the requested path.

The pipeline must not require YCB object attachment, contact-based rewards, or lift-success evaluation.

### Checkpoint export behavior

After stable-baselines3 training completes, the pipeline must:

1. Construct a legacy policy module using `build_policy_network` with the same observation dimension, action dimension, hidden dimension, and two hidden layers.
2. Transfer the deterministic mean-action weights from the stable-baselines3 policy into the legacy policy module.
3. Verify through tests that the legacy policy module produces actions compatible with the trained stable-baselines3 policy for the same observation.
4. Save a Torch checkpoint containing at least:
    - `model_state_dict` from the legacy policy module,
    - an epoch or update-count field,
    - any additional scalar metadata required for future inspection.
5. Ensure the checkpoint can be loaded by `load_rl_policy_checkpoint` and used by `build_rl_policy_runner` without modifying inference code.

### Edge cases

- Robot model has zero actuators: environment construction must fail.
- Action vector is empty: step must pad to zero controls if `nu` is positive.
- Action vector is too long: step must truncate to `nu`.
- Action contains NaN or infinity: step must raise a value error.
- Simulation state becomes non-finite: step must terminate and return a finite reset observation.
- Observation dimension argument does not match environment space: pipeline must raise a value error.
- Action dimension argument does not match environment space: pipeline must raise a value error.
- Requested update count is zero or negative: pipeline must raise a value error, preserving existing validation.
- Stable-baselines3 training terminates early because of an environment failure: checkpoint export must not occur unless training completed successfully.

### Failure cases

- Missing robot XML must raise a filesystem error.
- Invalid MJCF content must raise a value error from the existing MuJoCo loader.
- Invalid device strings must propagate naturally from Torch or stable-baselines3.
- Failure to export compatible weights must raise an explicit error rather than writing an unusable checkpoint.

## 9. Cross-Phase Impact

### Impact on Phase 1: Foundation and Math Primitives

- No math primitives are changed.
- Package import behavior must remain stable.
- Adding Gymnasium to the simulation module must not introduce global state.

Required backward compatibility:

- Existing imports of `grasping_ai.simulation.mujoco_env` functional API must continue to work.

### Impact on Phase 2: Simulation and Robotics Core

- Existing MuJoCo functional primitives remain unchanged.
- The new environment consumes those primitives.
- Scene assembly and YCB loading are not modified.

Required regression tests:

- Existing simulation tests must continue to pass.
- New environment tests must validate that the Gymnasium wrapper does not alter MuJoCo reset or step primitives.

### Impact on Phase 3: Data Pipeline and Perception

- No data loading or perception behavior is changed.

Required regression tests:

- Existing data and perception tests must continue to pass.

### Impact on Phase 4: Generative Grasp Model

- No generative model behavior is changed.

Required regression tests:

- Existing generative model tests must continue to pass.

### Impact on Phase 5: Reinforcement Learning Policy

- The policy network construction function remains unchanged.
- The custom RL trainer remains present but is no longer the active end-to-end training path.
- The trained policy checkpoint must remain compatible with the existing policy inference path.

Required regression tests:

- Existing policy network tests must continue to pass.
- Existing custom RL math tests may remain as regression coverage, but they no longer validate the supported end-to-end training path.

### Impact on Phase 6: End-to-End Orchestration and Evaluation

- Evaluation behavior is unchanged.
- Simulation-based grasp execution is unchanged.

Required regression tests:

- Existing orchestration and evaluation tests must continue to pass.

## 10. Regression Protection

### Unit tests

Required unit tests must cover:

- Environment construction from a minimal temporary MJCF model.
- Observation space shape and dtype.
- Action space shape, dtype, and finite bounds.
- Reset returns a finite observation and info dictionary.
- Step returns a five-element Gymnasium step tuple.
- Step with zero action returns finite observation and reward.
- Step rejects non-finite actions.
- Reset is deterministic across repeated calls.
- Environment passes Gymnasium environment checking without errors.

### Integration tests

Required integration tests must cover:

- Stable-baselines3 training for a tiny number of steps using the new environment.
- Checkpoint export after training.
- Loading the exported checkpoint through `load_rl_policy_checkpoint`.
- Building a policy runner through `build_rl_policy_runner`.
- Running the policy runner on an environment observation and receiving a finite action of the correct shape.

### Simulation tests

Simulation tests must validate:

- The environment uses the existing MuJoCo step callable.
- Reset restores the same initial observation.
- Repeated identical actions produce deterministic observations on CPU.

### Numerical validation tests

Required numerical validation:

- Observation equals concatenation of MuJoCo `qpos` and `qvel`.
- Reward remains finite.
- Reward preserves the existing action-cost and survival-bonus structure.
- Exported legacy policy produces finite actions and matches the trained stable-baselines3 deterministic policy behavior within tolerance for identical observations.

### Explicit protection of previous phases

Before committing, the full existing test suite must pass, including:

- Phase 1 foundation tests.
- Phase 2 simulation tests.
- Phase 3 data and perception tests.
- Phase 4 generative model tests.
- Phase 5 policy and RL math tests.
- Phase 6 evaluation and orchestration tests.

## 11. New Test Suite

### `tests/unit/test_phase9_gymnasium_env.py`

**Purpose:** Validate the Gymnasium environment class.

**Input strategy:**

- Create a minimal temporary MJCF model with at least one actuated joint.
- Construct the environment directly.
- Use fixed actions and repeated resets.

**Expected output:**

- Correct observation and action spaces.
- Finite reset observation.
- Valid step tuple.
- Deterministic reset behavior.
- Rejection of non-finite actions.

**Failure condition:**

- Incorrect space shapes.
- Non-finite observations.
- Missing termination flag on non-finite simulation state.
- Exceptions from valid zero-action steps.

**Determinism requirement:**

- CPU-only MuJoCo simulation.
- Fixed seed where applicable.
- No stochastic environment dynamics.

**CI suitability:**

- Fast, local, no network, no GPU, no YCB assets.

### `tests/unit/test_phase9_env_checker.py`

**Purpose:** Validate Gymnasium compliance.

**Input strategy:**

- Construct the environment from a minimal temporary MJCF model.
- Run the Gymnasium environment checker.

**Expected output:**

- No errors from the environment checker.
- Warnings about unbounded observation spaces may be tolerated if they do not indicate contract violations.

**Failure condition:**

- Environment checker raises an error.
- Reset or step contract violations are detected.

**Determinism requirement:**

- Static temporary MJCF model.
- CPU-only execution.

**CI suitability:**

- Fast and local.

### `tests/integration/test_phase9_sb3_training.py`

**Purpose:** Validate stable-baselines3 integration and checkpoint export.

**Input strategy:**

- Create a minimal temporary MJCF model.
- Run `run_rl_training_pipeline` with tiny dimensions and a small update count.
- Use a temporary checkpoint path.
- Use an explicit seed if the pipeline exposes one.

**Expected output:**

- Training completes without error.
- A checkpoint file is written.
- The checkpoint loads through `load_rl_policy_checkpoint`.
- `build_rl_policy_runner` returns a callable policy.
- The callable policy produces a finite action with the correct shape.

**Failure condition:**

- Training crashes.
- Checkpoint file missing.
- Checkpoint incompatible with existing inference.
- Policy runner returns incorrect action shape or non-finite action.

**Determinism requirement:**

- CPU-only execution.
- Tiny model dimensions.
- Small fixed number of environment steps.

**CI suitability:**

- Must remain short by using tiny dimensions and minimal update counts.

## 12. Test Matrix

| Code unit | Affected phase | Test | Regression coverage | Commit gate |
|---|---:|---|---|---|
| New Gymnasium environment class | Phase 2, Phase 9 | `tests/unit/test_phase9_gymnasium_env.py` | MuJoCo reset, step, observation construction | Unit tests pass |
| Gymnasium compliance | Phase 9 | `tests/unit/test_phase9_env_checker.py` | Standard reset/step contract | Environment checker passes |
| `run_rl_training_pipeline` | Phase 5, Phase 9 | `tests/integration/test_phase9_sb3_training.py` | End-to-end RL training path | Integration test passes |
| Checkpoint export | Phase 5, Phase 9 | `tests/integration/test_phase9_sb3_training.py` | Existing inference compatibility | Checkpoint loads |
| `policy_runner` compatibility | Phase 5, Phase 9 | `tests/integration/test_phase9_sb3_training.py` | Existing inference path remains valid | Runner produces valid action |
| Existing MuJoCo functional API | Phase 2 | Existing Phase 2 tests | Functional simulation remains unchanged | Existing tests pass |
| Existing policy network | Phase 5 | Existing Phase 5 tests | Policy construction remains unchanged | Existing tests pass |

## 13. Numerical and Robotics Validation

### MuJoCo simulation validation

- Reset must restore the initial MuJoCo state.
- Step must use the existing MuJoCo model and data objects.
- The environment must not directly mutate MuJoCo state outside the existing functional wrapper except through the exposed step and control assignment required by the environment contract.

### Observation validation

- Observation must be exactly the concatenation of `qpos` and `qvel`.
- Observation dtype must be `float32`.
- Observation dimension must equal `nq + nv`.

### Action validation

- Action dimension must equal `nu`.
- Action must be finite before being written to MuJoCo control.
- Action padding and truncation must preserve the legacy behavior for size mismatches.
- Action bounds in the Gymnasium action space must be finite.

### Reward validation

- Reward must remain finite.
- Reward must include an action-cost term.
- Reward must include a survival bonus when the next observation is finite.
- Reward must be clipped.

### RL policy output validation

- Exported legacy policy must produce action vectors of shape `(action_dim,)`.
- Exported legacy policy actions must be finite.
- Exported legacy policy behavior must be compatible with the trained stable-baselines3 deterministic policy within a defined tolerance.

### Determinism constraints

- Environment dynamics must be deterministic for identical initial states and identical action sequences on CPU.
- Seed handling must not introduce repository-owned global mutable state.
- Tests must not depend on GPU determinism.

## 14. Data and Interface Contracts

### Environment interface

- Observation space: one-dimensional Box.
- Action space: one-dimensional Box.
- Reset returns observation and info mapping.
- Step returns observation, reward, terminated, truncated, info.

### Training interface

- Existing required pipeline arguments remain unchanged.
- Optional seed argument may be added if absent.
- The pipeline returns no value.
- The pipeline writes a checkpoint file.

### Checkpoint format

- Torch save file.
- Contains `model_state_dict`.
- State dict keys must be compatible with the sequential policy produced by `build_policy_network`.
- Additional scalar metadata fields are allowed.

### Units and coordinate systems

- Observation units are MuJoCo joint position and velocity units.
- Action units are MuJoCo actuator control units.
- No coordinate frame conversion is introduced.

### Persistence

- Checkpoint persistence remains Torch-based.
- No new persistence format is introduced.

## 15. Configuration and Dependency Impact

### Configuration files

No YAML configuration changes are required.

No YAML parsing is introduced.

No global configuration object is introduced.

### Dependency justification

- `gymnasium` is already declared and is required for the standardized environment contract.
- `stable-baselines3` is already declared and is required for the standardized RL algorithm implementation.
- `mujoco` is already declared and remains the simulation backend.
- `torch` is already declared and remains required for policy export.

### Environment constraints

- Local filesystem access is required for robot MJCF files and checkpoint output.
- No network access is required.
- No GPU is required.
- No physical robot hardware is required.
- No YCB dataset assets are required for minimal environment tests.

## 16. Reproducibility and Local Validation

### How to run locally

1. Create or use a minimal local MJCF robot description.
2. Run `scripts/train_rl.py` with tiny dimensions, a small update count, a CPU device, and a temporary checkpoint path.
3. Optionally provide an explicit seed if the pipeline exposes one.
4. Inspect the produced checkpoint file.
5. Load the checkpoint using the existing inference path and run a single policy step against an environment observation.

### Deterministic setup

- Use CPU device.
- Use a fixed seed when available.
- Use a minimal MJCF model with deterministic initial state.
- Use small dimensions and short training runs.

### Synthetic data usage

- No dataset is required.
- Tests must use temporary MJCF files rather than full YCB assets.

### No physical robot dependency

All validation must occur in local MuJoCo simulation.

## 17. Implementation Order

### Step 1: Add the Gymnasium environment class

Modify `src/grasping_ai/simulation/mujoco_env.py` to add the required environment class.

Run existing simulation tests.

### Step 2: Add environment unit tests

Add tests for spaces, reset, step, determinism, non-finite action rejection, and Gymnasium compliance.

Run new environment tests.

### Step 3: Replace the active RL pipeline path

Modify `run_rl_training_pipeline` to use the new environment and stable-baselines3.

Preserve path validation and dimension validation.

### Step 4: Implement checkpoint export

Add the legacy-compatible export step to the RL pipeline.

Verify checkpoint loading through the existing inference path.

### Step 5: Update CLI if needed

Add an optional seed argument to `scripts/train_rl.py` only if such an argument does not already exist.

Preserve existing required arguments.

### Step 6: Remove or isolate obsolete rollout logic

Remove active use of `build_rl_environment` and `collect_rl_rollout`.

If existing tests directly depend on them, migrate their behavioral coverage to the new environment tests before removal.

### Step 7: Add stable-baselines3 integration tests

Add integration tests for training, checkpoint export, and inference compatibility.

### Step 8: Run full regression suite

Run all repository tests before committing.

## 18. Commit Gates

A commit for this phase is allowed only if:

1. All existing tests pass.
2. All new Phase 9 tests pass.
3. The Gymnasium environment passes environment checking without errors.
4. Stable-baselines3 training completes locally on CPU with tiny dimensions.
5. The exported checkpoint loads through the existing inference path.
6. No helper functions are introduced.
7. No utility modules are introduced.
8. No helper classes are introduced.
9. The only new class is the required Gymnasium environment class.
10. No global variables, global constants, or global mutable state are introduced.
11. Every new class and method has a Google-style docstring.
12. Every modified function has an updated Google-style docstring.
13. No YAML configuration loader is introduced.
14. No physical robot dependency is introduced.
15. Static checks required by the repository pass.

## 19. Definition of Done

This phase is complete when all of the following are true:

- A Gymnasium-compatible MuJoCo environment exists in the simulation module.
- The environment exposes valid observation and action spaces.
- The environment reset and step behavior is locally testable and deterministic on CPU.
- The environment rejects non-finite actions.
- The environment returns finite observations after reset and step under normal conditions.
- `run_rl_training_pipeline` trains through stable-baselines3 using the new environment.
- The pipeline no longer depends on the custom rollout generator for end-to-end training.
- The trained policy is exported to a checkpoint compatible with `policy_runner`.
- The existing inference path loads and runs the exported checkpoint without modification.
- All repository tests pass.
- No forbidden constructs are introduced.
- Local validation requires no GPU, network, dataset download, or physical robot.

## 20. Risks and Failure Modes

### Technical risk: stable-baselines3 policy export fragility

**Risk:** The stable-baselines3 policy structure may not map cleanly to the existing sequential policy contract.

**Mitigation:** Define export success by behavioral compatibility: the exported legacy policy must produce finite actions and match the trained deterministic policy within tolerance. Integration tests must fail if the mapping is incorrect.

### Technical risk: Gymnasium action bounds change policy behavior

**Risk:** Finite action bounds may constrain actions differently from the legacy unbounded rollout path.

**Mitigation:** Derive bounds from MuJoCo actuator limits where available. Document that bounded action spaces are required for standardized RL. Preserve step-level padding and truncation for shape compatibility.

### Numerical risk: RL optimization behavior changes

**Risk:** Replacing the custom update step with PPO changes learning dynamics.

**Mitigation:** This change is required by the phase. Regression protection focuses on interface compatibility, checkpoint compatibility, and deterministic environment behavior rather than identical learning curves.

### Integration risk: checkpoint incompatibility with inference

**Risk:** The exported checkpoint may not load through `build_rl_policy_runner`.

**Detection:** Integration test must load the checkpoint and run one inference step.

**Mitigation:** Export using the existing `build_policy_network` module rather than saving the stable-baselines3 model directly.

### Regression risk: existing simulation API breakage

**Risk:** Adding a class to the simulation module may accidentally change existing functional behavior.

**Mitigation:** Existing functional primitives must remain unchanged. Existing Phase 2 tests must pass.

### Reproducibility risk: GPU nondeterminism

**Risk:** Training on GPU may produce nondeterministic results.

**Mitigation:** Validation requires CPU-only execution. Tests must use CPU.

### Architectural risk: legacy RL trainer becomes obsolete

**Risk:** The custom RL trainer remains in the repository while the active pipeline uses stable-baselines3.

**Mitigation:** Phase 9 explicitly marks the custom trainer as legacy for regression coverage. Removal or consolidation belongs to a later cleanup phase unless repository tests are migrated in this phase.

## 21. Out of Scope

The following are explicitly out of scope for this phase:

- YCB object attachment in the RL environment.
- Grasp-specific reward functions.
- Lift-success rewards.
- Contact-based reward shaping.
- Rendering.
- Human visualization.
- Physical robot deployment.
- Hardware communication.
- Real sensor integration.
- Experiment tracking.
- Hyperparameter optimization.
- Configuration file parsing.
- Global configuration systems.
- Changes to generative grasp models.
- Changes to supervised training.
- Changes to evaluation metrics.
- Changes to force closure or collision checking.
- Removal of all legacy RL trainer code unless directly required by test migration.

## 22. Design Review Checklist

- Repository state verified against simulation, RL pipeline, RL trainer, policy model, inference, CLI, and dependency files.
- Phase boundary restricted to standardized Gymnasium environment and stable-baselines3 integration.
- Existing MuJoCo functional API preserved.
- Existing inference checkpoint contract preserved.
- One required Gymnasium environment class introduced.
- No helper functions introduced.
- No utility modules introduced.
- No helper classes introduced.
- No global variables introduced.
- No global constants introduced.
- No global mutable state introduced.
- No YAML configuration loading introduced.
- Tests cover environment contract, Gymnasium compliance, stable-baselines3 training, and checkpoint compatibility.
- Local validation uses temporary MJCF models and CPU-only simulation.
- No physical robot dependency introduced.