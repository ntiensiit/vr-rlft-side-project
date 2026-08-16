# Phase 2 — Simulation & Robotics Core

## 1. Phase Objective

Phase 2 establishes the minimum simulation and robotics core required by later reinforcement learning, evaluation, and inference phases.

The phase must deliver only the following capabilities:

1. A verified MuJoCo-based simulation core that can initialize, reset, step, and expose state or observations.
2. A verified robot control surface sufficient for later grasp execution, including basic gripper command handling.
3. A verified inverse kinematics behavior sufficient for later Cartesian target execution.
4. A stable runtime dependency contract for MuJoCo.
5. Tests that prove the simulation core and robotics core can be used by downstream phases without global state, hidden configuration, or new abstractions.

This phase is intentionally conservative. The repository is a skeleton, and the exact source files inside `src/grasping_ai/simulation/` and `src/grasping_ai/robotics/` were not fully verifiable in the available snapshot. Therefore, this design distinguishes verified facts from unverified repository details. Where an exact file path or symbol cannot be verified, it is marked `Unverified`.

The implementation must not create new classes, helper utilities, global state, or abstraction layers. If the repository does not already contain a skeleton environment class, environment factory, or equivalent simulation entry point, the phase must be paused for a scope decision rather than creating a new object-oriented environment abstraction.

---

## 2. Verified Repository Context

The following facts were verified from the repository snapshot available during analysis.

### Verified packaging facts

- `pyproject.toml` exists.
- The build backend is `hatchling.build`.
- The wheel target uses `packages = ["src"]`.
- Python requirement is `>=3.12`.
- Runtime dependencies already include:
  - `gymnasium`
  - `pytransform3d`
  - `scipy`
  - `stable-baselines3`
  - `torch`
  - `torchvision`
  - `open3d`
  - `theseus`
- The development dependency group already includes:
  - `mujoco>=3.11.0`
  - `pytest`
  - `pytest-cov`
  - `ruff`
  - `mypy`

### Verified project documentation facts

- `README.md` states that the repository is a Python source-code skeleton and that function bodies raise `NotImplementedError`.
- `README.md` describes:
  - `src/grasping_ai/simulation/` as containing MuJoCo environment, scene, and YCB loading.
  - `src/grasping_ai/robotics/` as containing coordinate transforms, kinematics, and gripper control.
- `docs/USAGE.md` describes:
  - `configs/simulation.yaml` as containing physics simulation timestep and simulation step settings.
  - `configs/robot.yaml` as containing robot, gripper, and inverse kinematics settings.
- `docs/PROJECT.md` defines the intended dependency direction:

  `configs -> data -> perception -> models -> inference -> robotics -> simulation -> evaluation`

It also states that training and pipeline modules orchestrate lower-level components rather than being dependencies of them.

### Verified script facts

- `scripts/train_rl.py` exists and imports:

  `python scripts/train_rl.py`

- The CLI signature of `scripts/train_rl.py` includes:
  - `--robot-xml`
  - `--ycb-root`
  - `--object-ids`
  - `--observation-dim`
  - `--action-dim`
  - `--device`

This confirms that later RL orchestration expects a robot description path, YCB root, object identifiers, observation dimensionality, and action dimensionality.

- The repository script table lists `scripts/run_simulation.py` as the script intended to execute generated grasps in MuJoCo on YCB objects.

### Non-verified or partially verified areas

The following are not verified in the available snapshot:

- Exact file names under `src/grasping_ai/simulation/`.
- Exact file names under `src/grasping_ai/robotics/`.
- Exact class names, function names, and method signatures inside those directories.
- Existing test files, fixtures, and test conventions.
- Exact YAML key names inside `configs/simulation.yaml` and `configs/robot.yaml`.
- Whether an existing Gymnasium-compatible environment class already exists.
- Whether an existing IK function or method already exists.
- Whether an existing gripper-control function or method already exists.

Because of this, any source-unit design inside those directories is marked `Unverified` where exact paths or symbols cannot be confirmed.

---

## 3. Scope

Phase 2 includes only the following work.

### In scope

1. Runtime dependency correction for MuJoCo.
   - MuJoCo must be available as a runtime dependency for the simulation core.
   - The version constraint should remain consistent with the verified development dependency.

2. Implementation of the existing simulation skeleton.
   - Environment initialization.
   - Environment reset.
   - Environment step.
   - Observation or state exposure.
   - Basic action validation.
   - Basic episode termination behavior.
   - Optional object loading path for YCB identifiers, without requiring external dataset download.

3. Implementation of the existing robotics skeleton.
   - Basic inverse kinematics behavior.
   - Basic gripper command handling.
   - Validation of invalid kinematics or gripper inputs.

4. Preservation of static configuration contracts.
   - Do not remove `configs/simulation.yaml`.
   - Do not remove `configs/robot.yaml`.
   - Do not introduce global configuration loading.
   - Do not introduce module-level configuration state.

5. Tests for the Phase 2 contract.
   - Simulation dependency availability.
   - Environment initialization.
   - Environment reset and step behavior.
   - Invalid input handling.
   - Basic IK behavior.
   - Basic gripper command validation.
   - Regression of Phase 1 package import behavior.

### Conditional scope

The simulation and robotics implementation is conditional on locating existing skeleton units.

If no existing environment class, environment factory, or equivalent simulation function exists, Phase 2 must not create a new environment class. The correct action is to stop and request a scope decision because the no-new-class constraint would otherwise be violated.

If no existing IK skeleton exists, Phase 2 must not create a new IK utility module. The correct action is to stop and request a scope decision.

If no existing gripper-control skeleton exists, Phase 2 must not create a new gripper utility module. The correct action is to stop and request a scope decision.

---

## 4. Out of Scope

The following are explicitly out of scope for Phase 2.

1. Reinforcement learning training.
   - Stable-Baselines3 policy creation.
   - Policy optimization.
   - Reward shaping beyond the minimum numeric reward required by the environment interface.
   - Learning-rate schedules.
   - Checkpointing.

2. Supervised model training.
   - Diffusion model training.
   - Flow model training.
   - Equivariant encoder training.

3. Grasp generation.
   - Sampling grasp poses.
   - Ranking grasp poses.
   - Loading generative model checkpoints.
   - Inference of grasp poses from point clouds.

4. Full evaluation pipeline.
   - Force-closure analysis.
   - Lift-success reporting.
   - Collision reporting.
   - Aggregate report generation.

5. Dataset handling.
   - YCB download.
   - Dataset indexing.
   - Point-cloud preprocessing.
   - Data augmentation.

6. Real-robot integration.
   - Hardware drivers.
   - Robot communication.
   - Safety controllers.
   - Calibration pipelines.
   - Deployment workflows.

7. Advanced simulation features.
   - Domain randomization.
   - Parallel environments.
   - Sensor noise simulation.
   - Actuator delay simulation.
   - Camera rendering pipelines.
   - Headless rendering optimization.
   - MuJoCo viewer integration unless already required by an existing skeleton function.

8. New architecture.
   - New environment base classes.
   - New robot abstraction classes.
   - New utility modules.
   - New helper functions.
   - New global configuration managers.
   - New plugin systems.

---

## 5. Existing Architecture and Patterns

Phase 2 must preserve the following verified or documented repository patterns.

### Pattern 1: Thin CLI scripts delegate to pipeline functions

`scripts/train_rl.py` demonstrates the existing pattern:

- The script parses CLI arguments.
- The script calls a pipeline function.
- The script does not implement core robotics logic.

Phase 2 must not move simulation or robotics logic into scripts. Simulation and robotics behavior must remain in the appropriate lower-level modules under `src/grasping_ai/simulation/` and `src/grasping_ai/robotics/`.

### Pattern 2: Skeleton functions raise `NotImplementedError`

The README states that the repository contains skeleton code with `NotImplementedError` bodies. Phase 2 should replace `NotImplementedError` only in existing functions, methods, or configuration entries that belong to the phase scope.

Phase 2 must not create new reusable helper functions to avoid implementing required behavior directly in the appropriate existing skeleton unit.

### Pattern 3: Lower-level modules do not depend on orchestration

The documented dependency direction places simulation and robotics below pipelines and training.

Phase 2 modules must not import pipeline functions, training loops, inference loaders, or CLI scripts.

### Pattern 4: Configuration remains static

The repository uses YAML configuration files, but no global configuration loader was verified.

Phase 2 must not introduce a global configuration object, module-level config cache, or singleton configuration manager. If simulation timestep, control step, robot XML path, YCB root, or object IDs are needed, they should be accepted explicitly by the existing skeleton interface or by the pipeline that later calls it.

### Pattern 5: Existing dependencies define the technical direction

The repository already includes:

- `gymnasium` for environment interfaces.
- `mujoco` in the development dependency group.
- `pytransform3d` and `theseus` for spatial math.
- `scipy` for numerical computation.

Phase 2 must use these existing dependencies where appropriate. It must not introduce new simulation, physics, IK, or robotics dependencies.

---

## 6. Implementation Dependencies

Phase 2 requires the following dependencies.

### Existing dependencies that must remain available

| Dependency | Reason |
| --- | --- |
| `gymnasium` | Later RL training and environment interaction expect a Gymnasium-compatible contract. |
| `pytransform3d` | SE(3) transforms, rotations, and kinematics math. |
| `scipy` | Numerical computation where existing skeleton code already uses it. |
| `theseus` | Existing spatial optimization dependency; use only if already required by the skeleton. |
| `pytest` | Development dependency for Phase 2 tests. |

### Dependency configuration change required

| Dependency | Current verified location | Required Phase 2 treatment |
| --- | --- | --- |
| `mujoco>=3.11.0` | Development dependency group. | Must be declared as a runtime dependency because Phase 2 implements runtime simulation behavior. |

No new third-party dependency may be introduced.

---

## 7. Source-Unit Change Matrix

Rows marked `Unverified` are required phase targets, but the exact file path or symbol must be resolved from the repository before implementation. They are not speculative new files.

| File | Symbol | Change Type | Current Responsibility | Proposed Change | Risk | Required Tests |
| --- | --- | --- | --- | --- | --- | --- |
| `pyproject.toml` | `project.dependencies` | Extend | Declares runtime dependencies. | Add the already-verified MuJoCo dependency to runtime dependencies. Keep version constraint consistent with the development dependency. | Medium | Dependency import test and Phase 1 dependency regression test. |
| `configs/simulation.yaml` | Documented simulation timestep and step settings | Configuration preservation | Declares simulation timing semantics according to documentation. | Preserve file and documented semantics. Do not introduce global loading. If exact keys are verified, keep them unchanged. | Low | Configuration existence test; key-contract test only after exact keys are verified. |
| `configs/robot.yaml` | Documented robot, gripper, and IK settings | Configuration preservation | Declares robot and IK semantics according to documentation. | Preserve file and documented semantics. Do not introduce global loading. If exact keys are verified, keep them unchanged. | Low | Configuration existence test; key-contract test only after exact keys are verified. |
| Unverified existing module under `src/grasping_ai/simulation/` | Unverified existing environment initialization function, method, or factory | Modify | Currently expected to raise `NotImplementedError`. | Implement initialization of the MuJoCo simulation state from explicit inputs. Do not create a new class. | High | Environment initialization tests. |
| Unverified existing module under `src/grasping_ai/simulation/` | Unverified existing reset function or method | Modify | Currently expected to raise `NotImplementedError`. | Implement deterministic reset behavior and return initial observation or state. | High | Reset tests. |
| Unverified existing module under `src/grasping_ai/simulation/` | Unverified existing step function or method | Modify | Currently expected to raise `NotImplementedError`. | Implement action validation, actuation, physics stepping, and return of observation, reward, termination, truncation, and info if the existing interface follows Gymnasium. | High | Step tests and invalid-action tests. |
| Unverified existing module under `src/grasping_ai/simulation/` | Unverified existing observation or state function or method | Modify | Currently expected to raise `NotImplementedError`. | Implement observation or state assembly from MuJoCo simulation state. | High | Observation shape and determinism tests. |
| Unverified existing module under `src/grasping_ai/robotics/` | Unverified existing inverse kinematics function or method | Modify | Currently expected to raise `NotImplementedError`. | Implement basic IK behavior using existing spatial math dependencies and the existing skeleton signature. | High | IK valid-target and invalid-target tests. |
| Unverified existing module under `src/grasping_ai/robotics/` | Unverified existing gripper-control function or method | Modify | Currently expected to raise `NotImplementedError`. | Implement basic gripper command validation and mapping to simulation-compatible actuator or joint commands. | High | Gripper valid-command and invalid-command tests. |
| Existing or minimum required test file under `tests/unit/` | Phase 2 simulation and robotics tests | Test-only change | No verified existing Phase 2 tests. | Add tests for dependency availability, environment initialization, reset, step, invalid input, IK, and gripper behavior. | Medium | All Phase 2 tests pass. |

---

## 8. Detailed Source-Unit Design

### `pyproject.toml::project.dependencies`

#### Current behavior

The verified `pyproject.toml` declares runtime dependencies for the project. MuJoCo is currently declared only in the development dependency group.

#### Required behavior

Phase 2 implements runtime simulation behavior. Therefore, MuJoCo must be declared as a runtime dependency. The version constraint should remain consistent with the verified development dependency.

#### Design

- Input: none.
- Output: none.
- Control flow: none.
- State changes: none.
- Existing dependencies: `hatchling`, Python packaging metadata.
- Error behavior: packaging errors should surface during dependency resolution.
- Edge cases: duplicate MuJoCo entries must be avoided.
- Interaction with surrounding code: simulation modules and tests will import MuJoCo at runtime.
- Existing pattern being followed: preserve declarative dependency management in `pyproject.toml`.

The change should be limited to dependency declaration. It must not introduce version upgrades unrelated to Phase 2.

---

### `configs/simulation.yaml`

#### Current behavior

Documentation states that this file contains physics simulation timestep and simulation step settings. The exact key names were not verified.

#### Required behavior

Phase 2 must preserve the file and its documented semantics. Phase 2 must not introduce global configuration loading. If the simulation core needs timestep or step count values, those values should be accepted explicitly through the existing skeleton interface or through a later pipeline.

#### Design

- Input: none.
- Output: none.
- Control flow: none.
- State changes: none.
- Existing dependencies: none.
- Error behavior: not applicable during Phase 2 because no runtime YAML parsing is introduced.
- Edge cases: missing keys should be handled by later pipeline validation if that pipeline chooses to consume the file.
- Interaction with surrounding code: later orchestration may read this file, but Phase 2 does not require it.
- Existing pattern being followed: static configuration without global state.

No YAML key may be renamed, removed, or added unless exact key verification proves that a documented key is missing.

---

### `configs/robot.yaml`

#### Current behavior

Documentation states that this file contains robot, gripper, and inverse kinematics settings. The exact key names were not verified.

#### Required behavior

Phase 2 must preserve the file and its documented semantics. Phase 2 must not introduce global robot configuration state. Robot XML paths, gripper parameters, and IK parameters should be explicit arguments where they are needed.

#### Design

- Input: none.
- Output: none.
- Control flow: none.
- State changes: none.
- Existing dependencies: none.
- Error behavior: not applicable during Phase 2 because no runtime YAML parsing is introduced.
- Edge cases: missing robot description files should raise explicit filesystem or simulation errors in the simulation initialization unit.
- Interaction with surrounding code: later orchestration may read this file, but Phase 2 does not require it.
- Existing pattern being followed: static configuration without global state.

---

### Unverified existing simulation initialization unit

#### Current behavior

The repository documentation states that `src/grasping_ai/simulation/` contains MuJoCo environment, scene, and YCB loading behavior. The exact symbol was not verified. Because the repository is a skeleton, the current behavior is expected to be `NotImplementedError`.

#### Required behavior

The existing simulation initialization unit must create or configure a MuJoCo-backed simulation state sufficient for reset, step, observation, and later robotics control.

It must support, at minimum:

1. Loading a robot description from an explicit path.
2. Accepting explicit simulation timing inputs if the existing skeleton signature includes them.
3. Optionally loading object assets by identifier if the existing skeleton signature includes object identifiers.
4. Failing explicitly when required inputs are missing or invalid.

#### Design

##### Input

Expected inputs may include:

- Robot description path.
- Optional YCB root path.
- Optional object identifier or identifiers.
- Optional physics timestep.
- Optional control step count.
- Optional rendering flag if already present in the skeleton.

The implementation must follow the existing skeleton signature. It must not widen the signature unnecessarily.

##### Output

The output must match the existing skeleton contract. Possible outputs include:

- A simulation environment object.
- A simulation state object.
- A Gymnasium-compatible environment.
- A tuple containing simulation handles and initial state.

The implementation must not introduce a new wrapper type or class.

##### Control flow

The initialization logic should:

1. Validate required paths.
2. Load the robot description through MuJoCo.
3. Attach or register object assets only if object inputs are provided.
4. Prepare internal simulation state for reset.
5. Return the initialized simulation unit.

##### State changes

Initialization may create local simulation state. It must not create module-level state.

##### Existing dependencies

- `mujoco`
- `gymnasium`, if the skeleton already exposes a Gymnasium-compatible object.
- Standard filesystem utilities where already used by the skeleton.

##### Error behavior

Expected errors:

- Missing robot description path raises a filesystem error.
- Invalid robot description content raises a simulation or value error.
- Missing YCB root with object IDs provided raises a filesystem or value error.
- Invalid object identifier raises a value error.

##### Edge cases

- Robot description exists but contains no controllable joints.
- Object IDs are provided but YCB root is empty.
- Object IDs are omitted.
- Rendering is unavailable in a headless test environment.

##### Interaction with surrounding code

Later pipeline functions, including RL training and simulation execution, will call this initialization behavior. It must remain stable and explicit.

##### Existing pattern being followed

Low-level simulation module does not depend on pipelines, training, or inference.

---

### Unverified existing simulation reset unit

#### Current behavior

The exact reset symbol was not verified. The current behavior is expected to be `NotImplementedError`.

#### Required behavior

The reset unit must restore the simulation to a deterministic initial state and return the initial observation or state required by downstream callers.

#### Design

##### Input

The existing skeleton may accept no inputs, or it may accept explicit seed, object identifier, or episode options. The implementation must preserve the existing signature.

##### Output

The output must match the existing skeleton contract. For a Gymnasium-compatible environment, this is typically:

- observation
- info mapping

If the skeleton uses a different contract, the existing contract must be preserved.

##### Control flow

1. Reset simulation state.
2. Reset robot joint state.
3. Reset object pose if object loading is supported.
4. Assemble initial observation or state.
5. Return observation and metadata.

##### State changes

Internal simulation state may be modified. Module-level state must not be modified.

##### Existing dependencies

- MuJoCo simulation state.
- Existing observation assembly behavior.

##### Error behavior

Reset should raise an error only if the simulation state is invalid or uninitialized.

##### Edge cases

- Reset before initialization.
- Reset after termination.
- Reset after truncation.
- Reset with missing optional object assets.

##### Interaction with surrounding code

RL rollouts and simulation evaluation will call reset at episode boundaries.

##### Existing pattern being followed

Environment behavior remains local to the environment instance or simulation state.

---

### Unverified existing simulation step unit

#### Current behavior

The exact step symbol was not verified. The current behavior is expected to be `NotImplementedError`.

#### Required behavior

The step unit must accept an action, validate it, apply it to the simulated robot or gripper, advance physics, and return the information required by downstream RL or evaluation code.

#### Design

##### Input

Expected input is an action compatible with the existing action contract.

The action may represent:

- Joint position targets.
- Joint velocity targets.
- Cartesian target deltas.
- Gripper commands.
- A combined manipulation action.

The implementation must follow the existing skeleton contract and must not invent a new action representation.

##### Output

If the existing skeleton follows Gymnasium, the output must be:

- observation
- reward
- terminated
- truncated
- info

If the skeleton uses a different contract, preserve that contract.

##### Control flow

1. Validate action type, shape, and bounds.
2. Map the action to simulation-compatible control commands.
3. Advance physics for the configured number of internal steps or timestep.
4. Read resulting simulation state.
5. Assemble observation.
6. Compute minimum numeric reward if required by the existing interface.
7. Determine termination and truncation.
8. Return the step result.

##### State changes

Internal simulation state changes are expected. Module-level state must not change.

##### Existing dependencies

- MuJoCo.
- Existing robotics control behavior if the step unit delegates gripper or joint commands.

##### Error behavior

- Invalid action shape raises a value or type error.
- Invalid action values raise a value error.
- Step before initialization raises an error.

##### Edge cases

- Zero action.
- Maximum and minimum valid action values.
- Action containing non-finite values.
- Step after termination.
- Step after truncation.

##### Interaction with surrounding code

RL training will call step repeatedly. Evaluation pipelines will call step to execute grasps.

##### Existing pattern being followed

Environment stepping remains deterministic, local, and free of global side effects.

---

### Unverified existing observation or state unit

#### Current behavior

The exact observation symbol was not verified. The current behavior is expected to be `NotImplementedError`.

#### Required behavior

The observation unit must assemble a stable representation of simulation state for downstream policy, evaluation, or debugging use.

#### Design

##### Input

The input is the current MuJoCo simulation state or environment state.

##### Output

The output must be a numeric observation or state structure compatible with the existing contract.

If the environment exposes a Gymnasium observation space, the output shape and dtype must remain consistent with that space.

##### Control flow

1. Read robot joint state.
2. Read gripper state if available.
3. Read object state if object loading is active.
4. Assemble the observation in a fixed order.
5. Return the observation.

##### State changes

None.

##### Existing dependencies

- MuJoCo simulation state.
- Existing SE(3) math behavior from Phase 1 if spatial transforms are required.

##### Error behavior

The observation unit should raise an error only when simulation state is unavailable or corrupted.

##### Edge cases

- Object state unavailable.
- Gripper state unavailable.
- Simulation state uninitialized.

##### Interaction with surrounding code

RL policies and evaluation code depend on stable observation ordering.

##### Existing pattern being followed

Observation assembly remains a low-level simulation behavior.

---

### Unverified existing inverse kinematics unit

#### Current behavior

The exact IK symbol was not verified. The current behavior is expected to be `NotImplementedError`.

#### Required behavior

The IK unit must convert a Cartesian target into a robot joint command compatible with the simulated robot.

#### Design

##### Input

Expected inputs may include:

- Target end-effector pose.
- Current joint state.
- Optional joint limits.
- Optional solver tolerance.
- Optional maximum iterations.

The implementation must follow the existing skeleton signature.

##### Output

The output must match the existing skeleton contract. Expected output is a joint position target, joint velocity target, or solver result.

##### Control flow

1. Validate target pose.
2. Validate current joint state.
3. Use the existing spatial math dependency selected by the skeleton.
4. Solve for joint commands.
5. Return joint commands or failure indicator.

##### State changes

None.

##### Existing dependencies

- `pytransform3d`
- `theseus` if already used by the skeleton.
- `scipy` if already used by the skeleton.

##### Error behavior

- Invalid target pose raises a value or type error.
- Invalid joint state raises a value or type error.
- Solver failure returns the failure behavior defined by the existing skeleton contract.

##### Edge cases

- Identity target pose.
- Target pose far from reachable workspace.
- Singular joint configurations.
- Joint limits reached.

##### Interaction with surrounding code

Later Cartesian control or grasp execution behavior will call IK.

##### Existing pattern being followed

IK remains a robotics module behavior and does not depend on pipelines or training.

---

### Unverified existing gripper-control unit

#### Current behavior

The exact gripper-control symbol was not verified. The current behavior is expected to be `NotImplementedError`.

#### Required behavior

The gripper-control unit must accept a gripper command and produce a simulation-compatible gripper command or state change.

#### Design

##### Input

Expected input may include:

- Normalized gripper command.
- Raw gripper position command.
- Gripper force or velocity command.

The implementation must follow the existing skeleton signature.

##### Output

The output must match the existing skeleton contract. Expected output is a gripper actuator command, joint target, or state update.

##### Control flow

1. Validate command type and range.
2. Map the command to the gripper actuator or joint representation.
3. Return or apply the command according to the skeleton contract.

##### State changes

If the skeleton applies the command directly to simulation state, internal simulation state may change. Module-level state must not change.

##### Existing dependencies

- MuJoCo simulation state if the gripper command is applied directly.
- Existing robotics configuration only if explicitly passed.

##### Error behavior

- Out-of-range command raises a value error.
- Non-finite command raises a value error.
- Command applied before initialization raises an error.

##### Edge cases

- Fully open command.
- Fully closed command.
- Command at boundary values.
- Repeated commands.

##### Interaction with surrounding code

Grasp execution and RL action post-processing will depend on stable gripper behavior.

##### Existing pattern being followed

Gripper control remains a low-level robotics behavior.

---

## 9. Data Flow

Phase 2 introduces the following data flows.

### Simulation initialization data flow

`Explicit robot and object inputs -> Existing simulation initialization unit -> MuJoCo model and simulation state -> Ready simulation state`

Input source:

- Pipeline or test code provides explicit paths and identifiers.

Transformation:

- File paths are loaded into MuJoCo.
- Optional object assets are attached.
- Initial simulation state is prepared.

Validation:

- Robot description path existence.
- Object root existence when object IDs are provided.
- Validity of simulation timing inputs if present.

Output:

- Initialized simulation environment or simulation state.

Consumers:

- Reset.
- Step.
- IK.
- Gripper control.
- Later RL and evaluation pipelines.

### Simulation step data flow

`Action -> Existing step unit -> Validation and actuation -> MuJoCo physics step -> Observation, reward, termination flags, info`

Input source:

- RL policy, evaluation pipeline, or direct test code.

Transformation:

- Action is validated and mapped to robot or gripper commands.
- Physics is advanced.

Validation:

- Action shape.
- Action bounds.
- Finite values.

Output:

- Observation.
- Reward.
- Termination flag.
- Truncation flag.
- Info metadata.

Consumers:

- RL training.
- Grasp execution.
- Evaluation.

### Inverse kinematics data flow

`Target SE(3) pose and current joint state -> Existing IK unit -> Joint command`

Input source:

- Later grasp execution or Cartesian control code.

Transformation:

- Target pose is validated.
- Solver computes joint commands.

Validation:

- Pose shape.
- Rotation validity.
- Joint state shape.

Output:

- Joint command or solver result.

Consumers:

- Simulation step or robot control behavior.

### Gripper-control data flow

`Gripper command -> Existing gripper-control unit -> Simulation-compatible gripper actuation`

Input source:

- Step unit, grasp execution pipeline, or direct test code.

Transformation:

- Command range validation.
- Mapping to gripper actuator or joint target.

Output:

- Gripper command or updated simulation state.

Consumers:

- MuJoCo simulation.
- Evaluation logic.

---

## 10. Execution Flow

Phase 2 does not create a new CLI entry point.

The verified execution context is:

1. A script such as `scripts/train_rl.py` calls a pipeline function.
2. The pipeline function later uses the simulation core.
3. The simulation core uses robotics behavior for control.

Phase 2 affects step 2 and step 3, but it does not implement full pipeline orchestration.

### Verified downstream entry path

`scripts/train_rl.py`

Phase 2 does not implement `run_rl_training_pipeline`. It provides the simulation and robotics behavior that such a pipeline will need.

### Direct validation path

For Phase 2 validation, tests may directly call the verified or located simulation and robotics skeleton units. This avoids depending on unimplemented pipeline behavior.

### Error propagation

Errors must propagate naturally:

- Missing files propagate filesystem errors.
- Invalid actions propagate value or type errors.
- Invalid poses propagate value or type errors.
- Uninitialized simulation state propagates runtime errors.

Phase 2 must not swallow errors silently.

---

## 11. Configuration Changes

Phase 2 requires one packaging configuration change and no YAML configuration additions.

### Packaging configuration change

| File | Key | Existing value | Required change | Reason | Consumed by | Tests |
| --- | --- | --- | --- | --- | --- | --- |
| `pyproject.toml` | `project.dependencies` | Does not currently list MuJoCo as a runtime dependency. | Add MuJoCo with the same minimum version already present in the development dependency group. | Phase 2 imports MuJoCo at runtime. | Simulation module and tests. | Dependency import test. |

### YAML configuration changes

No YAML configuration additions are required.

The following files must remain available:

- `configs/simulation.yaml`
- `configs/robot.yaml`

Phase 2 must not:

- Add new YAML keys.
- Rename existing YAML keys.
- Introduce a YAML parser.
- Introduce global configuration state.
- Read configuration at module import time.

If later pipeline code needs values from these files, that code must receive explicit values or implement its own explicit configuration handling in a later phase.

---

## 12. Test Strategy

Testing must use `pytest`, which is already a development dependency.

No existing test suite was verified. Therefore, the implementation engineer must first check whether a test directory and existing conventions are present.

If tests already exist, Phase 2 tests must follow the existing conventions for:

- File naming.
- Test function naming.
- Fixture usage.
- Assertion style.
- Parameterization.
- Mocking, if already used.

If no tests exist, Phase 2 may add the minimum required test file under a conventional unit-test location. This is a test-only addition.

### Testing principles

1. Tests must be deterministic.
2. Tests must not require network access.
3. Tests must not require downloaded YCB assets.
4. Tests must not require GPUs.
5. Tests must not require model checkpoints.
6. Tests must not require real robot hardware.
7. Tests must not introduce global state.
8. Tests must not create helper functions or helper classes.
9. Tests must not depend on visualization.
10. Tests must avoid heavy physics simulation where possible.

### Asset strategy

Because YCB assets are not guaranteed, tests must use the smallest possible local MuJoCo description.

The preferred approach is:

- Use a minimal robot description created locally inside the test scope.
- Use no external dataset.
- Avoid committing binary assets.
- Avoid creating shared helper fixture code unless an existing fixture already provides the same behavior.

If the existing skeleton requires a specific robot description and cannot operate with a minimal test description, then Phase 2 validation is blocked until a verified test asset is available.

---

## 13. Test Suite

The following tests are required. Rows that depend on unverified skeleton symbols are marked accordingly.

| Test | File | Target | Scenario | Expected Result | Regression Risk |
| --- | --- | --- | --- | --- | --- |
| `test_mujoco_runtime_dependency_available` | Existing or minimum required Phase 2 unit test file | `pyproject.toml` runtime dependency | Import MuJoCo. | Import succeeds. | Medium if dependency configuration is incorrect. |
| `test_phase1_package_import_remains_stable` | Existing or minimum required Phase 2 unit test file | `src/grasping_ai/__init__.py` | Import `grasping_ai`. | Import succeeds. | High if Phase 2 changes package initialization. |
| `test_simulation_config_file_exists` | Existing or minimum required Phase 2 unit test file | `configs/simulation.yaml` | Check file existence. | File exists. | Low. |
| `test_robot_config_file_exists` | Existing or minimum required Phase 2 unit test file | `configs/robot.yaml` | Check file existence. | File exists. | Low. |
| `test_simulation_initializes_with_minimal_robot_description` | Existing or minimum required Phase 2 unit test file | Unverified simulation initialization unit | Initialize simulation with a minimal local robot description. | Initialization succeeds without global side effects. | High. |
| `test_simulation_initialization_rejects_missing_robot_description` | Existing or minimum required Phase 2 unit test file | Unverified simulation initialization unit | Provide a missing robot description path. | Raises filesystem or value error. | Medium. |
| `test_simulation_reset_returns_initial_observation` | Existing or minimum required Phase 2 unit test file | Unverified reset unit | Reset initialized simulation. | Returns initial observation and metadata according to existing contract. | High. |
| `test_simulation_step_accepts_valid_action` | Existing or minimum required Phase 2 unit test file | Unverified step unit | Step with a valid action. | Returns observation, reward, termination flags, and info according to existing contract. | High. |
| `test_simulation_step_rejects_invalid_action_shape` | Existing or minimum required Phase 2 unit test file | Unverified step unit | Step with an action of invalid shape. | Raises value or type error. | Medium. |
| `test_simulation_step_rejects_non_finite_action` | Existing or minimum required Phase 2 unit test file | Unverified step unit | Step with NaN or infinite action values. | Raises value error. | Medium. |
| `test_simulation_observation_shape_is_stable` | Existing or minimum required Phase 2 unit test file | Unverified observation unit | Reset and step, then inspect observation. | Observation shape and dtype remain consistent with environment contract. | High for downstream RL. |
| `test_ik_returns_joint_command_for_valid_target` | Existing or minimum required Phase 2 unit test file | Unverified IK unit | Provide a valid target pose and current joint state. | Returns joint command or solver success result. | High. |
| `test_ik_rejects_invalid_target_pose` | Existing or minimum required Phase 2 unit test file | Unverified IK unit | Provide invalid pose shape or non-finite pose values. | Raises value or type error. | Medium. |
| `test_gripper_command_accepts_valid_range` | Existing or minimum required Phase 2 unit test file | Unverified gripper-control unit | Provide valid open and close commands. | Returns or applies valid gripper command. | High. |
| `test_gripper_command_rejects_out_of_range_command` | Existing or minimum required Phase 2 unit test file | Unverified gripper-control unit | Provide command outside valid range. | Raises value error. | Medium. |
| `test_simulation_state_does_not_leak_between_instances` | Existing or minimum required Phase 2 unit test file | Unverified simulation initialization and reset units | Initialize, step, then initialize again or reset. | New state does not depend on previous state beyond explicit inputs. | Medium. |

---

## 14. Regression Test Plan

Phase 2 can introduce regressions in packaging, imports, configuration, and downstream pipeline expectations.

### Unit-level regressions

| Existing behavior | Why it could break | Protecting test | Expected result |
| --- | --- | --- | --- |
| Package import succeeds. | Simulation or robotics modules may introduce import cycles or heavy imports. | `test_phase1_package_import_remains_stable` | Import succeeds. |
| Phase 1 math dependencies remain usable. | Dependency change may affect resolution. | Phase 1 dependency tests and Phase 2 dependency test. | Imports succeed. |
| SE(3) math remains pure. | Robotics code may accidentally depend on global state. | Phase 1 SE(3) tests, if present. | Existing Phase 1 behavior remains intact. |

### Integration-level regressions

| Existing behavior | Why it could break | Protecting test | Expected result |
| --- | --- | --- | --- |
| Scripts can import pipeline modules. | Simulation modules may be imported indirectly and cause errors. | Existing import-related tests, if present. | Import path remains valid. |
| Downstream RL pipeline can expect a Gymnasium-compatible environment. | Step or reset contract may be inconsistent. | Environment reset and step tests. | Contract remains stable. |

### Pipeline-level regressions

No full pipeline behavior is verified yet. Therefore, no pipeline-level regression test can be specified without invention.

If an existing pipeline smoke test exists, it must be rerun. If no such test exists, Phase 2 must not create a broad pipeline smoke test.

---

## 15. Impact on Preceding Phases

### Phase 1 — Foundation & Math Primitives

| Item | Description |
| --- | --- |
| Existing contract | Phase 1 established package importability, dependency availability, static configuration preservation, and pure SE(3) math behavior where verified. |
| What Phase 2 changes | Phase 2 adds MuJoCo as a runtime dependency and introduces simulation and robotics behavior. |
| What remains compatible | Package layout, existing dependencies, static configuration files, and Phase 1 math behavior must remain unchanged. |
| Previous-phase tests to rerun | Phase 1 package import test, dependency tests, SE(3) math tests if present, and configuration existence tests if present. |
| Whether Phase 1 needs modification | No modification is expected unless dependency resolution fails. |

No other preceding phase exists.

---

## 16. Impact on Downstream Phases

### Downstream Phase 3 — Data Pipeline & Perception

| Item | Description |
| --- | --- |
| Dependency | Phase 3 does not strictly depend on Phase 2, but future synthetic data generation may use simulation. |
| Interface relied on | If simulation is later used for data generation, initialization and reset must remain explicit and deterministic. |
| New behavior available after Phase 2 | A simulation core exists. |
| Constraints preserved | Simulation must not depend on data pipeline modules. |
| Protecting tests | Environment initialization and reset tests. |

### Downstream Phase 4 — Generative Grasp Model

| Item | Description |
| --- | --- |
| Dependency | Phase 4 does not strictly depend on Phase 2 for model training, but grasp execution validation will later require simulation. |
| Interface relied on | Stable SE(3) conventions and later simulation execution. |
| New behavior available after Phase 2 | Grasp poses can later be tested in simulation. |
| Constraints preserved | Simulation must not depend on generative model code. |
| Protecting tests | SE(3) and environment observation tests. |

### Downstream Phase 5 — Reinforcement Learning Policy

| Item | Description |
| --- | --- |
| Dependency | Phase 5 depends heavily on Phase 2. |
| Interface relied on | Reset, step, observation, action, reward, termination, and truncation behavior. |
| New behavior available after Phase 2 | A runnable simulation environment for RL rollouts. |
| Constraints preserved | Observation and action contracts must remain stable. No global state. |
| Protecting tests | Reset, step, observation shape, invalid action, and no-state-leak tests. |

### Downstream Phase 6 — End-to-End Orchestration & Evaluation

| Item | Description |
| --- | --- |
| Dependency | Phase 6 depends on Phase 2 for executing grasps and observing outcomes. |
| Interface relied on | Simulation initialization, stepping, gripper control, and state exposure. |
| New behavior available after Phase 2 | Generated grasps can later be executed and evaluated. |
| Constraints preserved | Simulation must remain deterministic and explicit. |
| Protecting tests | Simulation initialization, step, gripper, and IK tests. |

---

## 17. Cross-Phase Contract

Phase 2 establishes the following contract.

### Inputs

1. Simulation initialization:
   - Explicit robot description path.
   - Optional object asset root.
   - Optional object identifiers.
   - Optional timing parameters if present in the existing skeleton.

2. Step:
   - Action compatible with the existing action contract.

3. IK:
   - Target pose.
   - Current joint state.
   - Optional solver constraints if present in the existing skeleton.

4. Gripper control:
   - Gripper command compatible with the existing command contract.

### Outputs

1. Simulation initialization:
   - Initialized simulation environment or state.

2. Reset:
   - Initial observation or state.
   - Metadata if required by the existing contract.

3. Step:
   - Observation.
   - Numeric reward.
   - Termination flag.
   - Truncation flag.
   - Info metadata if the contract follows Gymnasium.

4. IK:
   - Joint command or solver result.

5. Gripper control:
   - Gripper actuation command or state update.

### Expected behavior

1. Simulation behavior is local to the initialized simulation object or state.
2. Reset is deterministic for a given seed or initialization input.
3. Step validates actions before advancing physics.
4. IK validates poses before solving.
5. Gripper commands are validated before actuation.

### Invariants

1. No module-level mutable state.
2. No global configuration state.
3. No dependency on pipelines, training, or inference.
4. Observation ordering remains stable.
5. Action contract remains stable.
6. Existing configuration files remain present.

### Error behavior

1. Missing required files raise explicit filesystem or value errors.
2. Invalid actions raise value or type errors.
3. Invalid poses raise value or type errors.
4. Invalid gripper commands raise value errors.
5. Uninitialized simulation state raises an error.

### Configuration assumptions

1. YAML files remain static.
2. Phase 2 does not parse YAML globally.
3. Later pipelines may consume configuration explicitly.

### Data assumptions

1. No external dataset download is required.
2. Tests must not depend on YCB assets.
3. Simulation tests should use minimal local descriptions where possible.

### Performance assumptions

1. Single-environment simulation is sufficient for Phase 2.
2. Parallel environments are out of scope.
3. Rendering is not required for tests.
4. Heavy physics loops should be avoided in tests.

---

## 18. Validation Before Commit

The implementation must not be considered ready for commit until the following sequence is complete.

1. Run Phase 2 dependency tests.
2. Run Phase 2 package import regression tests.
3. Run Phase 2 configuration existence tests.
4. Run simulation initialization tests.
5. Run simulation reset tests.
6. Run simulation step tests.
7. Run invalid-action tests.
8. Run observation stability tests.
9. Run IK valid-target tests.
10. Run IK invalid-target tests.
11. Run gripper valid-command tests.
12. Run gripper invalid-command tests.
13. Run no-state-leak tests.
14. Rerun Phase 1 tests.
15. Run the broader relevant test suite.
16. Verify that no unrelated module changed.
17. Verify that no new class was introduced.
18. Verify that no helper or utility function was introduced.
19. Verify that no global variable or constant was introduced.
20. Verify that no file-level source description was introduced.
21. Verify that every modified or newly introduced function or method has a Google-style docstring.
22. Verify that no unrelated refactoring is included.
23. Verify that no new dependency other than the MuJoCo runtime declaration was introduced.
24. Verify that downstream contracts remain compatible.

The final commit gate is:

`Implementation -> Targeted Tests -> Regression Tests -> Integration Tests -> Full Relevant Suite -> Static/Structural Review -> Commit`

---

## 19. Commit Boundary

The Phase 2 commit must contain only the following.

### Allowed in the commit

1. Runtime dependency declaration for MuJoCo.
2. Modifications to verified existing simulation skeleton units.
3. Modifications to verified existing robotics skeleton units.
4. Required Phase 2 tests.
5. Minimal test-only fixtures if required by the existing testing convention and no existing fixture is available.

### Not allowed in the commit

1. New environment base classes.
2. New robot abstraction classes.
3. New utility modules.
4. New helper functions.
5. New global constants.
6. New global variables.
7. New configuration loaders.
8. Unrelated formatting changes.
9. Refactoring of scripts unrelated to Phase 2.
10. Training code.
11. Model code.
12. Evaluation metric code.
13. Dataset ingestion code.
14. Real-robot integration code.
15. Temporary debugging code.
16. Experimental code.

---

## 20. Implementation Risks

| Risk | Severity | Cause | Affected Component | Detection Method | Mitigation |
| --- | --- | --- | --- | --- | --- |
| Exact simulation skeleton file or symbol is unverified. | High | Repository snapshot did not expose source details under `src/grasping_ai/simulation/`. | Simulation implementation. | Manual source inspection before implementation. | Modify only located existing units. If absent, pause for scope decision. |
| Exact robotics skeleton file or symbol is unverified. | High | Repository snapshot did not expose source details under `src/grasping_ai/robotics/`. | IK and gripper implementation. | Manual source inspection before implementation. | Modify only located existing units. If absent, pause for scope decision. |
| No existing Gymnasium-compatible environment class exists. | High | Gymnasium environments commonly use classes, but new classes are forbidden. | Simulation core. | Source inspection. | Use existing class or factory. If none exists, escalate. |
| MuJoCo dependency remains dev-only. | Medium | Phase 2 runtime code may not be installable in non-development environments. | Packaging and downstream phases. | Dependency import test. | Add MuJoCo to runtime dependencies. |
| Tests require unavailable YCB assets. | Medium | YCB dataset is external and not verified. | Phase 2 tests. | Test execution. | Use minimal local robot description and avoid YCB dependency in tests. |
| Observation or action contract mismatch breaks RL. | High | Downstream RL expects stable spaces. | Phase 5. | Observation and step tests. | Preserve existing skeleton contract and test shape stability. |
| IK solver behavior is robot-specific. | High | Robot model is not verified. | Robotics core. | IK tests with minimal model. | Implement only the existing skeleton contract; avoid hardcoded robot assumptions. |
| Accidental global state. | High | Simulation environments often cache state. | Whole package. | Structural review and no-state-leak test. | Keep state local to environment or simulation object. |
| Accidental helper abstraction. | Medium | Simulation setup and validation invite reuse. | Phase 2 modules. | Structural review. | Implement behavior directly in existing skeleton units. |

---

## 21. Design Decisions

| Decision | Evidence | Alternatives Considered | Reason for Selection |
| --- | --- | --- | --- |
| Add MuJoCo to runtime dependencies. | Phase 2 implements runtime simulation; MuJoCo currently verified only as dev dependency. | Keep MuJoCo dev-only. | Downstream phases require simulation at runtime. |
| Modify only existing simulation skeleton units. | Repository is a skeleton and constraints forbid new classes and utilities. | Create a new environment class or module. | Preserves architecture and constraints. |
| Modify only existing robotics skeleton units. | README places kinematics and gripper control under `robotics/`. | Create new IK or gripper utilities. | Preserves documented architecture and constraints. |
| Do not parse YAML configuration in Phase 2. | No config loader verified; YAML parser dependency not verified. | Add global config loader. | Avoids global state and unnecessary dependency. |
| Use explicit inputs for robot path, object root, and timing. | `scripts/train_rl.py` uses explicit CLI arguments. | Infer values from config implicitly. | Preserves explicit dependency flow and testability. |
| Avoid YCB assets in tests. | YCB dataset source and local availability are not verified. | Require YCB download for tests. | Keeps tests deterministic and self-contained. |
| Defer meaningful reward shaping. | Phase 2 scope is simulation and robotics core, not RL reward design. | Implement complex grasp reward now. | Keeps phase minimal and avoids unverified reward semantics. |

---

## 22. Explicitly Rejected Changes

The following changes were considered and rejected.

### Rejected: creating a new Gymnasium environment base class

A new base class would simplify environment implementation, but it violates the no-new-class constraint. If an existing environment class or factory exists, it must be used. If none exists, the phase must be paused.

### Rejected: creating a simulation helper module

Helper loaders, scene builders, or validation utilities would be reusable, but they violate the no-helper constraint. Required behavior must be implemented directly in the appropriate existing skeleton unit.

### Rejected: creating a global configuration loader

A config loader would make YAML files easier to use, but Phase 2 does not require global configuration. It would introduce hidden state and unnecessary dependency risk.

### Rejected: adding a YAML parsing dependency

YAML parsing is not required for Phase 2 behavior because explicit arguments can be used. Adding a YAML dependency would expand the dependency surface unnecessarily.

### Rejected: implementing full grasp evaluation

Force-closure, lift success, and collision reporting belong to later evaluation phases. Including them now would expand scope and increase regression risk.

### Rejected: implementing domain randomization

Domain randomization is a sim-to-real or robustness concern. It is not required for the minimum simulation core and would complicate testing.

### Rejected: implementing parallel environments

Parallel environments are performance optimizations. They are not required for Phase 2 correctness and would complicate the environment contract.

### Rejected: implementing real-robot control

Phase 2 is simulation-only. Real-robot control would introduce safety, hardware, and deployment concerns that are explicitly out of scope.

---

## 23. Verification Evidence

| Claim | Evidence |
| --- | --- |
| Repository uses `hatchling` and `src` layout. | Verified `pyproject.toml`. |
| Python requirement is `>=3.12`. | Verified `pyproject.toml`. |
| Gymnasium is a runtime dependency. | Verified `pyproject.toml`. |
| MuJoCo is a development dependency. | Verified `pyproject.toml`. |
| Simulation and robotics modules are intended. | README repository layout. |
| Simulation configuration is intended. | `docs/USAGE.md` description of `configs/simulation.yaml`. |
| Robot and IK configuration is intended. | `docs/USAGE.md` description of `configs/robot.yaml`. |
| RL training will require robot XML, YCB root, object IDs, observation dimension, and action dimension. | Verified `scripts/train_rl.py` CLI signature. |
| Repository is a skeleton. | README status section. |
| Exact simulation source files are unverified. | Available snapshot did not expose contents of `src/grasping_ai/simulation/`. |
| Exact robotics source files are unverified. | Available snapshot did not expose contents of `src/grasping_ai/robotics/`. |
| Existing tests are unverified. | Available snapshot did not expose a test suite. |

---

## 24. Definition of Done

Phase 2 is considered complete only when:

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

- [ ] Verified exact simulation skeleton file and symbol.
- [ ] Verified exact robotics skeleton file and symbol.
- [ ] Added MuJoCo to runtime dependencies.
- [ ] Implemented existing simulation initialization unit.
- [ ] Implemented existing reset unit.
- [ ] Implemented existing step unit.
- [ ] Implemented existing observation or state unit.
- [ ] Implemented existing IK unit.
- [ ] Implemented existing gripper-control unit.
- [ ] Did not create new classes.
- [ ] Did not create helper functions.
- [ ] Did not create utility modules.
- [ ] Did not introduce global variables.
- [ ] Did not introduce global constants.
- [ ] Did not add file-level descriptions.
- [ ] Added Google-style docstrings to modified functions or methods.

### Tests

- [ ] Added MuJoCo dependency test.
- [ ] Added package import regression test.
- [ ] Added simulation configuration existence test.
- [ ] Added robot configuration existence test.
- [ ] Added environment initialization test.
- [ ] Added missing robot description failure test.
- [ ] Added reset test.
- [ ] Added valid step test.
- [ ] Added invalid action shape test.
- [ ] Added non-finite action test.
- [ ] Added observation stability test.
- [ ] Added IK valid target test.
- [ ] Added IK invalid target test.
- [ ] Added gripper valid command test.
- [ ] Added gripper invalid command test.
- [ ] Added no-state-leak test.

### Regression Verification

- [ ] Phase 1 package import remains successful.
- [ ] Phase 1 dependency tests remain successful.
- [ ] Phase 1 SE(3) tests remain successful if present.
- [ ] No import cycles were introduced.
- [ ] No module-level side effects were introduced.

### Cross-Phase Verification

- [ ] Phase 5 can rely on reset, step, observation, and action contracts.
- [ ] Phase 6 can rely on simulation execution behavior.
- [ ] Simulation modules do not import pipelines.
- [ ] Robotics modules do not import training or inference.
- [ ] Configuration remains static and unparsed globally.

### Structural Constraints

- [ ] No new environment base class exists.
- [ ] No new robot abstraction class exists.
- [ ] No helper abstraction exists.
- [ ] No utility abstraction exists.
- [ ] No global configuration abstraction exists.
- [ ] No unrelated refactoring exists.
- [ ] No experimental code exists.

### Commit Readiness

- [ ] Targeted Phase 2 tests pass.
- [ ] Regression tests pass.
- [ ] Broader relevant test suite passes.
- [ ] Static and structural review confirms forbidden constructs are absent.
- [ ] Commit contains only Phase 2 changes.