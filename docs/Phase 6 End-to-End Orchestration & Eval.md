# Phase 6 — End-to-End Orchestration & Eval

## 1. Phase Objective

Phase 6 establishes the minimum end-to-end orchestration and evaluation capability required to connect the preceding phases into a verifiable grasping research pipeline.

The phase must deliver only the following capabilities:

1. A simulation execution pipeline that can load generated grasp poses, execute them in the Phase 2 MuJoCo simulation environment, and write a machine-readable simulation outcome report.
2. An evaluation pipeline that can load generated grasp poses and associated point-cloud data, compute evaluation metrics using the existing evaluation skeleton, and write a machine-readable evaluation report.
3. A stable report contract for downstream experiment analysis.
4. Tests that prove argument validation, report creation, metric output validity, and integration with preceding phase contracts without requiring GPUs, external datasets, or real robot hardware.

This phase is intentionally conservative. The repository is a skeleton, and the exact source files inside `src/grasping_ai/pipelines/`, `src/grasping_ai/evaluation/`, `src/grasping_ai/simulation/`, and `src/grasping_ai/robotics/` were not fully verifiable in the available snapshot. Therefore, this design distinguishes verified facts from unverified repository details. Where an exact file path or symbol cannot be verified, it is marked `Unverified`.

The implementation must not create new classes, helper utilities, global state, or abstraction layers. If the repository does not already contain the necessary skeleton pipeline, evaluation, simulation, or robotics units, the phase must be paused for a scope decision rather than creating a new orchestration class or utility module.

---

## 2. Verified Repository Context

The following facts were verified from the repository snapshot available during analysis.

### Verified script facts

- The repository script table lists:
  - `scripts/run_simulation.py`
  - `scripts/evaluate.py`
- `scripts/run_simulation.py` is documented as executing generated grasps in MuJoCo on YCB objects.
- `scripts/evaluate.py` is documented as evaluating generated grasps and writing a report.

### Verified usage facts

`docs/USAGE.md` documents the following arguments for simulation execution:

- `--grasps`
- `--object-id`
- `--ycb-root`
- `--robot-xml`
- `--output`
- `--num-simulation-steps`
- `--gripper-close-command`

`docs/USAGE.md` documents the following arguments for evaluation:

- `--grasps`
- `--object-id`
- `--object-point-cloud`
- `--gripper-point-cloud`
- `--report`
- `--friction-coefficient`
- `--lift-height-threshold`

`docs/USAGE.md` also documents example artifact paths:

- Generated grasps as `.npy` files.
- Simulation outcomes as JSON files.
- Evaluation reports as JSON files.

### Verified packaging facts

- `pyproject.toml` exists.
- The build backend is `hatchling.build`.
- The wheel target uses `packages = ["src"]`.
- Python requirement is `>=3.12`.
- Runtime dependencies already include:
  - `open3d`
  - `scipy`
  - `pytransform3d`
  - `theseus`
  - `torch`
  - `gymnasium`
  - `stable-baselines3`
- Development dependencies already include:
  - `mujoco`
  - `pytest`
  - `pytest-cov`
  - `ruff`
  - `mypy`

### Verified project documentation facts

- `README.md` states that the repository is a Python source-code skeleton and that function bodies raise `NotImplementedError`.
- `README.md` describes:
  - `src/grasping_ai/pipelines/` as end-to-end orchestration.
  - `src/grasping_ai/evaluation/` as force closure, collision, stability, and lift evaluation.
  - `src/grasping_ai/simulation/` as MuJoCo environment, scene, and YCB loading.
  - `src/grasping_ai/robotics/` as coordinate transforms, kinematics, and gripper control.
- `README.md` defines the simulation and evaluation workflow as:

  `pipelines.simulate_grasp -> simulation + robotics + evaluation`

- `docs/PROJECT.md` defines the dependency direction:

  `configs -> data -> perception -> models -> inference -> robotics -> simulation -> evaluation`

It also states that training and pipeline modules orchestrate lower-level components rather than being dependencies of them.

### Non-verified or partially verified areas

The following are not verified in the available snapshot:

- Exact contents of `scripts/run_simulation.py`.
- Exact contents of `scripts/evaluate.py`.
- Exact file names under `src/grasping_ai/pipelines/`.
- Exact file names under `src/grasping_ai/evaluation/`.
- Exact function names, class names, and signatures inside simulation, robotics, evaluation, and pipeline modules.
- Existing test files, fixtures, and test conventions.
- Exact YAML key names inside `configs/evaluation.yaml`.
- Whether evaluation metrics are analytical, simulation-based, or both.
- Whether the simulation execution pipeline uses direct gripper control, IK, or an RL policy.

Because of this, any source-unit design inside those directories is marked `Unverified` where exact paths or symbols cannot be confirmed.

---

## 3. Scope

Phase 6 includes only the following work.

### In scope

1. Implementation of the existing simulation execution pipeline skeleton.
   - Load generated grasp poses.
   - Validate grasp array shape.
   - Initialize the Phase 2 simulation environment.
   - Execute each grasp using existing robotics behavior.
   - Step the simulation for the requested number of steps.
   - Collect per-grasp simulation outcomes.
   - Write a JSON simulation outcome report.

2. Implementation of the existing evaluation pipeline skeleton.
   - Load generated grasp poses.
   - Load object and gripper point clouds.
   - Validate inputs.
   - Compute evaluation metrics using existing evaluation behavior.
   - Write a JSON evaluation report.

3. Implementation or completion of existing evaluation metric skeleton units.
   - Force closure.
   - Collision.
   - Stability or lift-related metrics where already declared by the skeleton.

4. Preservation of static evaluation configuration.
   - Do not remove `configs/evaluation.yaml`.
   - Do not introduce global configuration loading.
   - Do not introduce module-level configuration state.

5. Tests for the Phase 6 contract.
   - Argument validation.
   - Missing file failure.
   - Invalid grasp shape failure.
   - Simulation outcome report creation where a minimal simulation asset is available.
   - Evaluation report creation using synthetic point clouds.
   - Metric output validity.
   - Regression of preceding phase contracts.

### Conditional scope

The end-to-end implementation is conditional on locating existing skeleton units.

If no existing simulation execution pipeline function exists, Phase 6 must not create a new orchestration class. The correct action is to stop and request a scope decision.

If no existing evaluation pipeline function exists, Phase 6 must not create a new evaluation utility module. The correct action is to stop and request a scope decision.

If no existing evaluation metric skeleton exists for a documented metric, Phase 6 must not invent a new metric abstraction. The correct action is to stop and request a scope decision for that metric.

---

## 4. Out of Scope

The following are explicitly out of scope for Phase 6.

1. New model training.
   - Supervised grasp-model training.
   - RL policy training.
   - Model checkpoint hyperparameter tuning.

2. New generative inference behavior.
   - Grasp sampling improvements.
   - Grasp ranking.
   - Grasp filtering.
   - Grasp refinement.

3. Real-robot integration.
   - Hardware drivers.
   - Robot communication.
   - Safety controllers.
   - Calibration pipelines.
   - Deployment workflows.

4. Advanced simulation features.
   - Domain randomization.
   - Parallel environments.
   - Sensor noise simulation.
   - Actuator delay simulation.
   - New physics engines.
   - New robot models.

5. Advanced evaluation features.
   - Benchmark dashboards.
   - Statistical experiment management.
   - Metric learning.
   - Learning-based grasp quality prediction.

6. New architecture.
   - Orchestrator classes.
   - Metric factory classes.
   - Report builder classes.
   - Helper functions.
   - Utility modules.
   - Global configuration managers.

---

## 5. Existing Architecture and Patterns

Phase 6 must preserve the following verified or documented repository patterns.

### Pattern 1: Thin CLI scripts delegate to pipeline functions

The repository documentation describes scripts as thin CLI entry points. `scripts/train_rl.py` was verified as a thin script that parses arguments and calls a pipeline function.

Phase 6 must preserve this pattern for `scripts/run_simulation.py` and `scripts/evaluate.py`. The scripts should parse arguments and delegate to existing pipeline functions. They should not contain simulation, robotics, metric, or report-writing business logic.

### Pattern 2: Skeleton functions raise `NotImplementedError`

The README states that the repository contains skeleton code with `NotImplementedError` bodies. Phase 6 should replace `NotImplementedError` only in existing functions, methods, or module execution blocks that belong to the phase scope.

Phase 6 must not create new reusable helper functions to avoid implementing required behavior directly in the appropriate existing skeleton unit.

### Pattern 3: Pipelines orchestrate lower-level modules

The documented architecture places pipelines above simulation, robotics, evaluation, inference, and training.

Phase 6 pipeline code may orchestrate:

- Grasp loading.
- Simulation initialization.
- Grasp execution.
- Metric computation.
- Report writing.

Lower-level modules must not import the pipeline.

### Pattern 4: Evaluation remains separate from model and robot business logic

Evaluation modules should compute metrics from explicit inputs. They must not depend on training state, inference state, global configuration, or robot hardware.

### Pattern 5: Configuration remains static

The repository uses YAML configuration files, but no global configuration loader was verified.

Phase 6 must not introduce a global configuration object, module-level config cache, or singleton configuration manager. Evaluation thresholds and simulation arguments should be accepted explicitly through the existing CLI or pipeline interface.

---

## 6. Implementation Dependencies

Phase 6 requires the following dependencies.

### Existing dependencies that must remain available

| Dependency | Reason |
| --- | --- |
| `mujoco` | Required for simulation execution established by Phase 2. |
| `gymnasium` | Required for environment interaction where the Phase 2 environment uses Gymnasium behavior. |
| `open3d` | Required for point-cloud evaluation behavior. |
| `scipy` | Required for numerical evaluation behavior. |
| `pytransform3d` | Required for SE(3) transform behavior. |
| `theseus` | Existing spatial optimization dependency; use only if already required by the evaluation skeleton. |
| `numpy` | Required for `.npy` grasp and point-cloud array handling established by Phase 3. |
| `pytest` | Development dependency for Phase 6 tests. |

### Dependencies explicitly not required

| Dependency | Reason for exclusion |
| --- | --- |
| New report library | Standard JSON behavior is sufficient. |
| New metric library | Existing evaluation skeleton and dependencies are sufficient. |
| New configuration library | Global configuration is out of scope. |
| New visualization library | Phase 6 does not require visual debugging. |

No new third-party dependency may be introduced.

---

## 7. Source-Unit Change Matrix

Rows marked `Unverified` are required phase targets, but the exact file path or symbol must be resolved from the repository before implementation. They are not speculative new files.

| File | Symbol | Change Type | Current Responsibility | Proposed Change | Risk | Required Tests |
| --- | --- | --- | --- | --- | --- | --- |
| `scripts/run_simulation.py` | Unverified existing CLI entry function or module-level execution block | Modify | Documented entry point for executing generated grasps. | Preserve thin CLI behavior and ensure it delegates to the existing simulation pipeline. | Medium | Entry-point argument validation tests. |
| Unverified pipeline module imported by `scripts/run_simulation.py` | Unverified existing simulation execution pipeline function | Modify | Expected to orchestrate grasp execution in simulation. | Implement grasp loading, environment initialization, grasp execution, outcome collection, and JSON report writing. | High | Simulation execution and report tests. |
| `scripts/evaluate.py` | Unverified existing CLI entry function or module-level execution block | Modify | Documented entry point for evaluating generated grasps. | Preserve thin CLI behavior and ensure it delegates to the existing evaluation pipeline. | Medium | Entry-point argument validation tests. |
| Unverified pipeline module imported by `scripts/evaluate.py` | Unverified existing evaluation pipeline function | Modify | Expected to orchestrate grasp evaluation and report writing. | Implement grasp loading, point-cloud loading, metric invocation, and JSON report writing. | High | Evaluation report tests. |
| Unverified existing module under `src/grasping_ai/evaluation/` | Unverified existing force-closure metric function | Modify | Expected to compute force-closure behavior. | Implement force-closure metric using explicit inputs and existing dependencies. | High | Force-closure validity tests. |
| Unverified existing module under `src/grasping_ai/evaluation/` | Unverified existing collision metric function | Modify | Expected to compute collision behavior. | Implement collision metric using explicit inputs and existing dependencies. | High | Collision validity tests. |
| Unverified existing module under `src/grasping_ai/evaluation/` | Unverified existing stability or lift metric function | Modify if present | Expected to compute stability or lift behavior. | Implement stability or lift metric only where the skeleton already declares it. | Medium | Stability/lift validity tests if applicable. |
| Unverified existing module under `src/grasping_ai/simulation/` or `src/grasping_ai/robotics/` | Unverified existing grasp execution behavior | Modify if present | Expected to move robot to grasp pose and close gripper. | Implement grasp execution using Phase 2 simulation and Phase 2 robotics behavior. | High | Simulation execution tests. |
| `configs/evaluation.yaml` | Documented evaluation settings | Configuration preservation | Declares evaluation semantics such as friction and lift threshold according to documentation. | Preserve file and documented semantics. Do not introduce global loading. If exact keys are verified, keep them unchanged. | Low | Configuration existence test. |
| Existing or minimum required test file under `tests/unit/` | Phase 6 orchestration and evaluation tests | Test-only change | No verified existing Phase 6 tests. | Add tests for argument validation, report creation, metric validity, and integration behavior. | Medium | All Phase 6 tests pass. |

---

## 8. Detailed Source-Unit Design

### `scripts/run_simulation.py`

#### Current behavior

The script exists and is documented as the entry point for executing generated grasps in MuJoCo. The exact CLI function or module-level execution block was not verified. Because the repository is a skeleton, the current behavior is expected to be `NotImplementedError` or delegation to an unimplemented pipeline function.

#### Required behavior

The script must support the documented simulation execution entry point. It must accept explicit arguments and invoke the existing simulation execution pipeline.

The script must remain thin. It must not contain simulation stepping, grasp execution, metric calculation, or report-writing logic.

#### Design

##### Input

Expected inputs include:

- Generated grasps path.
- Object identifier.
- YCB root path.
- Robot XML path.
- Output report path.
- Number of simulation steps.
- Gripper close command.

Additional inputs may exist in the skeleton. The implementation must preserve the existing CLI contract.

##### Output

The script itself produces no meaningful return value. Its side effect is the creation of a JSON simulation outcome report.

##### Control flow

1. Parse CLI arguments using the existing argument parsing behavior.
2. Call the existing simulation execution pipeline function.
3. Propagate errors naturally.

##### State changes

No module-level state.

##### Existing dependencies

- Existing simulation pipeline module.
- Standard argument parsing if already used by the script.

##### Error behavior

- Missing generated grasps file raises a filesystem error.
- Missing robot XML path raises a filesystem error.
- Missing YCB root raises a filesystem error.
- Invalid output path raises a filesystem or value error.

##### Edge cases

- Output parent directory does not exist.
- Grasps file contains zero grasps.
- Gripper close command has invalid length or values.

##### Interaction with surrounding code

Later experiment analysis consumes the JSON simulation outcome report.

##### Existing pattern being followed

Thin CLI script delegates to lower-level behavior.

---

### Unverified existing simulation execution pipeline function

#### Current behavior

The exact pipeline symbol was not verified. The current behavior is expected to be `NotImplementedError`.

#### Required behavior

The simulation execution pipeline must orchestrate execution of generated grasps in the Phase 2 simulation environment and write a JSON report.

#### Design

##### Input

Expected inputs include:

- Generated grasps path.
- Object identifier.
- YCB root path.
- Robot XML path.
- Output report path.
- Number of simulation steps.
- Gripper close command.

##### Output

The function may return the report path or a report structure if the existing skeleton contract defines a return value. Its side effect is the creation of a JSON simulation outcome report.

##### Control flow

1. Validate explicit arguments.
2. Load generated grasps from the `.npy` file.
3. Validate grasp array shape and SE(3) structure.
4. Initialize the Phase 2 simulation environment using the robot XML, YCB root, and object identifier.
5. For each grasp:
   1. Reset the environment.
   2. Move the robot or gripper toward the grasp pose using existing robotics behavior.
   3. Apply the gripper close command.
   4. Step the simulation for the requested number of steps.
   5. Collect outcome fields.
6. Write the JSON report.

##### State changes

Simulation state changes during execution. Module-level state must not be introduced.

##### Existing dependencies

- Phase 2 simulation initialization and stepping behavior.
- Phase 2 robotics or gripper-control behavior.
- Phase 4 generated grasp contract.
- Standard JSON serialization.

##### Error behavior

- Missing generated grasps file raises a filesystem error.
- Invalid grasp shape raises a value error.
- Invalid SE(3) transforms raise a value error.
- Simulation initialization failure propagates naturally.

##### Edge cases

- Zero grasps.
- Single grasp.
- Batched grasp array.
- Gripper command length mismatch.
- Object asset missing.

##### Interaction with surrounding code

The report may be consumed by later experiment analysis or by evaluation behavior if future orchestration chooses to combine reports.

##### Existing pattern being followed

Pipeline orchestration remains above simulation and robotics modules.

---

### `scripts/evaluate.py`

#### Current behavior

The script exists and is documented as the entry point for evaluating generated grasps and writing a report. The exact CLI function or module-level execution block was not verified. Because the repository is a skeleton, the current behavior is expected to be `NotImplementedError` or delegation to an unimplemented pipeline function.

#### Required behavior

The script must support the documented evaluation entry point. It must accept explicit arguments and invoke the existing evaluation pipeline.

The script must remain thin. It must not contain point-cloud processing, metric computation, or report-writing logic.

#### Design

##### Input

Expected inputs include:

- Generated grasps path.
- Object identifier.
- Object point-cloud path.
- Gripper point-cloud path.
- Report output path.
- Friction coefficient.
- Lift height threshold.

Additional inputs may exist in the skeleton. The implementation must preserve the existing CLI contract.

##### Output

The script itself produces no meaningful return value. Its side effect is the creation of a JSON evaluation report.

##### Control flow

1. Parse CLI arguments using the existing argument parsing behavior.
2. Call the existing evaluation pipeline function.
3. Propagate errors naturally.

##### State changes

No module-level state.

##### Existing dependencies

- Existing evaluation pipeline module.
- Standard argument parsing if already used by the script.

##### Error behavior

- Missing generated grasps file raises a filesystem error.
- Missing point-cloud file raises a filesystem error.
- Invalid friction coefficient raises a value error.
- Invalid lift threshold raises a value error.

##### Edge cases

- Report parent directory does not exist.
- Grasps file contains zero grasps.
- Point cloud is empty.

##### Interaction with surrounding code

Later experiment analysis consumes the JSON evaluation report.

##### Existing pattern being followed

Thin CLI script delegates to lower-level behavior.

---

### Unverified existing evaluation pipeline function

#### Current behavior

The exact pipeline symbol was not verified. The current behavior is expected to be `NotImplementedError`.

#### Required behavior

The evaluation pipeline must orchestrate evaluation of generated grasps and write a JSON report.

#### Design

##### Input

Expected inputs include:

- Generated grasps path.
- Object identifier.
- Object point-cloud path.
- Gripper point-cloud path.
- Report output path.
- Friction coefficient.
- Lift height threshold.

##### Output

The function may return the report path or a report structure if the existing skeleton contract defines a return value. Its side effect is the creation of a JSON evaluation report.

##### Control flow

1. Validate explicit arguments.
2. Load generated grasps from the `.npy` file.
3. Load object point cloud.
4. Load gripper point cloud.
5. Validate shapes and finite values.
6. For each grasp:
   1. Invoke force-closure behavior.
   2. Invoke collision behavior.
   3. Invoke stability or lift behavior if the skeleton declares it.
   4. Collect per-grasp metric results.
7. Aggregate results.
8. Write the JSON report.

##### State changes

Filesystem side effect: writes report. No module-level state.

##### Existing dependencies

- Phase 3 point-cloud loading or perception behavior where applicable.
- Existing evaluation metric units.
- Standard JSON serialization.

##### Error behavior

- Missing generated grasps file raises a filesystem error.
- Missing point-cloud file raises a filesystem error.
- Invalid point-cloud shape raises a value error.
- Non-finite point-cloud values raise a value error.
- Invalid metric output raises a value error.

##### Edge cases

- Zero grasps.
- Empty point cloud.
- Single-point point cloud.
- Friction coefficient at boundary values.
- Lift threshold at boundary values.

##### Interaction with surrounding code

The evaluation report is consumed by later experiment analysis.

##### Existing pattern being followed

Pipeline orchestration remains above evaluation modules.

---

### Unverified existing force-closure metric function

#### Current behavior

The exact force-closure symbol was not verified. The current behavior is expected to be `NotImplementedError`.

#### Required behavior

The force-closure metric must evaluate whether a grasp can resist external disturbances under the specified friction coefficient.

#### Design

##### Input

Expected inputs include:

- Grasp pose.
- Object point cloud.
- Gripper point cloud.
- Friction coefficient.

##### Output

The output must be a JSON-serializable metric. The preferred minimum contract is a boolean or a finite scalar quality value.

##### Control flow

1. Validate grasp pose.
2. Validate point clouds.
3. Transform gripper points into the grasp frame using existing SE(3) behavior.
4. Identify contact or proximity relationships between gripper points and object points.
5. Evaluate friction or force-closure conditions using the existing skeleton algorithm.
6. Return the metric.

##### State changes

None.

##### Existing dependencies

- NumPy.
- Open3D.
- SciPy.
- pytransform3d or Phase 1 SE(3) behavior.

##### Error behavior

- Invalid grasp pose raises a value error.
- Invalid point-cloud shape raises a value error.
- Negative friction coefficient raises a value error.
- Non-finite values raise a value error.

##### Edge cases

- No contact points.
- Single contact point.
- Friction coefficient zero.
- Very large friction coefficient.

##### Interaction with surrounding code

The evaluation pipeline calls this metric for each grasp.

##### Existing pattern being followed

Evaluation remains stateless and low-level.

---

### Unverified existing collision metric function

#### Current behavior

The exact collision symbol was not verified. The current behavior is expected to be `NotImplementedError`.

#### Required behavior

The collision metric must evaluate whether a grasp pose produces unacceptable penetration or collision between gripper geometry and object geometry.

#### Design

##### Input

Expected inputs include:

- Grasp pose.
- Object point cloud.
- Gripper point cloud.
- Optional collision tolerance if present in the existing skeleton.

##### Output

The output must be a JSON-serializable metric. The preferred minimum contract is a boolean collision-free flag or a finite scalar penetration measure.

##### Control flow

1. Validate grasp pose.
2. Validate point clouds.
3. Transform gripper points into the grasp frame.
4. Evaluate distance or penetration between gripper points and object points.
5. Return the collision metric.

##### State changes

None.

##### Existing dependencies

- NumPy.
- Open3D.
- SciPy.

##### Error behavior

- Invalid grasp pose raises a value error.
- Invalid point-cloud shape raises a value error.
- Non-finite values raise a value error.

##### Edge cases

- Empty point cloud.
- Single-point point cloud.
- Grasps far from object.
- Grasps intersecting object.

##### Interaction with surrounding code

The evaluation pipeline calls this metric for each grasp.

##### Existing pattern being followed

Evaluation remains stateless and low-level.

---

### Unverified existing stability or lift metric function

#### Current behavior

The exact stability or lift symbol was not verified. The current behavior is expected to be `NotImplementedError` if such a unit exists.

#### Required behavior

If the skeleton declares a stability or lift metric, Phase 6 must implement that existing behavior.

If the skeleton does not declare such a metric, Phase 6 must not invent a new metric abstraction. The lift threshold should still be recorded in the evaluation report as an explicit input parameter.

#### Design

##### Input

Expected inputs may include:

- Grasp pose.
- Object point cloud.
- Gripper point cloud.
- Lift height threshold.
- Simulation outcome fields if the existing skeleton supports them.

##### Output

The output must be JSON-serializable. The preferred minimum contract is a boolean, a finite scalar, or an explicit null value when the metric is not computable from available inputs.

##### Control flow

1. Validate inputs.
2. Compute stability or lift behavior according to the existing skeleton contract.
3. Return the metric.

##### State changes

None.

##### Existing dependencies

- Existing evaluation skeleton.
- NumPy or SciPy where required.

##### Error behavior

- Invalid threshold raises a value error.
- Non-finite inputs raise a value error.

##### Edge cases

- Threshold zero.
- Metric not computable without simulation outcome.
- Missing object state.

##### Interaction with surrounding code

The evaluation pipeline includes this metric only when the skeleton declares it.

##### Existing pattern being followed

Evaluation remains stateless and low-level.

---

### Unverified existing grasp execution behavior

#### Current behavior

The exact grasp execution symbol was not verified. The current behavior is expected to be `NotImplementedError` if such a unit exists.

#### Required behavior

The grasp execution behavior must execute a single grasp pose in the Phase 2 simulation environment using existing robotics behavior.

#### Design

##### Input

Expected inputs include:

- Initialized simulation environment or simulation state.
- Grasp pose.
- Gripper close command.
- Number of simulation steps.

##### Output

The output must include execution outcome fields required by the simulation report.

##### Control flow

1. Validate grasp pose.
2. Move robot or gripper toward the grasp pose using existing IK or control behavior.
3. Apply gripper close command.
4. Step the simulation for the requested number of steps.
5. Collect outcome fields.

##### State changes

Simulation state changes. Module-level state must not change.

##### Existing dependencies

- Phase 2 simulation stepping behavior.
- Phase 2 robotics or gripper-control behavior.

##### Error behavior

- Invalid grasp pose raises a value error.
- Invalid gripper command raises a value error.
- Uninitialized simulation state raises an error.

##### Edge cases

- Grasp pose far from object.
- Gripper command at boundary values.
- Simulation termination before requested steps.

##### Interaction with surrounding code

The simulation execution pipeline calls this behavior for each grasp.

##### Existing pattern being followed

Robotics control remains separate from pipeline orchestration.

---

### `configs/evaluation.yaml`

#### Current behavior

Documentation states that this file contains evaluation settings such as friction coefficient and minimum lift height threshold. The exact key names were not verified.

#### Required behavior

Phase 6 must preserve the file and its documented semantics. Phase 6 must not introduce global configuration loading. Evaluation thresholds should be accepted explicitly through the existing CLI or pipeline interface.

#### Design

- Input: none.
- Output: none.
- Control flow: none.
- State changes: none.
- Existing dependencies: none.
- Error behavior: not applicable during Phase 6 because no runtime YAML parsing is introduced.
- Edge cases: missing keys should be handled by later pipeline validation if that pipeline chooses to consume the file.
- Interaction with surrounding code: later orchestration may read this file, but Phase 6 does not require it.
- Existing pattern being followed: static configuration without global state.

No YAML key may be renamed, removed, or added unless exact key verification proves that a documented key is missing.

---

## 9. Data Flow

Phase 6 introduces two primary data flows.

### Simulation execution data flow

`Generated grasps file -> Simulation execution pipeline -> Phase 2 simulation environment -> Robotics execution -> Simulation outcome report`

Input source:

- Phase 4 generated grasps file.
- Explicit simulation arguments.

Transformation:

- Grasps are loaded and validated.
- Simulation environment is initialized.
- Each grasp is executed.
- Outcome fields are collected.

Validation:

- Grasp shape.
- SE(3) validity.
- Robot XML existence.
- YCB root existence.
- Output path validity.

Output:

- JSON simulation outcome report.

Consumers:

- Later experiment analysis.
- Potential future combined evaluation pipelines.

### Evaluation data flow

`Generated grasps file -> Evaluation pipeline -> Point-cloud loading -> Evaluation metrics -> Evaluation report`

Input source:

- Phase 4 generated grasps file.
- Object point cloud.
- Gripper point cloud.
- Explicit evaluation thresholds.

Transformation:

- Grasps are loaded.
- Point clouds are loaded and validated.
- Metrics are computed per grasp.
- Results are aggregated.

Validation:

- Grasp shape.
- Point-cloud shape.
- Finite values.
- Friction coefficient validity.
- Lift threshold validity.

Output:

- JSON evaluation report.

Consumers:

- Later experiment analysis.
- Research comparison workflows.

---

## 10. Execution Flow

Phase 6 uses the existing script entry points.

### Simulation execution path

`scripts/run_simulation.py -> Unverified existing simulation execution pipeline -> Phase 2 simulation environment -> Phase 2 robotics behavior -> JSON outcome report`

Phase 6 implements the missing behavior in the appropriate existing units. It does not implement new simulation engines or robot controllers.

### Evaluation path

`scripts/evaluate.py -> Unverified existing evaluation pipeline -> Existing evaluation metrics -> JSON evaluation report`

Phase 6 implements the missing behavior in the appropriate existing units. It does not implement new metric libraries.

### Direct validation path

For Phase 6 validation, tests may directly call the located pipeline and metric skeleton units. This avoids depending on unimplemented downstream experiment tooling.

### Error propagation

Errors must propagate naturally:

- Missing files propagate filesystem errors.
- Invalid arrays propagate value errors.
- Simulation failures propagate simulation errors.
- Metric failures propagate value errors.

Phase 6 must not swallow errors silently.

---

## 11. Configuration Changes

Phase 6 requires no YAML configuration additions.

### Existing configuration preserved

| File | Key area | Existing semantics | Required Phase 6 treatment | Consumed by |
| --- | --- | --- | --- | --- |
| `configs/evaluation.yaml` | Friction coefficient, lift threshold, evaluation settings | Documented evaluation semantics. | Preserve file and documented semantics. Do not introduce global loading. | Future orchestration if it explicitly consumes configuration. |

### Configuration changes not allowed

Phase 6 must not:

- Add a new YAML schema.
- Add environment-variable configuration.
- Add global config state.
- Add config parser utilities.
- Add default values in code that shadow configuration values unless the existing skeleton already defines such defaults.

Thresholds and simulation arguments must remain explicit through the verified CLI or pipeline contract.

---

## 12. Test Strategy

Testing must use `pytest`, which is already a development dependency.

No existing test suite was verified. Therefore, the implementation engineer must first check whether a test directory and existing conventions are present.

If tests already exist, Phase 6 tests must follow the existing conventions for:

- File naming.
- Test function naming.
- Fixture usage.
- Assertion style.
- Parameterization.
- Mocking, if already used.

If no tests exist, Phase 6 may add the minimum required test file under a conventional unit-test location. This is a test-only addition.

### Testing principles

1. Tests must be deterministic enough for CI.
2. Tests must not require network access.
3. Tests must not require downloaded datasets.
4. Tests must not require YCB assets unless Phase 2 simulation tests already require them.
5. Tests must not require GPUs.
6. Tests must not require real robot hardware.
7. Tests must not introduce global state.
8. Tests must not create helper functions or helper classes.
9. Tests must use tiny grasp arrays and tiny point clouds.
10. Tests must avoid long simulation runs.

### Asset strategy

Because external datasets are not guaranteed, tests must use temporary synthetic data.

The preferred approach is:

- Create temporary generated-grasp arrays inside the test scope.
- Create temporary point-cloud arrays inside the test scope.
- Use the minimal simulation description established by Phase 2 tests if simulation execution is tested.
- Avoid committing binary assets.
- Avoid creating shared helper fixture code unless an existing fixture already provides the same behavior.

If Phase 2 simulation cannot initialize without external YCB assets, simulation execution tests are blocked until a minimal testable simulation contract is available. Evaluation report tests that do not require simulation should still be possible using synthetic point clouds.

---

## 13. Test Suite

The following tests are required. Rows that depend on unverified skeleton symbols are marked accordingly.

| Test | File | Target | Scenario | Expected Result | Regression Risk |
| --- | --- | --- | --- | --- | --- |
| `test_phase1_package_import_remains_stable` | Existing or minimum required Phase 6 unit test file | `src/grasping_ai/__init__.py` | Import `grasping_ai`. | Import succeeds. | High if Phase 6 changes package initialization. |
| `test_evaluation_config_file_exists` | Existing or minimum required Phase 6 unit test file | `configs/evaluation.yaml` | Check file existence. | File exists. | Low. |
| `test_run_simulation_rejects_missing_grasps_file` | Existing or minimum required Phase 6 unit test file | Unverified simulation execution pipeline | Provide missing generated grasps path. | Raises filesystem or value error. | Medium. |
| `test_run_simulation_rejects_invalid_grasp_shape` | Existing or minimum required Phase 6 unit test file | Unverified simulation execution pipeline | Provide invalid grasp array shape. | Raises value error. | High. |
| `test_run_simulation_rejects_missing_robot_xml` | Existing or minimum required Phase 6 unit test file | Unverified simulation execution pipeline | Provide missing robot XML path. | Raises filesystem or value error. | Medium. |
| `test_run_simulation_creates_outcome_report` | Existing or minimum required Phase 6 unit test file | Unverified simulation execution pipeline | Execute one tiny synthetic grasp with minimal simulation assets. | JSON outcome report exists and contains per-grasp outcome fields. | High. |
| `test_run_simulation_outcome_fields_are_json_serializable` | Existing or minimum required Phase 6 unit test file | Unverified simulation execution pipeline | Inspect report content. | Report contains JSON-serializable fields. | Medium. |
| `test_evaluate_rejects_missing_grasps_file` | Existing or minimum required Phase 6 unit test file | Unverified evaluation pipeline | Provide missing generated grasps path. | Raises filesystem or value error. | Medium. |
| `test_evaluate_rejects_missing_point_cloud` | Existing or minimum required Phase 6 unit test file | Unverified evaluation pipeline | Provide missing object or gripper point-cloud path. | Raises filesystem or value error. | Medium. |
| `test_evaluate_rejects_invalid_point_cloud_shape` | Existing or minimum required Phase 6 unit test file | Unverified evaluation pipeline | Provide invalid point-cloud shape. | Raises value error. | Medium. |
| `test_evaluate_creates_report_from_synthetic_inputs` | Existing or minimum required Phase 6 unit test file | Unverified evaluation pipeline | Evaluate one synthetic grasp with synthetic point clouds. | JSON evaluation report exists and contains per-grasp metrics. | High. |
| `test_force_closure_metric_returns_valid_output` | Existing or minimum required Phase 6 unit test file | Unverified force-closure metric | Evaluate one valid synthetic grasp. | Output is boolean or finite scalar and JSON-serializable. | High. |
| `test_collision_metric_returns_valid_output` | Existing or minimum required Phase 6 unit test file | Unverified collision metric | Evaluate one valid synthetic grasp. | Output is boolean or finite scalar and JSON-serializable. | High. |
| `test_evaluate_rejects_negative_friction_coefficient` | Existing or minimum required Phase 6 unit test file | Unverified evaluation pipeline or metric | Provide negative friction coefficient. | Raises value error. | Medium. |
| `test_evaluate_rejects_negative_lift_threshold` | Existing or minimum required Phase 6 unit test file | Unverified evaluation pipeline or metric | Provide negative lift threshold. | Raises value error. | Medium. |
| `test_phase6_pipelines_do_not_leak_global_state` | Existing or minimum required Phase 6 unit test file | Phase 6 pipeline units | Run pipeline twice with different temporary outputs. | Outputs depend only on explicit inputs. | Medium. |

---

## 14. Regression Test Plan

Phase 6 can introduce regressions in package imports, simulation contracts, generated-grasp consumption, point-cloud loading, and report expectations.

### Unit-level regressions

| Existing behavior | Why it could break | Protecting test | Expected result |
| --- | --- | --- | --- |
| Package import succeeds. | Pipeline or evaluation modules may introduce import cycles or heavy imports. | `test_phase1_package_import_remains_stable` | Import succeeds. |
| Phase 3 point-cloud loading remains stable. | Evaluation may pressure changes into point-cloud loading or perception. | Phase 3 tests and Phase 6 evaluation tests. | Phase 3 behavior remains unchanged. |
| Phase 4 generated grasp contract remains stable. | Simulation and evaluation may assume a different grasp array shape. | Phase 6 grasp validation tests. | Phase 4 output contract remains compatible. |

### Integration-level regressions

| Existing behavior | Why it could break | Protecting test | Expected result |
| --- | --- | --- | --- |
| Phase 2 environment step contract remains stable. | Grasp execution may alter simulation stepping behavior. | Phase 2 environment tests and Phase 6 simulation tests. | Step contract remains stable. |
| Scripts can import pipeline modules. | Phase 6 pipeline implementation may introduce import errors. | Existing import-related tests, if present. | Import path remains valid. |

### Pipeline-level regressions

No broader experiment-management pipeline behavior is verified. Therefore, no higher-level pipeline regression test can be specified without invention.

If an existing pipeline smoke test exists, it must be rerun. If no such test exists, Phase 6 must not create a broad experiment-management smoke test.

---

## 15. Impact on Preceding Phases

### Phase 1 — Foundation & Math Primitives

| Item | Description |
| --- | --- |
| Existing contract | Phase 1 established package importability, dependency availability, static configuration preservation, and pure SE(3) math behavior where verified. |
| What Phase 6 changes | Phase 6 adds orchestration and evaluation behavior that may consume SE(3) transforms. |
| What remains compatible | Package layout, existing dependencies, static configuration files, and Phase 1 math behavior must remain unchanged. |
| Previous-phase tests to rerun | Phase 1 package import test, dependency tests, and SE(3) math tests if present. |
| Whether Phase 1 needs modification | No modification is expected unless dependency resolution fails. |

### Phase 2 — Simulation & Robotics Core

| Item | Description |
| --- | --- |
| Existing contract | Phase 2 established simulation initialization, reset, step, observation, action, and basic robotics behavior where verified. |
| What Phase 6 changes | Phase 6 uses Phase 2 simulation and robotics behavior to execute grasps. It may complete an existing grasp-execution skeleton. |
| What remains compatible | Environment reset, step, observation, and action contracts must remain unchanged unless explicitly required by the existing skeleton. |
| Previous-phase tests to rerun | Phase 2 environment initialization, reset, step, invalid action, observation stability, and gripper tests if present. |
| Whether Phase 2 needs modification | Only existing grasp-execution or robotics behavior may need completion. No unrelated Phase 2 modification is expected. |

### Phase 3 — Data Pipeline & Perception

| Item | Description |
| --- | --- |
| Existing contract | Phase 3 established dataset indexing, point-cloud loading, and perception preprocessing. |
| What Phase 6 changes | Phase 6 may consume point-cloud loading behavior for evaluation. |
| What remains compatible | Point-cloud shape, finite-value validation, and coordinate-frame behavior must remain stable. |
| Previous-phase tests to rerun | Phase 3 point-cloud loading and perception preprocessing tests. |
| Whether Phase 3 needs modification | No modification is expected unless the point-cloud loading contract is verified to be incompatible. |

### Phase 4 — Generative Grasp Model

| Item | Description |
| --- | --- |
| Existing contract | Phase 4 established generated grasp output shape and SE(3) validity. |
| What Phase 6 changes | Phase 6 consumes generated grasp files. |
| What remains compatible | Generated grasps must remain valid homogeneous transforms in `.npy` format. |
| Previous-phase tests to rerun | Phase 4 inference output shape and SE(3) validity tests. |
| Whether Phase 4 needs modification | No modification is expected unless the generated grasp contract is verified to be incompatible. |

### Phase 5 — Reinforcement Learning Policy

| Item | Description |
| --- | --- |
| Existing contract | Phase 5 established RL policy training and checkpoint saving where verified. |
| What Phase 6 changes | Phase 6 does not require the RL checkpoint for the verified `run_simulation.py` argument contract. |
| What remains compatible | Phase 5 checkpoint behavior remains unchanged. |
| Previous-phase tests to rerun | Phase 5 tests if they are part of the broader relevant suite. |
| Whether Phase 5 needs modification | No modification is expected. |

---

## 16. Impact on Downstream Phases

Phase 6 is the final implementation phase in the verified roadmap. There are no verified later implementation phases.

However, future experiment analysis may depend on Phase 6 outputs.

### Future experiment analysis

| Item | Description |
| --- | --- |
| Dependency | Experiment analysis may consume simulation outcome reports and evaluation reports. |
| Interface relied on | JSON report structure, metric field names, and explicit input metadata. |
| New behavior available after Phase 6 | Machine-readable reports exist for generated grasps. |
| Constraints preserved | Reports must remain JSON-serializable and deterministic for fixed inputs. |
| Protecting tests | Report creation, field presence, and metric validity tests. |

---

## 17. Cross-Phase Contract

Phase 6 establishes the following contract.

### Inputs

1. Simulation execution:
   - Generated grasps path.
   - Object identifier.
   - YCB root path.
   - Robot XML path.
   - Output report path.
   - Number of simulation steps.
   - Gripper close command.

2. Evaluation:
   - Generated grasps path.
   - Object identifier.
   - Object point-cloud path.
   - Gripper point-cloud path.
   - Report output path.
   - Friction coefficient.
   - Lift height threshold.

3. Metrics:
   - Grasp pose.
   - Object point cloud.
   - Gripper point cloud.
   - Explicit metric parameters.

### Outputs

1. Simulation execution:
   - JSON simulation outcome report.

2. Evaluation:
   - JSON evaluation report.

3. Metrics:
   - JSON-serializable metric values.

### Expected behavior

1. Pipelines validate explicit inputs before execution.
2. Generated grasps are consumed in the Phase 4 output contract.
3. Metrics return finite, JSON-serializable values.
4. Reports are written only to explicit output paths.
5. No global state is introduced.

### Invariants

1. No module-level mutable state.
2. No global configuration state.
3. No dependency on real robot hardware.
4. Generated grasps remain SE(3) homogeneous transforms.
5. Reports remain JSON-serializable.

### Error behavior

1. Missing required files raise explicit filesystem or value errors.
2. Invalid grasp shapes raise value errors.
3. Invalid point-cloud shapes raise value errors.
4. Invalid friction or lift thresholds raise value errors.
5. Simulation failures propagate naturally.

### Configuration assumptions

1. YAML files remain static.
2. Phase 6 does not parse YAML globally.
3. Thresholds are explicit through CLI or pipeline arguments.

### Data assumptions

1. Generated grasps are stored as `.npy` arrays.
2. Point clouds are numeric arrays with shape `N x 3`.
3. Coordinates are expressed in the frame provided by the data unless an existing skeleton explicitly declares transformation behavior.
4. Reports are UTF-8 JSON-serializable mappings.

### Performance assumptions

1. Single-process execution is sufficient.
2. Parallel simulation is out of scope.
3. Tests must use tiny grasp arrays and tiny point clouds.
4. Long physics rollouts are avoided in tests.

---

## 18. Validation Before Commit

The implementation must not be considered ready for commit until the following sequence is complete.

1. Run Phase 6 package import regression tests.
2. Run Phase 6 configuration existence tests.
3. Run simulation pipeline argument validation tests.
4. Run simulation grasp validation tests.
5. Run simulation outcome report tests where minimal simulation assets are available.
6. Run evaluation pipeline argument validation tests.
7. Run evaluation point-cloud validation tests.
8. Run evaluation report creation tests.
9. Run force-closure metric validity tests.
10. Run collision metric validity tests.
11. Run stability or lift metric validity tests if applicable.
12. Rerun Phase 1 tests.
13. Rerun Phase 2 simulation tests.
14. Rerun Phase 3 point-cloud tests.
15. Rerun Phase 4 generated grasp tests.
16. Run the broader relevant test suite.
17. Verify that no unrelated module changed.
18. Verify that no new class was introduced.
19. Verify that no helper or utility function was introduced.
20. Verify that no global variable or constant was introduced.
21. Verify that no file-level source description was introduced.
22. Verify that every modified or newly introduced function or method has a Google-style docstring.
23. Verify that no unrelated refactoring is included.
24. Verify that no new dependency was introduced.
25. Verify that downstream report contracts remain compatible.

The final commit gate is:

`Implementation -> Targeted Tests -> Regression Tests -> Integration Tests -> Full Relevant Suite -> Static/Structural Review -> Commit`

---

## 19. Commit Boundary

The Phase 6 commit must contain only the following.

### Allowed in the commit

1. Modifications to verified existing simulation execution skeleton units.
2. Modifications to verified existing evaluation pipeline skeleton units.
3. Modifications to verified existing evaluation metric skeleton units.
4. Modifications to verified existing grasp execution skeleton units.
5. Modifications to verified existing script entry behavior where required to preserve thin CLI delegation.
6. Required Phase 6 tests.
7. Minimal test-only synthetic assets created inside test scope.

### Not allowed in the commit

1. New orchestrator classes.
2. New metric factory classes.
3. New report builder classes.
4. New helper functions.
5. New utility modules.
6. New global constants.
7. New global variables.
8. New configuration loaders.
9. Unrelated formatting changes.
10. Refactoring of scripts unrelated to Phase 6.
11. Training code.
12. Model code.
13. Real-robot integration code.
14. Temporary debugging code.
15. Experimental code.

---

## 20. Implementation Risks

| Risk | Severity | Cause | Affected Component | Detection Method | Mitigation |
| --- | --- | --- | --- | --- | --- |
| Exact pipeline skeleton files are unverified. | High | Repository snapshot did not expose full contents of pipeline modules. | End-to-end orchestration. | Manual source inspection before implementation. | Modify only located existing units. If absent, pause for scope decision. |
| Exact evaluation metric skeletons are unverified. | High | Repository snapshot did not expose contents of evaluation modules. | Metric implementation. | Manual source inspection. | Implement only existing metric units. If absent, pause for scope decision. |
| Simulation execution may require YCB assets. | High | Documented usage references YCB objects. | Simulation tests. | Test execution. | Use minimal simulation assets where possible. Block simulation tests if no minimal asset contract exists. |
| Grasp execution control path is unverified. | High | It is unclear whether execution uses IK, direct joint commands, or RL policy. | Simulation execution. | Source inspection. | Follow existing skeleton. Do not invent a new control path. |
| Evaluation algorithms are mathematically complex. | High | Force closure and collision checking require geometry and friction reasoning. | Evaluation metrics. | Metric validity tests. | Implement only existing skeleton behavior. Keep outputs JSON-serializable and finite. |
| Report schema mismatch. | Medium | Downstream analysis may expect specific fields. | Reports. | Report field tests. | Define minimal explicit report contract and preserve it. |
| Phase 4 grasp shape mismatch. | High | Phase 6 may expect a different grasp array layout. | Simulation and evaluation loading. | Grasp validation tests. | Validate generated grasp shape explicitly and fail early. |
| Creating new helper abstractions would violate constraints. | Medium | Orchestration and metrics invite reuse. | Phase 6 modules. | Structural review. | Implement behavior directly in existing skeleton units. |

---

## 21. Design Decisions

| Decision | Evidence | Alternatives Considered | Reason for Selection |
| --- | --- | --- | --- |
| Preserve separate simulation execution and evaluation pipelines. | Repository documents separate scripts `run_simulation.py` and `evaluate.py`. | Create one unified end-to-end orchestrator. | Matches existing script contract and avoids unnecessary abstraction. |
| Keep scripts thin and delegate to pipeline functions. | Verified pattern in `scripts/train_rl.py` and documented architecture. | Implement logic directly in scripts. | Preserves existing architecture. |
| Consume Phase 4 generated grasps as `.npy` homogeneous transforms. | USAGE examples show generated grasps as `.npy` and simulation consumes generated grasps. | Invent a new grasp file format. | Preserves cross-phase contract. |
| Write JSON reports. | USAGE examples show JSON output paths for simulation outcomes and evaluation reports. | Use pickle, CSV, or database output. | Matches documented usage and standard library support. |
| Do not parse YAML configuration in Phase 6. | No config loader verified; CLI arguments already explicit. | Add global config loader. | Avoids global state and unnecessary dependency. |
| Do not require RL policy checkpoint for simulation execution. | Verified `run_simulation.py` usage arguments do not include a policy checkpoint. | Force Phase 5 checkpoint into simulation execution. | Avoids inventing an interface not evidenced by the repository. |
| Implement metrics inside existing evaluation units. | README describes evaluation modules for force closure, collision, stability, and lift. | Create a new metric library or factory. | Preserves architecture and constraints. |
| Keep evaluation stateless. | Evaluation should be reproducible and low-level. | Add global metric registry or cache. | Avoids hidden state and simplifies testing. |

---

## 22. Explicitly Rejected Changes

The following changes were considered and rejected.

### Rejected: creating an orchestrator class

A central orchestrator class could coordinate simulation, evaluation, and reporting, but it violates the no-new-class constraint and conflicts with the repository pipeline-function pattern.

### Rejected: creating helper functions for report building

Report building is required, but creating separate helper utilities would violate the no-helper constraint. Required report behavior must be implemented directly inside the appropriate existing pipeline unit.

### Rejected: creating helper functions for metric conversion

Metric conversion is required, but separate conversion utilities would violate the no-helper constraint. Required conversion behavior must be implemented directly inside the appropriate existing evaluation unit.

### Rejected: adding a global configuration loader

A config loader would make `configs/evaluation.yaml` easier to use, but Phase 6 does not require global configuration. It would introduce hidden state and unnecessary dependency risk.

### Rejected: adding visualization

Visualization of grasps, point clouds, or simulation outcomes is useful for debugging but is not required for the minimum end-to-end contract. It would expand scope and introduce unnecessary dependencies.

### Rejected: adding parallel simulation execution

Parallel simulation is a performance optimization. It is not required for Phase 6 correctness and would complicate testing.

### Rejected: adding RL policy execution to `run_simulation.py`

The verified usage contract for `run_simulation.py` does not include a policy checkpoint argument. Adding one would invent a new interface and expand scope.

### Rejected: adding experiment tracking or benchmark databases

Phase 6 only needs machine-readable reports. Experiment management tooling is out of scope.

---

## 23. Verification Evidence

| Claim | Evidence |
| --- | --- |
| Repository uses `hatchling` and `src` layout. | Verified `pyproject.toml`. |
| Python requirement is `>=3.12`. | Verified `pyproject.toml`. |
| Open3D, SciPy, pytransform3d, and theseus are existing dependencies. | Verified `pyproject.toml`. |
| MuJoCo is a development dependency. | Verified `pyproject.toml`. |
| Simulation execution script exists. | Repository script table. |
| Evaluation script exists. | Repository script table. |
| Simulation execution consumes generated grasps and writes JSON outcomes. | `docs/USAGE.md` example arguments. |
| Evaluation consumes generated grasps and point clouds and writes JSON reports. | `docs/USAGE.md` example arguments. |
| Evaluation modules are intended for force closure, collision, stability, and lift. | README repository layout. |
| Pipelines are intended for end-to-end orchestration. | README repository layout. |
| Simulation and robotics modules are intended. | README repository layout. |
| Exact pipeline source files are unverified. | Available snapshot did not expose contents of `src/grasping_ai/pipelines/`. |
| Exact evaluation source files are unverified. | Available snapshot did not expose contents of `src/grasping_ai/evaluation/`. |
| Existing tests are unverified. | Available snapshot did not expose a test suite. |

---

## 24. Definition of Done

Phase 6 is considered complete only when:

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

- [ ] Verified exact simulation execution pipeline file and symbol.
- [ ] Verified exact evaluation pipeline file and symbol.
- [ ] Verified exact evaluation metric skeleton units.
- [ ] Verified exact grasp execution skeleton unit if present.
- [ ] Implemented simulation execution pipeline behavior.
- [ ] Implemented evaluation pipeline behavior.
- [ ] Implemented force-closure metric behavior.
- [ ] Implemented collision metric behavior.
- [ ] Implemented stability or lift metric behavior only if already declared by the skeleton.
- [ ] Preserved thin CLI behavior for `scripts/run_simulation.py`.
- [ ] Preserved thin CLI behavior for `scripts/evaluate.py`.
- [ ] Did not create new classes.
- [ ] Did not create helper functions.
- [ ] Did not create utility modules.
- [ ] Did not introduce global variables.
- [ ] Did not introduce global constants.
- [ ] Did not add file-level descriptions.
- [ ] Added Google-style docstrings to modified functions or methods.

### Tests

- [ ] Added package import regression test.
- [ ] Added evaluation configuration existence test.
- [ ] Added missing generated grasps failure tests.
- [ ] Added missing robot XML failure test.
- [ ] Added invalid grasp shape failure tests.
- [ ] Added missing point-cloud failure tests.
- [ ] Added invalid point-cloud shape failure tests.
- [ ] Added simulation outcome report creation test where minimal simulation assets are available.
- [ ] Added simulation report JSON-serializability test.
- [ ] Added evaluation report creation test.
- [ ] Added force-closure metric validity test.
- [ ] Added collision metric validity test.
- [ ] Added friction coefficient validation test.
- [ ] Added lift threshold validation test.
- [ ] Added no-global-state test.

### Regression Verification

- [ ] Phase 1 package import remains successful.
- [ ] Phase 1 dependency tests remain successful.
- [ ] Phase 2 environment initialization remains successful.
- [ ] Phase 2 reset and step tests remain successful.
- [ ] Phase 3 point-cloud loading tests remain successful.
- [ ] Phase 4 generated grasp contract remains successful.
- [ ] Phase 5 tests remain successful if present.
- [ ] No import cycles were introduced.
- [ ] No module-level side effects were introduced.

### Cross-Phase Verification

- [ ] Simulation execution consumes Phase 4 generated grasps.
- [ ] Evaluation consumes Phase 4 generated grasps.
- [ ] Evaluation consumes Phase 3 point-cloud data.
- [ ] Simulation execution uses Phase 2 simulation and robotics behavior.
- [ ] Reports are JSON-serializable and explicit.
- [ ] Evaluation modules do not import training or inference state.
- [ ] Pipelines do not introduce global configuration.

### Structural Constraints

- [ ] No new orchestrator class exists.
- [ ] No new metric factory exists.
- [ ] No new report builder class exists.
- [ ] No helper abstraction exists.
- [ ] No utility abstraction exists.
- [ ] No global configuration abstraction exists.
- [ ] No unrelated refactoring exists.
- [ ] No experimental code exists.

### Commit Readiness

- [ ] Targeted Phase 6 tests pass.
- [ ] Regression tests pass.
- [ ] Broader relevant test suite passes.
- [ ] Static and structural review confirms forbidden constructs are absent.
- [ ] Commit contains only Phase 6 changes.