# Phase 1 — Foundation & Math Primitives

> **Historical design record.** This document captures the Phase 1 plan from the skeleton era. Current architecture, CLIs, and contracts live in [architecture.md](architecture.md), [USAGE.md](USAGE.md), and the repository [README.md](../README.md).

## 1. Phase Objective

The objective of Phase 1 is to establish the minimum verified foundation required before any simulation, data, model, training, evaluation, or inference work begins.

Phase 1 must achieve only the following:

1. Confirm that the Python package foundation is stable and importable under the existing build layout.
2. Preserve the existing dependency boundary and avoid introducing new dependencies unless a verified repository requirement forces it.
3. Preserve the existing configuration files and define a stable, non-global configuration contract for the keys already described by the repository.
4. Implement the smallest set of SE(3) math behaviors that already exist as skeleton functions in the repository, without creating new utility modules, helper functions, helper classes, global state, or new abstractions.
5. Add only the tests necessary to prove that the package foundation, dependency set, configuration contract, and verified SE(3) primitive behaviors are stable.

This phase is intentionally conservative. The repository is a skeleton, and several source paths inside `src/grasping_ai/` could not be fully verified in the available snapshot. Therefore, this design distinguishes verified facts from unverified assumptions and avoids expanding scope into areas that are not yet proven to exist.

---

## 2. Verified Repository Context

The following items were verified from the repository snapshot available during analysis.

### Verified root and packaging facts

- The repository contains a `pyproject.toml` file.
- `pyproject.toml` declares:
  - project name `vr-rlft-side-project`
  - Python requirement `>=3.12`
  - build backend `hatchling.build`
  - wheel packages target `["src"]`
  - runtime dependencies including:
    - `gymnasium`
    - `matplotlib`
    - `open3d`
    - `pytransform3d`
    - `scipy`
    - `stable-baselines3`
    - `tensorboard`
    - `theseus`
    - `torch`
    - `torchvision`
    - `wandb`
  - development dependencies including:
    - `ruff`
    - `pytest`
    - `pytest-cov`
    - `mypy`
    - `mujoco`

### Verified documentation facts

- `README.md` states that the repository contains a Python source-code skeleton and that function bodies raise `NotImplementedError`.
- `README.md` describes `src/grasping_ai/perception/` as containing point-cloud preprocessing and SE(3) helpers.
- `docs/PROJECT.md` describes the intended dependency direction:

  `configs -> data -> perception -> models -> inference -> robotics -> simulation -> evaluation`

- `docs/PROJECT.md` states that training and pipeline modules orchestrate lower-level components rather than being dependencies of them.
- `docs/USAGE.md` describes configuration files under `configs/`:
  - `configs/base.yaml`
  - `configs/data.yaml`
  - `configs/model.yaml`
  - `configs/training.yaml`
  - `configs/evaluation.yaml`
  - `configs/robot.yaml`
  - `configs/simulation.yaml`
- `docs/USAGE.md` states that `configs/base.yaml` contains general configuration such as target device, random seed, and output directory.

### verified script facts

- `scripts/train_rl.py` exists and contains a thin CLI entry point.
- `scripts/train_rl.py` imports:

  `python scripts/train_rl.py`

- This confirms that `src/grasping_ai/pipelines/train_rl.py`, or an equivalent pipeline module, is expected to exist in the package layout.

### Verified absence or non-verification

- A direct request for `src/grasping_ai/simulation/env.py` returned a 404 result in the available snapshot. Therefore, the exact simulation file layout is not verified.
- The contents of the following were not verified:
  - `src/grasping_ai/perception/`
  - `src/grasping_ai/robotics/`
  - `src/grasping_ai/models/`
  - `src/grasping_ai/inference/`
  - `src/grasping_ai/data/`
  - `src/grasping_ai/training/`
  - `src/grasping_ai/pipelines/`
  - `tests/`
- Because of this, any source-unit design inside those directories is marked `Unverified` where exact file paths or symbols cannot be confirmed.

---

## 3. Scope

Phase 1 includes only the following work.

### In scope

1. Verification and preservation of the existing packaging foundation.
   - Keep the existing `hatchling` build backend.
   - Keep the existing `src` package layout.
   - Keep the existing Python version requirement.
   - Keep the existing runtime dependency set unless a verified missing dependency blocks Phase 1.

2. Preservation of the existing base configuration contract.
   - Keep `configs/base.yaml` as a static configuration artifact.
   - Preserve the documented keys for device, random seed, and output directory.
   - Do not introduce a global configuration loader.
   - Do not introduce module-level configuration state.

3. Implementation of only already-existing SE(3) skeleton behaviors under `src/grasping_ai/perception/`, if such skeleton functions are verified during implementation.
   - Use existing dependencies such as `pytransform3d`, `scipy`, and NumPy-compatible arrays.
   - Do not create new utility modules.
   - Do not create helper functions outside the existing skeleton function being completed.
   - Do not create classes.

4. Tests for the Phase 1 contract.
   - Package import test.
   - Dependency availability test for the math stack required by Phase 1.
   - Configuration file existence and key-contract test.
   - SE(3) primitive tests, only for verified existing skeleton functions.

### Conditional scope

The SE(3) math portion of this phase is conditional on locating existing skeleton functions in `src/grasping_ai/perception/`. If no such functions exist, Phase 1 must not create a new math utility module. In that case, the correct action is to stop and request a scope decision, because creating a new utility module would violate the phase constraints.

---

## 4. Out of Scope

The following are explicitly out of scope for Phase 1.

1. Simulation implementation.
   - MuJoCo environment behavior.
   - Gymnasium environment wrappers.
   - Scene loading.
   - YCB asset loading.
   - Physics stepping.
   - Rendering.

2. Data pipeline implementation.
   - Dataset indexing.
   - Dataset loading.
   - Point-cloud file ingestion.
   - Data augmentation.
   - Data splitting.

3. Model implementation.
   - Equivariant encoder.
   - Diffusion model.
   - Flow model.
   - RL policy network.
   - Model checkpoint loading.
   - Model inference.

4. Training pipeline implementation.
   - Supervised training loops.
   - RL training loops.
   - Optimizer setup.
   - Learning-rate scheduling.
   - Experiment logging.
   - Checkpoint saving.

5. Evaluation pipeline implementation.
   - Force-closure evaluation.
   - Collision evaluation.
   - Lift-success evaluation.
   - Report generation.

6. Inference pipeline implementation.
   - Grasp generation.
   - Policy rollout.
   - Action post-processing.
   - Observation preprocessing beyond verified SE(3) primitive behavior.

7. Real-robot integration.
   - Hardware drivers.
   - Robot communication.
   - Safety logic.
   - Deployment logic.

8. New architecture.
   - New utility packages.
   - New abstraction layers.
   - New class hierarchies.
   - New global configuration systems.
   - New plugin systems.

9. Unrequired dependency additions.
   - YAML parsing libraries.
   - New math libraries.
   - New logging frameworks.
   - New CLI frameworks.

---

## 5. Existing Architecture and Patterns

The repository establishes several patterns that Phase 1 must preserve.

### Pattern 1: Thin CLI scripts delegate to pipeline functions

`scripts/train_rl.py` demonstrates the pattern:

- The script parses CLI arguments.
- The script calls a function inside `grasping_ai.pipelines`.
- The script does not contain core business logic.

Phase 1 must not move logic into scripts. Foundation behavior must remain inside the appropriate package modules or remain static configuration.

### Pattern 2: Skeleton functions raise `NotImplementedError`

The README states that function bodies raise `NotImplementedError`. Phase 1 should only replace `NotImplementedError` in existing functions that belong to the Phase 1 scope. It must not create new reusable helper functions to avoid implementing the required behavior directly in the appropriate existing function.

### Pattern 3: Low-level modules do not depend on orchestration

`docs/PROJECT.md` defines a dependency direction where low-level modules such as `perception` do not depend on pipelines or training orchestration.

Phase 1 must preserve this rule. Any SE(3) math behavior implemented under `src/grasping_ai/perception/` must not import pipeline, training, simulation, or inference modules.

### Pattern 4: Configuration is static and explicit

The repository uses YAML configuration files under `configs/`. There is no verified global configuration loader.

Phase 1 must not introduce a global configuration object, module-level config cache, or singleton config manager. If later phases need configuration values, they should receive them explicitly through function arguments or through pipeline-level orchestration.

### Pattern 5: Existing math dependencies are already selected

The repository already includes:

- `pytransform3d`
- `theseus`
- `scipy`

Phase 1 must use these existing dependencies where math behavior is required. It must not introduce a new math dependency.

---

## 6. Implementation Dependencies

Phase 1 requires only dependencies already present in the verified `pyproject.toml`.

### Required existing dependencies

| Dependency | Reason |
| --- | --- |
| `pytransform3d` | Provides SE(3) transform, rotation, quaternion, and homogeneous transform primitives. |
| `scipy` | Provides numerical utilities that may be used by existing skeleton math functions. |
| `torch` | Required by the broader package, but Phase 1 should avoid using Torch unless an existing skeleton function already requires it. |
| `pytest` | Development dependency for Phase 1 tests. |

### Dependencies explicitly not required

| Dependency | Reason for exclusion |
| --- | --- |
| YAML parsing library | Phase 1 does not require runtime parsing of YAML configuration. Introducing YAML parsing would add an unnecessary dependency and would create a config-loading abstraction that is not required by the phase. |
| New math utility library | Existing `pytransform3d` and `scipy` are sufficient. |
| New configuration library | Global configuration is out of scope. |
| New logging library | Phase 1 does not require logging beyond normal test output. |

---

## 7. Source-Unit Change Matrix

Only source units that are verified or conditionally verified are listed. Rows marked `Unverified` require the implementation engineer to first locate the actual existing skeleton file and symbol before modifying anything.

| File | Symbol | Change Type | Current Responsibility | Proposed Change | Risk | Required Tests |
| --- | --- | --- | --- | --- | --- | --- |
| `pyproject.toml` | `project.dependencies` | Configuration change | Declares runtime dependencies. | Preserve existing dependencies. Do not add new dependencies. Verify that `pytransform3d`, `scipy`, and `theseus` remain present. | Low | Dependency import test. |
| `pyproject.toml` | `tool.hatch.build.targets.wheel.packages` | Configuration change | Declares wheel packaging from `src`. | Preserve `src` packaging. Do not reorganize package layout. | Low | Package import test. |
| `configs/base.yaml` | `device`, `random_seed`, `output_dir` | Configuration change | Declares base experiment settings according to `USAGE.md`. | Preserve these keys and their intended scalar semantics. Do not introduce code that reads them into global state. | Low | Configuration file existence and key-contract test. |
| `src/grasping_ai/__init__.py` | package initialization | Modify only if importability is broken | Initializes the `grasping_ai` package. | Preserve package importability. Do not add module-level state, imports, or side effects. | Low | Package import test. |
| Unverified existing file under `src/grasping_ai/perception/` | Unverified existing SE(3) skeleton function | Modify | Currently expected to raise `NotImplementedError`. | Implement the existing function directly using `pytransform3d` and related existing dependencies. Do not create a new helper function. | High | SE(3) primitive unit tests. |
| Unverified existing test file or new Phase 1 test file under `tests/unit/` | Phase 1 foundation tests | Test-only change | No verified existing Phase 1 tests. | Add tests for package import, dependency availability, config file contract, and verified SE(3) behavior. | Medium | All Phase 1 tests pass. |

---

## 8. Detailed Source-Unit Design

### `pyproject.toml::project.dependencies`

#### Current behavior

The verified `pyproject.toml` declares the runtime dependencies required by the project. It already includes the dependencies relevant to Phase 1 math work:

- `pytransform3d`
- `scipy`
- `theseus`

It also includes broader dependencies for later phases, such as `gymnasium`, `stable-baselines3`, `torch`, `torchvision`, `open3d`, and `wandb`.

#### Required behavior

Phase 1 must preserve the existing dependency set. It must not add new dependencies. It must not remove dependencies needed by later phases. It must not change the package name, Python requirement, or build backend.

#### Design

The implementation should treat `pyproject.toml` as a protected foundation file.

- Input: none.
- Output: none.
- Control flow: none.
- State changes: none.
- Existing dependencies: unchanged.
- Error behavior: not applicable.
- Edge cases: not applicable.
- Interaction with surrounding code: the build system and test runner rely on this file.
- Existing pattern being followed: keep packaging configuration minimal and declarative.

No code should be added to consume dependencies in `pyproject.toml` during Phase 1, except where an existing skeleton function already requires them.

---

### `pyproject.toml::tool.hatch.build.targets.wheel.packages`

#### Current behavior

The verified `pyproject.toml` declares:

- `packages = ["src"]`

This establishes the `src` layout as the package source root.

#### Required behavior

Phase 1 must preserve the `src` layout. It must not move `src/grasping_ai` to another directory. It must not introduce a second package root. It must not rename the package.

#### Design

- Input: none.
- Output: none.
- Control flow: none.
- State changes: none.
- Existing dependencies: `hatchling`.
- Error behavior: not applicable.
- Edge cases: not applicable.
- Interaction with surrounding code: the RL training entrypoint is `scripts/train_rl.py`.
- Existing pattern being followed: source code remains under `src`, while CLI scripts remain under `scripts`.

---

### `configs/base.yaml::device`

#### Current behavior

`docs/USAGE.md` states that `configs/base.yaml` contains general configuration such as target device. The exact file content was not fully extracted, but the key contract is documented.

#### Required behavior

The configuration file must continue to represent a target device setting, such as `cpu` or `cuda`. Phase 1 must not parse this value into global state. Phase 1 must not create a config-loader function.

#### Design

- Input: none.
- Output: none.
- Control flow: none.
- State changes: none.
- Existing dependencies: none.
- Error behavior: not applicable during Phase 1 because no runtime parsing is introduced.
- Edge cases: device validation belongs to a later phase that actually consumes the value.
- Interaction with surrounding code: later pipeline functions may receive the device as an explicit argument.
- Existing pattern being followed: configuration remains static and explicit.

---

### `configs/base.yaml::random_seed`

#### Current behavior

The documented base configuration includes a random seed.

#### Required behavior

The key must remain part of the base configuration contract. Phase 1 must not introduce a global seed-setting function. Seed handling must remain local to the future component that needs it.

#### Design

- Input: none.
- Output: none.
- Control flow: none.
- State changes: none.
- Existing dependencies: none.
- Error behavior: not applicable.
- Edge cases: seed type validation and seeding behavior belong to a later phase.
- Interaction with surrounding code: training, evaluation, and inference phases may later receive the seed as an explicit argument.
- Existing pattern being followed: no global mutable state.

---

### `configs/base.yaml::output_dir`

#### Current behavior

The documented base configuration includes an output directory, described as `artifacts`.

#### Required behavior

The key must remain part of the base configuration contract. Phase 1 must not create directories as a side effect of importing the package. It must not introduce a global artifact manager.

#### Design

- Input: none.
- Output: none.
- Control flow: none.
- State changes: none.
- Existing dependencies: none.
- Error behavior: not applicable.
- Edge cases: directory creation belongs to the pipeline that writes artifacts.
- Interaction with surrounding code: later phases may pass the output directory explicitly.
- Existing pattern being followed: side effects are avoided in foundation code.

---

### `src/grasping_ai/__init__.py`

#### Current behavior

The file exists as part of the package skeleton. Its exact contents were not verified, but it must allow `grasping_ai` to be imported as a package.

#### Required behavior

The package initialization must remain side-effect free. It must not import heavy modules unless the existing skeleton already does so. It must not define global constants, global configuration objects, or helper utilities.

#### Design

- Input: none.
- Output: none.
- Control flow: package initialization only.
- State changes: none.
- Existing dependencies: none beyond the Python import system.
- Error behavior: import errors must surface naturally if the package layout is broken.
- Edge cases: none.
- Interaction with surrounding code: scripts such as `scripts/train_rl.py` import from `grasping_ai`.
- Existing pattern being followed: thin package root with no orchestration logic.

If the file is empty, it should remain empty unless a verified importability problem requires a minimal fix.

---

### Unverified existing SE(3) helper module under `src/grasping_ai/perception/`

#### Current behavior

The repository documentation states that `src/grasping_ai/perception/` contains SE(3) helpers. The exact file name and function names were not verified in the available snapshot.

Because the repository is described as a skeleton, the likely current behavior of the relevant function or functions is:

- Accept math inputs related to SE(3) transforms.
- Raise `NotImplementedError`.

This current behavior is marked `Unverified`.

#### Required behavior

The existing SE(3) skeleton function or functions must implement only the minimum math behavior required by downstream phases.

The required behaviors are:

1. Accept homogeneous transform inputs as 4x4 numeric arrays.
2. Validate transform shape.
3. Validate rotation matrix structure where required.
4. Normalize or sanitize SE(3) transforms where the existing skeleton function already declares that responsibility.
5. Use `pytransform3d` for transform math where appropriate.
6. Raise explicit exceptions for invalid input.
7. Return only the math result.
8. Avoid side effects.
9. Avoid global state.
10. Avoid creating new helper functions.

#### Design

Because the exact symbol is unverified, the implementation engineer must first locate the existing function that already declares SE(3) helper behavior.

Once located, the implementation should follow this logic.

##### Input

The expected input is one of the following, depending on the existing skeleton signature:

- A 4x4 homogeneous transform.
- A rotation matrix and translation vector.
- A quaternion and translation vector.
- Another SE(3) representation already declared by the skeleton function.

The implementation must not widen the function signature beyond what is required by the existing skeleton and downstream callers.

##### Output

The output must match the existing skeleton contract.

Possible outputs include:

- A normalized 4x4 homogeneous transform.
- A validated rotation matrix.
- A validated quaternion.
- A composed SE(3) transform.
- An inverted SE(3) transform.

The implementation must not return richer objects, classes, or wrappers.

##### Control flow

The function should:

1. Validate input type and shape.
2. Convert the input to the numeric representation required by `pytransform3d` if conversion is already part of the skeleton contract.
3. Call the appropriate existing `pytransform3d` primitive directly.
4. Return the resulting array or scalar.
5. Raise an exception if validation fails.

The function must not delegate to a newly created helper function. If multiple operations are required, they must be written directly inside the existing function unless an already existing repository function provides the exact behavior.

##### State changes

None.

##### Existing dependencies

- `pytransform3d`
- `scipy`, if already used by the skeleton function.
- NumPy-compatible array behavior, where provided indirectly through existing dependencies.

##### Error behavior

The function must raise standard Python exceptions:

- `TypeError` for unsupported types.
- `ValueError` for invalid shapes, invalid rotation structure, or invalid numeric contents.

The function must not silently correct invalid transforms unless the existing skeleton contract explicitly defines sanitization as its responsibility.

##### Edge cases

The implementation must handle:

- Identity transform.
- Near-identity rotation.
- Orthogonal but numerically imperfect rotation matrices.
- Translation-only transforms.
- Invalid 3x3, 4x3, 3x4, or flat inputs.
- Non-finite values where validation is required by the skeleton contract.

##### Interaction with surrounding code

Downstream perception, models, robotics, or simulation phases may call this function. Because the exact callers are not verified, the function must remain stable, pure, and side-effect free.

##### Existing pattern being followed

The function must follow the skeleton pattern:

- Implement only the declared responsibility.
- Do not orchestrate pipelines.
- Do not access configuration.
- Do not access hardware.
- Do not create global state.

---

## 9. Data Flow

Phase 1 introduces or preserves only minimal data flows.

### Configuration data flow

`configs/base.yaml` is a static artifact.

The Phase 1 data flow is:

`Static YAML file -> Future explicit argument passing -> Future pipeline function`

Phase 1 does not introduce a runtime configuration parser.

### SE(3) math data flow

The intended SE(3) data flow is:

`Raw transform or rotation input -> Existing perception SE(3) skeleton function -> Validated or normalized SE(3) output -> Downstream model, robotics, or simulation code`

Phase 1 does not connect this flow to training, inference, or simulation. It only ensures that the math primitive, once verified, behaves correctly in isolation.

### Package import data flow

`Python import system -> src layout -> grasping_ai package -> downstream scripts and modules`

Phase 1 must preserve this import path.

---

## 10. Execution Flow

Phase 1 does not introduce a new runtime execution pipeline.

The verified execution context is:

1. A script under `scripts/` is invoked.
2. The script imports a `grasping_ai.pipelines` function.
3. The pipeline function will later orchestrate lower-level modules.

Phase 1 affects only the foundation required for step 2 and the low-level math primitives that may later be called during step 3.

### Verified execution path

`scripts/train_rl.py`

Phase 1 does not implement `run_rl_training_pipeline`. It only ensures that the package foundation required by this import path remains valid.

### SE(3) execution path

The SE(3) execution path is currently unverified because the exact caller is unverified.

The intended path is:

`Future caller -> existing perception SE(3) function -> pytransform3d math -> returned transform`

No side effects are expected.

---

## 11. Configuration Changes

Phase 1 does not require new configuration.

### Existing configuration preserved

| File | Key | Existing Value or Semantics | Required Phase 1 Treatment | Consumed By |
| --- | --- | --- | --- | --- |
| `configs/base.yaml` | `device` | Target compute device. | Preserve key and scalar semantics. Do not parse globally. | Future training, evaluation, or inference pipeline. |
| `configs/base.yaml` | `random_seed` | Random seed for reproducibility. | Preserve key and scalar semantics. Do not set globally. | Future training, evaluation, or inference pipeline. |
| `configs/base.yaml` | `output_dir` | Output artifact directory. | Preserve key and path semantics. Do not create directories as import side effect. | Future pipeline that writes artifacts. |

### Configuration changes not allowed

Phase 1 must not:

- Add a new YAML schema.
- Add environment-variable configuration.
- Add global config state.
- Add config validation classes.
- Add config parser utilities.
- Add default values in code that shadow configuration values.

---

## 12. Test Strategy

The test strategy must follow the repository’s existing development dependency: `pytest`.

No existing test suite was verified. Therefore, the implementation engineer must first check whether a `tests/` directory already exists and whether it contains unit tests.

If tests already exist, Phase 1 tests must follow the existing naming, fixture, assertion, and organization patterns.

If no tests exist, Phase 1 may introduce the minimum required test file under a conventional `tests/unit/` location. This is a test-only addition and is permitted because testing is mandatory for this phase.

### Test principles

1. Tests must be deterministic.
2. Tests must not depend on network access.
3. Tests must not depend on GPUs.
4. Tests must not depend on downloaded datasets.
5. Tests must not depend on MuJoCo simulation.
6. Tests must not require model checkpoints.
7. Tests must not require external services.
8. Tests must not introduce global state.
9. Tests must not create helper utilities.
10. Tests must not require YAML parsing unless a YAML dependency is already verified.

Because no YAML parsing dependency is verified, configuration tests should validate file existence and key presence through plain text inspection only. This is intentionally conservative.

---

## 13. Test Suite

The following tests are required. Some rows are marked `Unverified` because the exact SE(3) source file and symbol are not verified.

| Test | File | Target | Scenario | Expected Result | Regression Risk |
| --- | --- | --- | --- | --- | --- |
| `test_package_imports` | `tests/unit/test_phase1_foundation.py` | `src/grasping_ai/__init__.py` | Import `grasping_ai`. | Import succeeds without side effects. | High if package layout breaks. |
| `test_math_dependencies_available` | `tests/unit/test_phase1_foundation.py` | `pyproject.toml` dependency set | Import `pytransform3d` and `scipy`. | Imports succeed. | Medium if dependency set is accidentally changed. |
| `test_base_config_file_exists` | `tests/unit/test_phase1_foundation.py` | `configs/base.yaml` | Check that the file exists. | File exists. | Low. |
| `test_base_config_contains_contract_keys` | `tests/unit/test_phase1_foundation.py` | `configs/base.yaml` | Read file as plain text and verify that `device`, `random_seed`, and `output_dir` appear as keys. | All required key names are present. | Low to medium; text-based check is brittle but avoids adding YAML dependency. |
| `test_pyproject_preserves_src_package_layout` | `tests/unit/test_phase1_foundation.py` | `pyproject.toml` | Read file as plain text and verify that the wheel package target includes `src`. | `src` packaging remains present. | Low. |
| `test_se3_primitive_identity_behavior` | `tests/unit/test_phase1_se3.py` or existing verified SE(3) test file | Unverified existing SE(3) function | Provide an identity 4x4 transform. | Function returns an identity-equivalent transform within tolerance. | High if math convention changes. |
| `test_se3_primitive_invalid_shape` | `tests/unit/test_phase1_se3.py` or existing verified SE(3) test file | Unverified existing SE(3) function | Provide an invalid shape such as 3x3 where 4x4 is required. | Function raises `ValueError` or `TypeError`. | Medium. |
| `test_se3_primitive_non_finite_input` | `tests/unit/test_phase1_se3.py` or existing verified SE(3) test file | Unverified existing SE(3) function | Provide NaN or infinite values if validation is part of the skeleton contract. | Function raises an exception or follows documented sanitization behavior. | Medium. |
| `test_se3_primitive_no_global_side_effects` | `tests/unit/test_phase1_se3.py` or existing verified SE(3) test file | Unverified existing SE(3) function | Call the function twice with different inputs. | Outputs depend only on inputs; no module state changes. | Medium. |

### Test notes

- The SE(3) tests must only be written after the actual SE(3) function is located.
- If no SE(3) skeleton function exists, the SE(3) tests must not be invented around a new utility function. Instead, the phase must be paused for scope clarification.
- Tests must use plain assertions and standard `pytest` behavior.
- Tests must not introduce fixtures that hide global state.
- Tests must not use mocks unless an external dependency is unavailable, which should not be necessary for Phase 1.

---

## 14. Regression Test Plan

Phase 1 is small, but it can still introduce regressions.

### Unit-level regressions

| Existing behavior | Why it could break | Protecting test | Expected result |
| --- | --- | --- | --- |
| `grasping_ai` package imports successfully. | A change to `src` layout or package initialization could break imports. | `test_package_imports` | Import succeeds. |
| Math dependencies are available. | Dependency removal or version constraint change could break imports. | `test_math_dependencies_available` | Imports succeed. |
| SE(3) functions remain pure. | Adding global state or caching could change repeated-call behavior. | `test_se3_primitive_no_global_side_effects` | Outputs are input-dependent only. |

### Integration-level regressions

| Existing behavior | Why it could break | Protecting test | Expected result |
| --- | --- | --- | --- |
| Scripts can import pipeline modules. | Package initialization changes could create import cycles. | Existing script import behavior, or a future smoke test if already present. | Import path remains valid. |
| Future perception modules can use SE(3) primitives. | Incorrect math convention could break downstream callers. | SE(3) identity and invalid-shape tests. | Math contract remains stable. |

### Pipeline-level regressions

No pipeline-level behavior is verified yet. Therefore, no pipeline-level regression test can be specified without invention.

If a smoke test for `scripts/train_rl.py` already exists, it should be rerun. If it does not exist, Phase 1 must not create a broad pipeline smoke test.

---

## 15. Impact on Preceding Phases

No verified impact on preceding phases.

Phase 1 is the first phase in the implementation roadmap. There is no preceding implementation phase whose contract must be preserved.

---

## 16. Impact on Downstream Phases

Phase 1 establishes constraints and contracts that downstream phases must rely on.

### Downstream Phase 2 — Simulation & Robotics Core

| Item | Description |
| --- | --- |
| Dependency | Phase 2 will need stable SE(3) math behavior for robot transforms and object poses. |
| Interface relied on | Pure SE(3) functions under `src/grasping_ai/perception/` or another verified low-level module. |
| New behavior after Phase 1 | Validated SE(3) primitives are available without global state. |
| Constraints preserved | No global config, no helper utilities, no classes, no side effects. |
| Protecting tests | SE(3) identity, invalid input, and no-side-effect tests. |

### Downstream Phase 3 — Data Pipeline & Perception

| Item | Description |
| --- | --- |
| Dependency | Phase 3 will need point-cloud preprocessing to coexist with SE(3) helpers. |
| Interface relied on | Perception module remains low-level and does not depend on pipelines. |
| New behavior after Phase 1 | Perception math primitives are stable and test-covered. |
| Constraints preserved | Perception must not import training, inference, or pipeline modules. |
| Protecting tests | Package import and SE(3) unit tests. |

### Downstream Phase 4 — Generative Grasp Model

| Item | Description |
| --- | --- |
| Dependency | Grasp pose representations will rely on stable SE(3) conventions. |
| Interface relied on | Homogeneous transform and rotation conventions. |
| New behavior after Phase 1 | SE(3) outputs are deterministic and compatible with `pytransform3d`. |
| Constraints preserved | No custom transform wrapper objects. |
| Protecting tests | SE(3) primitive tests. |

### Downstream Phase 5 — Reinforcement Learning Policy

| Item | Description |
| --- | --- |
| Dependency | Observation and action post-processing may later require SE(3) math. |
| Interface relied on | Pure math functions that can be called from policy-related code without side effects. |
| New behavior after Phase 1 | Math primitives do not mutate global state. |
| Constraints preserved | No global seed setting, no global device state. |
| Protecting tests | No-side-effect tests. |

### Downstream Phase 6 — End-to-End Orchestration & Eval

| Item | Description |
| --- | --- |
| Dependency | Pipeline orchestration will consume explicit configuration values. |
| Interface relied on | Static config contract for device, seed, and output directory. |
| New behavior after Phase 1 | Configuration keys are stable, but not globally loaded. |
| Constraints preserved | Pipelines must receive values explicitly. |
| Protecting tests | Configuration key-contract tests. |

---

## 17. Cross-Phase Contract

Phase 1 establishes the following contract.

### Inputs

1. Package layout:
   - Source code remains under `src`.
   - Package name remains `grasping_ai`.

2. Configuration:
   - `configs/base.yaml` exists.
   - It contains the documented keys:
     - `device`
     - `random_seed`
     - `output_dir`

3. SE(3) math:
   - Inputs are numeric arrays or values compatible with `pytransform3d`.
   - Homogeneous transforms are represented as 4x4 arrays where applicable.

### Outputs

1. Package import succeeds.
2. Math dependencies are importable.
3. SE(3) functions return plain numeric results.
4. No output depends on global mutable state.

### Expected behavior

1. Foundation code is side-effect free.
2. SE(3) functions are pure.
3. Configuration remains static.
4. No new utility abstraction is introduced.

### Invariants

1. The `src` layout remains unchanged.
2. The package name remains unchanged.
3. Existing dependencies are not removed without verification.
4. SE(3) transform math follows `pytransform3d` conventions.
5. No module-level mutable state is introduced.

### Error behavior

1. Invalid SE(3) input raises `TypeError` or `ValueError`.
2. Missing configuration files are detected by tests, not by import-time failure.
3. Missing dependencies are detected by dependency import tests.

### Configuration assumptions

1. Configuration values are not parsed globally in Phase 1.
2. Later phases must explicitly pass configuration values where needed.

### Data assumptions

1. No dataset is required.
2. No checkpoint is required.
3. No simulation asset is required.

### Performance assumptions

1. Phase 1 functions must be lightweight.
2. No heavy imports should be added to package initialization.
3. No GPU initialization should occur during Phase 1 tests.

---

## 18. Validation Before Commit

The implementation must not be considered ready for commit until the following sequence is complete.

1. Run the Phase 1 package import test.
2. Run the Phase 1 dependency availability test.
3. Run the Phase 1 configuration file contract tests.
4. Run the SE(3) primitive tests for every verified SE(3) function modified.
5. Run any existing tests that cover package imports or perception behavior.
6. Run the broader relevant test suite, if one exists.
7. Verify that no unrelated module was modified.
8. Verify that no new utility module was created.
9. Verify that no helper function was created.
10. Verify that no helper class was created.
11. Verify that no new class was created.
12. Verify that no global variable was created.
13. Verify that no global constant was created.
14. Verify that no module-level mutable state was created.
15. Verify that no file-level source description was added.
16. Verify that every modified or newly introduced function has a Google-style docstring.
17. Verify that no unrelated refactoring is included.
18. Verify that no new dependency was added.
19. Verify that the `src` layout is unchanged.
20. Verify that downstream phase contracts remain compatible.

The final commit gate is:

`Implementation -> Targeted Tests -> Regression Tests -> Integration Tests -> Full Relevant Suite -> Static/Structural Review -> Commit`

---

## 19. Commit Boundary

The Phase 1 commit must contain only the following.

### Allowed in the commit

1. Minimal modifications to existing SE(3) skeleton functions, if verified.
2. Minimal modifications to `src/grasping_ai/__init__.py` only if required to preserve importability.
3. Required Phase 1 tests.
4. Required test configuration if a test runner requires it and if such configuration already follows repository conventions.

### Not allowed in the commit

1. New utility modules.
2. New helper functions.
3. New helper classes.
4. New global constants.
5. New global variables.
6. New configuration loaders.
7. Unrelated formatting changes.
8. Refactoring of scripts unrelated to Phase 1.
9. Simulation code.
10. Data pipeline code.
11. Model code.
12. Training code.
13. Evaluation code.
14. Inference code.
15. Temporary debugging code.
16. Experimental code.

---

## 20. Implementation Risks

| Risk | Severity | Cause | Affected Component | Detection Method | Mitigation |
| --- | --- | --- | --- | --- | --- |
| Exact SE(3) source file is unverified. | High | Repository snapshot did not expose full `src/grasping_ai/perception` contents. | Perception math implementation. | Manual source inspection before implementation. | Do not create a new utility module. Pause if no existing skeleton function is found. |
| Accidental creation of utility abstraction. | High | Phase constraints forbid helpers, but math code often invites reuse. | Phase 1 source files. | Structural review before commit. | Implement behavior directly inside the existing skeleton function. |
| Introduction of global configuration state. | High | Configuration files exist but no loader is verified. | Package foundation. | Review of module-level code. | Keep configuration static and unparsed in Phase 1. |
| Adding YAML parser dependency. | Medium | Tests or config validation may tempt implementation to parse YAML. | `pyproject.toml`, tests. | Dependency diff review. | Use plain-text key checks for Phase 1 contract tests. |
| Breaking package import path. | High | Changing `src` layout or package init can break scripts. | `src/grasping_ai/__init__.py`, `pyproject.toml`. | Package import test. | Preserve existing layout and avoid heavy imports. |
| Incorrect SE(3) convention. | High | Different libraries use different transform conventions. | Downstream perception, robotics, and model phases. | Identity and composition tests. | Use `pytransform3d` conventions directly. |
| Tests become brittle due text-based config checks. | Low | YAML parsing is intentionally avoided. | Phase 1 tests. | Review of test assertions. | Keep checks limited to key presence and file existence. |

---

## 21. Design Decisions

| Decision | Evidence | Alternatives Considered | Reason for Selection |
| --- | --- | --- | --- |
| Preserve existing `hatchling` and `src` layout. | Verified `pyproject.toml`. | Move to flat layout or add package root. | Minimal change and preserves verified import path. |
| Do not add a YAML parser. | `pyproject.toml` does not verify a YAML dependency. | Add PyYAML or OmegaConf. | Avoids new dependency and avoids introducing config-loading scope. |
| Do not create a math utility module. | README places SE(3) helpers under perception; constraints forbid helper utilities. | Create `src/grasping_ai/utils/se3.py`. | Preserves documented architecture and phase constraints. |
| Use `pytransform3d` directly inside existing skeleton functions. | `pytransform3d` is already a dependency and README mentions SE(3) helpers. | Add custom math wrappers. | Avoids abstraction and uses existing dependency. |
| Keep configuration static. | No verified config loader exists. | Implement global config object. | Avoids global state and preserves downstream flexibility. |
| Mark SE(3) exact source units as unverified. | File contents under perception were not verified. | Guess file names and symbols. | Prevents invention and keeps design evidence-based. |

---

## 22. Explicitly Rejected Changes

The following changes were considered and rejected.

### Rejected: creating a helper or utility module

A shared math utility module would make SE(3) code reusable, but it violates the phase constraint against helper and utility abstractions. It would also conflict with the documented placement of SE(3) helpers under `perception`.

### Rejected: creating a configuration loader

A configuration loader would make YAML files easier to consume, but Phase 1 does not require runtime configuration parsing. Adding one would introduce global state risk and an unnecessary dependency.

### Rejected: adding PyYAML

YAML parsing is not required for Phase 1 validation. Configuration key presence can be checked conservatively without parsing. Adding PyYAML would expand the dependency surface unnecessarily.

### Rejected: creating new classes for transforms

Introducing transform classes would create object-oriented abstractions that are not required and are forbidden by the phase constraints. Arrays and existing `pytransform3d` functions are sufficient.

### Rejected: moving SE(3) helpers into `robotics`

The repository documentation places SE(3) helpers under perception. Moving them would reorganize the architecture without verified necessity.

### Rejected: implementing simulation, data, model, training, evaluation, or inference code

These belong to later phases. Implementing them now would break phase boundaries and increase regression risk.

### Rejected: adding global constants for tolerance or device names

Global constants are forbidden by the phase constraints. Required literal values should remain local to the function or test scope.

### Rejected: adding file-level source descriptions

File banners and module descriptions are forbidden by the phase constraints.

---

## 23. Verification Evidence

The following evidence supports the design.

| Claim | Evidence |
| --- | --- |
| Repository uses `hatchling` and `src` layout. | Verified `pyproject.toml` content. |
| Python requirement is `>=3.12`. | Verified `pyproject.toml` content. |
| Math dependencies already exist. | Verified `pyproject.toml` dependencies include `pytransform3d`, `scipy`, and `theseus`. |
| Perception contains SE(3) helpers. | README repository layout description. |
| Configuration files exist under `configs/`. | `docs/USAGE.md` configuration section. |
| Base configuration includes device, seed, and output directory. | `docs/USAGE.md` description of `configs/base.yaml`. |
| Scripts delegate to pipeline functions. | Verified `scripts/train_rl.py` content. |
| Repository is a skeleton. | README status section. |
| Exact perception source files are not verified. | Available snapshot did not expose file contents under `src/grasping_ai/perception`. |
| Exact simulation environment file is not verified. | Direct request for `src/grasping_ai/simulation/env.py` returned 404. |

---

## 24. Definition of Done

Phase 1 is considered complete only when:

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

- [ ] Verified the exact SE(3) skeleton file under `src/grasping_ai/perception/`.
- [ ] Verified the exact SE(3) skeleton function name.
- [ ] Implemented only the existing SE(3) skeleton function.
- [ ] Did not create a new utility module.
- [ ] Did not create helper functions.
- [ ] Did not create classes.
- [ ] Did not introduce global variables.
- [ ] Did not introduce global constants.
- [ ] Did not add file-level descriptions.
- [ ] Added Google-style docstrings to modified functions.
- [ ] Preserved `src` package layout.
- [ ] Preserved existing `pyproject.toml` dependency set.

### Tests

- [ ] Added package import test.
- [ ] Added math dependency availability test.
- [ ] Added base configuration file existence test.
- [ ] Added base configuration key-contract test.
- [ ] Added SE(3) identity behavior test.
- [ ] Added SE(3) invalid shape test.
- [ ] Added SE(3) no-side-effect test.
- [ ] Used only existing test framework conventions.
- [ ] Did not introduce YAML parsing dependency.
- [ ] Did not introduce network-dependent tests.

### Regression Verification

- [ ] Package import remains successful.
- [ ] Scripts that import `grasping_ai` remain unaffected.
- [ ] Existing dependency set remains usable.
- [ ] No module-level side effects were introduced.
- [ ] No import cycles were introduced.

### Cross-Phase Verification

- [ ] Phase 2 can rely on stable SE(3) math behavior.
- [ ] Phase 3 can rely on perception remaining low-level.
- [ ] Phase 4 can rely on stable SE(3) representation.
- [ ] Phase 5 can rely on pure math functions.
- [ ] Phase 6 can rely on static configuration keys without global config state.

### Structural Constraints

- [ ] No new utility abstraction exists.
- [ ] No new configuration abstraction exists.
- [ ] No new dependency was added.
- [ ] No unrelated file was modified.
- [ ] No unrelated formatting was included.
- [ ] No experimental code was included.

### Commit Readiness

- [ ] Targeted Phase 1 tests pass.
- [ ] Regression tests pass.
- [ ] Broader relevant test suite passes.
- [ ] Static/structural review confirms forbidden constructs are absent.
- [ ] Commit contains only Phase 1 changes.