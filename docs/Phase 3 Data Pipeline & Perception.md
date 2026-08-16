# Phase 3 — Data Pipeline & Perception

> **Historical design record.** This document captures the Phase 3 plan from the skeleton era. Current architecture, CLIs, and contracts live in [architecture.md](architecture.md), [USAGE.md](USAGE.md), and the repository [README.md](../README.md).

## 1. Phase Objective

Phase 3 establishes the minimum data ingestion and point-cloud perception foundation required before supervised grasp-model training can be implemented.

The phase must deliver only the following capabilities:

1. A dataset indexing behavior that can convert a raw dataset location into an explicit JSON index.
2. A dataset loading behavior that can read the index and load point-cloud records.
3. A point-cloud perception preprocessing behavior that can validate and prepare point clouds for downstream model consumption.
4. A stable data contract for downstream training, evaluation, and inference phases.
5. Tests that prove data preparation, index loading, point-cloud loading, and perception preprocessing work without external datasets, global state, helper abstractions, or new classes.

This phase is intentionally conservative. The repository is a skeleton, and the exact source files inside `src/grasping_ai/data/` and `src/grasping_ai/perception/` were not fully verifiable in the available snapshot. Therefore, this design distinguishes verified facts from unverified repository details. Where an exact file path or symbol cannot be verified, it is marked `Unverified`.

The implementation must not create new classes, helper utilities, global state, or abstraction layers. If the repository does not already contain a skeleton data-loading function, perception function, or equivalent entry point, the phase must be paused for a scope decision rather than creating a new dataset class or utility module.

---

## 2. Verified Repository Context

The following facts were verified from the repository snapshot available during analysis.

### Verified packaging facts

- `pyproject.toml` exists.
- The build backend is `hatchling.build`.
- The wheel target uses `packages = ["src"]`.
- Python requirement is `>=3.12`.
- Runtime dependencies already include:
  - `open3d`
  - `torch`
  - `torchvision`
  - `pytransform3d`
  - `scipy`
  - `gymnasium`
  - `stable-baselines3`
  - `tensorboard`
  - `theseus`
  - `wandb`
- Development dependencies already include:
  - `pytest`
  - `pytest-cov`
  - `ruff`
  - `mypy`
  - `mujoco`

### Verified project documentation facts

- `README.md` states that the repository is a Python source-code skeleton and that function bodies raise `NotImplementedError`.
- `README.md` describes:
  - `src/grasping_ai/data/` as dataset loading and transforms.
  - `src/grasping_ai/perception/` as point-cloud preprocessing and SE(3) helpers.
- `README.md` defines the supervised training flow as:

  `data -> perception -> models -> training -> inference`

- `docs/PROJECT.md` defines the dependency direction:

  `configs -> data -> perception -> models -> inference -> robotics -> simulation -> evaluation`

It also states that dataset loaders must not contain model or robot business logic.

- `docs/USAGE.md` describes:
  - `configs/data.yaml` as containing training dataset paths, YCB dataset paths, YCB object IDs, and dataset splits.
  - `scripts/prepare_data.py` as the script that builds the dataset index from raw records.
  - `scripts/train.py` as the script that trains a grasp-generation model using a processed dataset root.
  - Example observation files using the `.npy` extension.

### Verified script facts

- The repository script table lists:
  - `scripts/prepare_data.py`
  - `scripts/train.py`
  - `scripts/generate_grasps.py`
- `docs/USAGE.md` shows `scripts/prepare_data.py` receiving:
  - a raw dataset root
  - an output index path
- `docs/USAGE.md` shows `scripts/train.py` receiving:
  - a processed dataset root
  - model and training parameters

### Non-verified or partially verified areas

The following are not verified in the available snapshot:

- Exact file names under `src/grasping_ai/data/`.
- Exact file names under `src/grasping_ai/perception/`.
- Exact function names, class names, and signatures inside those directories.
- Existing test files, fixtures, and test conventions.
- Exact YAML key names inside `configs/data.yaml`.
- Exact raw dataset record layout.
- Exact index schema expected by downstream training code.
- Whether existing data-loading skeleton units are functions or classes.

Because of this, any source-unit design inside those directories is marked `Unverified` where exact paths or symbols cannot be confirmed.

---

## 3. Scope

Phase 3 includes only the following work.

### In scope

1. Runtime dependency correction for direct numeric array handling.
   - The data pipeline must handle `.npy` point-cloud arrays.
   - Direct NumPy use is required for `.npy` loading and point-cloud validation.
   - NumPy must be declared as a runtime dependency if direct NumPy imports are used.

2. Implementation of the existing data-preparation skeleton.
   - Build an explicit JSON index from a raw dataset root.
   - Validate raw dataset existence.
   - Fail explicitly for empty or invalid datasets.
   - Write the output index to the requested path.

3. Implementation of the existing dataset loading skeleton.
   - Load the JSON index.
   - Validate index structure.
   - Load point-cloud records.
   - Optionally load associated target arrays if the existing skeleton contract includes them.
   - Return data in a format suitable for downstream batching.

4. Implementation of the existing point-cloud perception preprocessing skeleton.
   - Validate point-cloud shape.
   - Validate finite values.
   - Preserve the source coordinate frame by default.
   - Optionally apply only preprocessing behavior already declared by the skeleton.
   - Return a deterministic numeric representation.

5. Preservation of static configuration contracts.
   - Do not remove `configs/data.yaml`.
   - Do not introduce global configuration loading.
   - Do not introduce module-level configuration state.

6. Tests for the Phase 3 contract.
   - Data preparation tests using temporary local records.
   - Index loading tests.
   - Point-cloud loading tests.
   - Perception validation tests.
   - Regression tests for Phase 1 and Phase 2 package behavior.

### Conditional scope

The data and perception implementation is conditional on locating existing skeleton units.

If no existing data-preparation function or script entry exists, Phase 3 must not create a new utility module. The correct action is to stop and request a scope decision.

If no existing dataset-loading function or method exists, Phase 3 must not create a new dataset class. The correct action is to stop and request a scope decision.

If no existing point-cloud preprocessing function or method exists, Phase 3 must not create a new perception utility module. The correct action is to stop and request a scope decision.

---

## 4. Out of Scope

The following are explicitly out of scope for Phase 3.

1. Model implementation.
   - Equivariant encoder.
   - Diffusion model.
   - Flow model.
   - RL policy network.
   - Grasp pose decoder.

2. Training pipeline implementation.
   - Supervised training loop.
   - Optimizer setup.
   - Learning-rate scheduling.
   - Checkpoint saving.
   - Experiment logging.

3. Inference pipeline implementation.
   - Grasp generation.
   - Model checkpoint loading.
   - Observation-to-grasp sampling.
   - Action post-processing.

4. Simulation behavior.
   - MuJoCo environment.
   - YCB object loading in simulation.
   - Physics stepping.
   - Grasp execution.

5. Evaluation behavior.
   - Force-closure evaluation.
   - Collision evaluation.
   - Lift-success evaluation.
   - Report generation.

6. Advanced dataset behavior.
   - Dataset downloading.
   - Dataset caching.
   - Dataset versioning.
   - Dataset registry.
   - Automatic train/validation/test splitting unless already declared by the existing skeleton.
   - Data augmentation unless already declared by the existing skeleton.

7. Advanced perception behavior.
   - Segmentation.
   - Detection.
   - Pose estimation.
   - Surface normal estimation unless already required by the existing skeleton.
   - Coordinate frame normalization unless already required by the existing skeleton.
   - Sensor calibration.

8. New architecture.
   - New dataset classes.
   - New helper functions.
   - New utility modules.
   - New global configuration managers.
   - New plugin systems.

---

## 5. Existing Architecture and Patterns

Phase 3 must preserve the following verified or documented repository patterns.

### Pattern 1: Thin CLI scripts delegate to pipeline functions

The repository documentation describes scripts as thin CLI entry points. `scripts/train_rl.py` was verified as a thin script that parses arguments and calls a pipeline function.

Phase 3 must not move data or perception business logic into scripts if an existing pipeline function is the intended target. If `scripts/prepare_data.py` delegates to an existing pipeline function, the implementation must occur in that pipeline function. The script should remain thin.

### Pattern 2: Skeleton functions raise `NotImplementedError`

The README states that the repository contains skeleton code with `NotImplementedError` bodies. Phase 3 should replace `NotImplementedError` only in existing functions, methods, or module execution blocks that belong to the phase scope.

Phase 3 must not create new reusable helper functions to avoid implementing required behavior directly in the appropriate existing skeleton unit.

### Pattern 3: Lower-level modules do not depend on orchestration

The documented dependency direction places `data` and `perception` below models, inference, robotics, simulation, and evaluation.

Phase 3 modules must not import pipeline functions, training loops, inference loaders, model definitions, simulation modules, or CLI scripts.

### Pattern 4: Dataset loaders do not contain model or robot business logic

`docs/PROJECT.md` explicitly states that dataset loaders must not contain model or robot business logic.

Phase 3 data loading must only load and validate data. It must not transform data into model-specific latent representations, grasp poses, robot commands, or simulation states.

### Pattern 5: Configuration remains static

The repository uses YAML configuration files, but no global configuration loader was verified.

Phase 3 must not introduce a global configuration object, module-level config cache, or singleton configuration manager. Dataset roots, index paths, and preprocessing options should be accepted explicitly by the existing skeleton interface or by the pipeline that later calls it.

---

## 6. Implementation Dependencies

Phase 3 requires the following dependencies.

### Existing dependencies that must remain available

| Dependency | Reason |
| --- | --- |
| `torch` | Numeric tensor representation for downstream model input. |
| `open3d` | Point-cloud processing behavior described by the repository. |
| `pytransform3d` | Existing spatial math dependency; used only if already required by Phase 1 or existing perception skeleton. |
| `pytest` | Development dependency for Phase 3 tests. |

### Dependency configuration change required

| Dependency | Current verified status | Required Phase 3 treatment |
| --- | --- | --- |
| `numpy` | Not explicitly listed in the verified runtime dependencies, but required for `.npy` loading and direct numeric array validation. | Add NumPy as a runtime dependency if Phase 3 implementation imports it directly. |

The NumPy version constraint must be chosen conservatively and must not break the already declared Open3D, SciPy, and Torch dependency constraints. If the exact resolved NumPy version cannot be verified, the version constraint is marked `Unverified` and must be resolved during implementation by dependency resolution testing.

No other new third-party dependency may be introduced.

---

## 7. Source-Unit Change Matrix

Rows marked `Unverified` are required phase targets, but the exact file path or symbol must be resolved from the repository before implementation. They are not speculative new files.

| File | Symbol | Change Type | Current Responsibility | Proposed Change | Risk | Required Tests |
| --- | --- | --- | --- | --- | --- | --- |
| `pyproject.toml` | `project.dependencies` | Extend | Declares runtime dependencies. | Add NumPy as a runtime dependency if direct NumPy imports are used. | Medium | Dependency import test and dependency resolution regression test. |
| `configs/data.yaml` | Documented dataset paths, YCB paths, object IDs, and splits | Configuration preservation | Declares data configuration semantics according to documentation. | Preserve file and documented semantics. Do not introduce global loading. If exact keys are verified, keep them unchanged. | Low | Configuration existence test. |
| `scripts/prepare_data.py` | Unverified existing CLI entry function or module-level execution block | Modify | Currently expected to raise `NotImplementedError` or delegate to an unimplemented pipeline function. | Implement or complete the data-preparation entry behavior while preserving the thin CLI pattern. | High | Data preparation tests. |
| Unverified existing pipeline module imported by `scripts/prepare_data.py` | Unverified existing data-preparation pipeline function | Modify | Currently expected to raise `NotImplementedError`. | Implement index construction from explicit dataset root and output index arguments. | High | Data preparation tests. |
| Unverified existing module under `src/grasping_ai/data/` | Unverified existing index-loading function or method | Modify | Currently expected to raise `NotImplementedError`. | Implement JSON index loading and validation. | High | Index loading tests. |
| Unverified existing module under `src/grasping_ai/data/` | Unverified existing point-cloud record loading function or method | Modify | Currently expected to raise `NotImplementedError`. | Implement point-cloud loading from explicit record paths. | High | Point-cloud loading tests. |
| Unverified existing module under `src/grasping_ai/perception/` | Unverified existing point-cloud preprocessing function or method | Modify | Currently expected to raise `NotImplementedError`. | Implement point-cloud validation and deterministic preprocessing behavior. | High | Perception validation tests. |
| Unverified existing module under `src/grasping_ai/data/` | Unverified existing batching or collation function or method | Modify | Currently expected to raise `NotImplementedError`. | Implement minimal fixed-shape batching behavior if the skeleton already defines such a unit. | Medium | Batching tests. |
| Existing or minimum required test file under `tests/unit/` | Phase 3 data and perception tests | Test-only change | No verified existing Phase 3 tests. | Add tests for data preparation, index loading, point-cloud loading, perception preprocessing, and regression behavior. | Medium | All Phase 3 tests pass. |

---

## 8. Detailed Source-Unit Design

### `pyproject.toml::project.dependencies`

#### Current behavior

The verified `pyproject.toml` declares runtime dependencies for the project. NumPy is not explicitly listed in the verified runtime dependency list, although it is required by the documented `.npy` data format and is normally present as a transitive dependency of Open3D, SciPy, and Torch.

#### Required behavior

Phase 3 loads `.npy` arrays and validates numeric point-cloud contents. If the implementation imports NumPy directly, NumPy must be declared as a runtime dependency.

#### Design

- Input: none.
- Output: none.
- Control flow: none.
- State changes: none.
- Existing dependencies: Python packaging metadata.
- Error behavior: packaging errors should surface during dependency resolution.
- Edge cases: version constraint must remain compatible with Open3D, SciPy, and Torch.
- Interaction with surrounding code: data loading and perception tests will import NumPy.
- Existing pattern being followed: preserve declarative dependency management in `pyproject.toml`.

The change should be limited to dependency declaration. It must not introduce unrelated dependency upgrades.

---

### `configs/data.yaml`

#### Current behavior

Documentation states that this file contains training dataset paths, YCB dataset paths, YCB object IDs, and dataset splits. The exact key names were not verified.

#### Required behavior

Phase 3 must preserve the file and its documented semantics. Phase 3 must not introduce global configuration loading. If dataset roots or splits are needed by Phase 3 behavior, they should be accepted explicitly through the existing skeleton interface or through a later pipeline.

#### Design

- Input: none.
- Output: none.
- Control flow: none.
- State changes: none.
- Existing dependencies: none.
- Error behavior: not applicable during Phase 3 because no runtime YAML parsing is introduced.
- Edge cases: missing keys should be handled by later pipeline validation if that pipeline chooses to consume the file.
- Interaction with surrounding code: later orchestration may read this file, but Phase 3 does not require it.
- Existing pattern being followed: static configuration without global state.

No YAML key may be renamed, removed, or added unless exact key verification proves that a documented key is missing.

---

### `scripts/prepare_data.py`

#### Current behavior

The script exists and is documented as the entry point for building the dataset index from raw records. The exact function or module-level execution block was not verified. Because the repository is a skeleton, the current behavior is expected to be `NotImplementedError` or delegation to an unimplemented pipeline function.

#### Required behavior

The script must support the documented data-preparation entry point. It must accept explicit input and output locations and invoke the existing data-preparation behavior.

The script must remain thin. It must not contain dataset discovery, validation, or index-writing logic if an existing pipeline function is the intended target.

#### Design

##### Input

Expected inputs include:

- Raw dataset root.
- Output index path.

Additional inputs may exist in the skeleton. The implementation must preserve the existing CLI contract.

##### Output

The script itself produces no meaningful return value. Its side effect is the creation of an index file at the requested output path.

##### Control flow

1. Parse CLI arguments using the existing argument parsing behavior.
2. Call the existing data-preparation pipeline or data function.
3. Propagate errors naturally.

##### State changes

No module-level state.

##### Existing dependencies

- Existing pipeline or data module.
- Standard argument parsing if already used by the script.

##### Error behavior

- Missing raw dataset root raises a filesystem or value error.
- Invalid output index path raises a filesystem or value error.
- Empty dataset raises a value error.

##### Edge cases

- Dataset root exists but contains no valid records.
- Output index parent directory does not exist.
- Output index path points to a directory.

##### Interaction with surrounding code

Later training behavior may depend on the index produced by this entry point.

##### Existing pattern being followed

Thin CLI script delegates to lower-level behavior.

---

### Unverified existing data-preparation pipeline function

#### Current behavior

The exact pipeline symbol was not verified. The current behavior is expected to be `NotImplementedError`.

#### Required behavior

The data-preparation function must build a JSON index from a raw dataset root.

#### Design

##### Input

Expected inputs include:

- Raw dataset root path.
- Output index path.
- Optional record filters if already present in the skeleton.

The implementation must follow the existing skeleton signature.

##### Output

The function may return the number of records written, the resolved output path, or nothing if the existing skeleton contract defines no return value.

##### Control flow

1. Validate that the raw dataset root exists.
2. Discover raw records according to the existing skeleton contract.
3. Validate discovered records.
4. Build a JSON-serializable index structure.
5. Create the output index parent directory if necessary.
6. Write the JSON index.

##### State changes

Filesystem side effect: writes the output index.

No module-level state.

##### Existing dependencies

- Standard filesystem utilities.
- Standard JSON serialization.
- Existing data record contract if present.

##### Error behavior

- Missing dataset root raises a filesystem error.
- Empty dataset raises a value error.
- Invalid record paths raise value errors.
- Non-serializable record metadata raises a type or value error.

##### Edge cases

- Empty subdirectories.
- Unsupported file extensions.
- Duplicate record identifiers if the skeleton defines identifiers.
- Output index overwriting an existing file.

##### Interaction with surrounding code

The produced index is consumed by dataset loading behavior.

##### Existing pattern being followed

Pipeline or data function performs orchestration, while lower-level loaders remain separate.

---

### Unverified existing index-loading function or method

#### Current behavior

The exact symbol was not verified. The current behavior is expected to be `NotImplementedError`.

#### Required behavior

The index-loading unit must read a JSON index and return explicit record entries that can be consumed by point-cloud loading behavior.

#### Design

##### Input

Expected input is an index path or dataset root containing an index path, depending on the existing skeleton contract.

##### Output

The output must be a JSON-derived structure containing explicit records.

A minimal stable record contract is:

- A point-cloud path.
- Optional target path if supervised targets are part of the existing skeleton.
- Optional object identifier.
- Optional split identifier.

If the existing skeleton already defines field names, those field names must be preserved. If no field names are verified, the implementation must define the minimal contract explicitly and record it as the Phase 3 cross-phase contract.

##### Control flow

1. Read the JSON file.
2. Validate top-level structure.
3. Validate each record entry.
4. Return the parsed records.

##### State changes

None.

##### Existing dependencies

- Standard JSON parsing.
- Standard filesystem utilities.

##### Error behavior

- Missing index file raises a filesystem error.
- Invalid JSON raises a value error.
- Missing required record fields raises a value error.
- Non-string path fields raises a type error.

##### Edge cases

- Empty record list.
- Relative paths.
- Duplicate records.
- Unknown optional fields.

##### Interaction with surrounding code

Dataset loading and training pipelines consume the returned records.

##### Existing pattern being followed

Data loading remains low-level and does not contain model logic.

---

### Unverified existing point-cloud record loading function or method

#### Current behavior

The exact symbol was not verified. The current behavior is expected to be `NotImplementedError`.

#### Required behavior

The point-cloud loading unit must load a point-cloud array from an explicit record path and return it in a numeric format suitable for perception preprocessing.

#### Design

##### Input

Expected input is a record entry or explicit point-cloud path.

##### Output

The output must be a numeric point-cloud array or tensor with shape `N x 3`.

If associated target data is part of the existing skeleton contract, the output may include the target array. The target array must be returned pass-through unless the existing skeleton already defines interpretation behavior.

##### Control flow

1. Resolve the point-cloud path.
2. Load the `.npy` array.
3. Validate that the array has two dimensions and three coordinate channels.
4. Validate that values are finite.
5. Return the point cloud and optional target data.

##### State changes

None.

##### Existing dependencies

- NumPy for `.npy` loading.
- Standard filesystem utilities.

##### Error behavior

- Missing file raises a filesystem error.
- Invalid array shape raises a value error.
- Non-finite values raise a value error.
- Unsupported file format raises a value or type error.

##### Edge cases

- Empty point cloud with zero points.
- Single-point point cloud.
- Very large point clouds.
- Relative paths.

##### Interaction with surrounding code

Perception preprocessing consumes the loaded point cloud.

##### Existing pattern being followed

Dataset loaders load data but do not interpret model semantics.

---

### Unverified existing point-cloud preprocessing function or method

#### Current behavior

The exact symbol was not verified. The current behavior is expected to be `NotImplementedError`.

#### Required behavior

The perception preprocessing unit must validate a point cloud and prepare it for downstream consumption without introducing hidden coordinate-frame changes.

#### Design

##### Input

Expected input is a numeric point-cloud array with shape `N x 3`.

Optional explicit preprocessing arguments may exist in the skeleton. The implementation must preserve the existing skeleton signature.

##### Output

The output must be a deterministic point-cloud representation compatible with downstream model input.

The preferred output representation is a Torch tensor with floating-point dtype, unless the existing skeleton contract defines another representation.

##### Control flow

1. Validate input type and shape.
2. Validate that all coordinate values are finite.
3. Preserve the source coordinate frame by default.
4. Apply only preprocessing behavior already declared by the skeleton, such as dtype conversion or explicit downsampling.
5. Return the preprocessed point cloud.

##### State changes

None.

##### Existing dependencies

- NumPy for numeric validation.
- Torch for tensor conversion.
- Open3D only if the existing skeleton already requires Open3D preprocessing behavior.

##### Error behavior

- Invalid shape raises a value error.
- Non-finite values raise a value error.
- Unsupported dtype raises a type error.

##### Edge cases

- Empty point cloud.
- Single-point point cloud.
- Duplicate points.
- Very large point clouds.
- Preprocessing on CPU-only systems.

##### Interaction with surrounding code

Downstream model training and inference consume the preprocessed point cloud.

##### Existing pattern being followed

Perception remains low-level and does not depend on models, training, or pipelines.

---

### Unverified existing batching or collation function or method

#### Current behavior

The exact symbol was not verified. The current behavior is expected to be `NotImplementedError`.

#### Required behavior

If the existing skeleton defines batching or collation behavior, Phase 3 must implement the minimum behavior needed to combine fixed-shape point-cloud samples.

#### Design

##### Input

Expected input is a sequence of loaded or preprocessed samples.

##### Output

The output must be a batched representation compatible with downstream training.

For Phase 3, batching is restricted to fixed-shape samples. Variable-shape batching is out of scope unless already declared by the skeleton.

##### Control flow

1. Validate that the sequence is non-empty.
2. Validate that all point-cloud samples have the same shape.
3. Validate that optional target samples have consistent shapes if present.
4. Combine samples into a batched numeric representation.
5. Return the batch.

##### State changes

None.

##### Existing dependencies

- Torch for tensor batching.
- NumPy if array conversion is required by the skeleton.

##### Error behavior

- Empty sample sequence raises a value error.
- Mismatched point-cloud shapes raise a value error.
- Mismatched target shapes raise a value error.

##### Edge cases

- Batch size one.
- Samples with no targets.
- Samples with optional metadata.

##### Interaction with surrounding code

Training pipelines consume batches.

##### Existing pattern being followed

Data loading and batching remain separate from model logic.

---

## 9. Data Flow

Phase 3 introduces the following data flows.

### Dataset preparation data flow

`Raw dataset root -> Existing data-preparation entry -> Record discovery and validation -> JSON index file`

Input source:

- CLI caller or pipeline caller provides raw dataset root and output index path.

Transformation:

- Raw records are discovered.
- Records are validated.
- A JSON-serializable index is produced.

Validation:

- Dataset root existence.
- Record presence.
- Record path validity.

Output:

- JSON index file.

Consumers:

- Dataset loading behavior.
- Training pipeline in a later phase.

### Dataset loading data flow

`JSON index -> Existing index-loading unit -> Record list -> Existing point-cloud loading unit -> Point cloud and optional target`

Input source:

- Processed dataset root or explicit index path.

Transformation:

- JSON index is parsed.
- Each record is resolved.
- Point-cloud arrays are loaded.

Validation:

- Index structure.
- Point-cloud shape.
- Finite coordinate values.

Output:

- Point-cloud arrays or tensors.
- Optional target arrays.

Consumers:

- Perception preprocessing.
- Batching.
- Later training pipeline.

### Perception preprocessing data flow

`Loaded point cloud -> Existing perception preprocessing unit -> Validated preprocessed point cloud`

Input source:

- Dataset loader or direct caller.

Transformation:

- Shape validation.
- Finite-value validation.
- Optional dtype conversion.
- Optional skeleton-declared preprocessing.

Validation:

- `N x 3` shape.
- Finite values.
- Supported numeric type.

Output:

- Deterministic point-cloud representation.

Consumers:

- Batching.
- Model training in Phase 4.
- Inference observation processing in later phases.

---

## 10. Execution Flow

Phase 3 does not create a new CLI entry point beyond the existing `scripts/prepare_data.py`.

### Verified preparation entry path

`scripts/prepare_data.py -> Unverified existing data-preparation pipeline or data function -> JSON index file`

Phase 3 implements the missing behavior in the appropriate existing unit. It does not implement training.

### Verified downstream training path

`scripts/train.py -> Unverified existing training pipeline -> Phase 3 dataset loading and perception behavior -> Phase 4 model behavior`

Phase 3 does not implement `scripts/train.py` or the training pipeline. It provides the data and perception behavior that later training will consume.

### Direct validation path

For Phase 3 validation, tests may directly call the located data and perception skeleton units. This avoids depending on unimplemented training behavior.

### Error propagation

Errors must propagate naturally:

- Missing files propagate filesystem errors.
- Invalid JSON propagates value errors.
- Invalid point-cloud shapes propagate value errors.
- Non-finite point-cloud values propagate value errors.

Phase 3 must not swallow errors silently.

---

## 11. Configuration Changes

Phase 3 requires one packaging configuration change and no YAML configuration additions.

### Packaging configuration change

| File | Key | Existing value | Required change | Reason | Consumed by | Tests |
| --- | --- | --- | --- | --- | --- | --- |
| `pyproject.toml` | `project.dependencies` | NumPy is not explicitly listed in the verified runtime dependencies. | Add NumPy if direct NumPy imports are used. | Phase 3 loads `.npy` arrays and validates numeric point clouds. | Data loading, perception, tests. | Dependency import test. |

### YAML configuration changes

No YAML configuration additions are required.

The following file must remain available:

- `configs/data.yaml`

Phase 3 must not:

- Add new YAML keys.
- Rename existing YAML keys.
- Introduce a YAML parser.
- Introduce global configuration state.
- Read configuration at module import time.

If later pipeline code needs values from `configs/data.yaml`, that code must receive explicit values or implement its own explicit configuration handling in a later phase.

---

## 12. Test Strategy

Testing must use `pytest`, which is already a development dependency.

No existing test suite was verified. Therefore, the implementation engineer must first check whether a test directory and existing conventions are present.

If tests already exist, Phase 3 tests must follow the existing conventions for:

- File naming.
- Test function naming.
- Fixture usage.
- Assertion style.
- Parameterization.
- Mocking, if already used.

If no tests exist, Phase 3 may add the minimum required test file under a conventional unit-test location. This is a test-only addition.

### Testing principles

1. Tests must be deterministic.
2. Tests must not require network access.
3. Tests must not require downloaded datasets.
4. Tests must not require YCB assets.
5. Tests must not require GPUs.
6. Tests must not require model checkpoints.
7. Tests must not require simulation.
8. Tests must not introduce global state.
9. Tests must not create helper functions or helper classes.
10. Tests must not depend on visualization.

### Asset strategy

Because raw dataset formats are not fully verified, tests must use the smallest possible local dataset contract.

The preferred approach is:

- Create a temporary dataset root.
- Create minimal point-cloud arrays inside the test scope.
- Create optional target arrays only if the existing skeleton contract requires them.
- Avoid committing binary assets.
- Avoid creating shared helper fixture code unless an existing fixture already provides the same behavior.

If the existing skeleton requires a specific raw dataset layout and cannot operate with a minimal temporary layout, then Phase 3 validation is blocked until a verified test asset contract is available.

---

## 13. Test Suite

The following tests are required. Rows that depend on unverified skeleton symbols are marked accordingly.

| Test | File | Target | Scenario | Expected Result | Regression Risk |
| --- | --- | --- | --- | --- | --- |
| `test_numpy_runtime_dependency_available` | Existing or minimum required Phase 3 unit test file | `pyproject.toml` runtime dependency | Import NumPy. | Import succeeds. | Medium if dependency configuration is incorrect. |
| `test_phase1_package_import_remains_stable` | Existing or minimum required Phase 3 unit test file | `src/grasping_ai/__init__.py` | Import `grasping_ai`. | Import succeeds. | High if Phase 3 changes package initialization. |
| `test_data_config_file_exists` | Existing or minimum required Phase 3 unit test file | `configs/data.yaml` | Check file existence. | File exists. | Low. |
| `test_prepare_data_creates_index_from_minimal_dataset` | Existing or minimum required Phase 3 unit test file | Unverified data-preparation unit | Prepare a temporary dataset with one valid point-cloud record. | Output index exists and contains one record. | High. |
| `test_prepare_data_rejects_missing_dataset_root` | Existing or minimum required Phase 3 unit test file | Unverified data-preparation unit | Provide a missing dataset root. | Raises filesystem or value error. | Medium. |
| `test_prepare_data_rejects_empty_dataset` | Existing or minimum required Phase 3 unit test file | Unverified data-preparation unit | Provide an empty dataset root. | Raises value error. | Medium. |
| `test_index_loader_reads_prepared_index` | Existing or minimum required Phase 3 unit test file | Unverified index-loading unit | Load the index produced by preparation. | Returns parsed records. | High. |
| `test_index_loader_rejects_invalid_structure` | Existing or minimum required Phase 3 unit test file | Unverified index-loading unit | Provide malformed JSON or missing required fields. | Raises value error. | Medium. |
| `test_point_cloud_loader_reads_valid_npy_point_cloud` | Existing or minimum required Phase 3 unit test file | Unverified point-cloud loading unit | Load a valid `N x 3` point-cloud array. | Returns point cloud with expected shape. | High. |
| `test_point_cloud_loader_rejects_wrong_shape` | Existing or minimum required Phase 3 unit test file | Unverified point-cloud loading unit | Load an array with invalid shape. | Raises value error. | Medium. |
| `test_point_cloud_loader_rejects_non_finite_values` | Existing or minimum required Phase 3 unit test file | Unverified point-cloud loading unit | Load a point cloud containing NaN or infinite values. | Raises value error. | Medium. |
| `test_perception_preprocess_preserves_valid_point_cloud_shape` | Existing or minimum required Phase 3 unit test file | Unverified perception preprocessing unit | Preprocess a valid point cloud. | Output shape remains `N x 3`. | High. |
| `test_perception_preprocess_rejects_invalid_shape` | Existing or minimum required Phase 3 unit test file | Unverified perception preprocessing unit | Provide invalid shape. | Raises value error. | Medium. |
| `test_perception_preprocess_rejects_non_finite_values` | Existing or minimum required Phase 3 unit test file | Unverified perception preprocessing unit | Provide point cloud with NaN or infinite values. | Raises value error. | Medium. |
| `test_perception_preprocess_is_deterministic` | Existing or minimum required Phase 3 unit test file | Unverified perception preprocessing unit | Preprocess the same input twice. | Outputs are identical. | Medium. |
| `test_batch_collate_combines_fixed_shape_samples` | Existing or minimum required Phase 3 unit test file | Unverified batching unit | Collate samples with identical shape. | Returns batch with expected leading batch dimension. | Medium. |
| `test_batch_collate_rejects_mismatched_shapes` | Existing or minimum required Phase 3 unit test file | Unverified batching unit | Collate samples with different point counts. | Raises value error. | Medium. |
| `test_data_functions_do_not_leak_global_state` | Existing or minimum required Phase 3 unit test file | All Phase 3 modified units | Call units repeatedly with different inputs. | Outputs depend only on inputs. | Medium. |

---

## 14. Regression Test Plan

Phase 3 can introduce regressions in packaging, imports, configuration, and downstream training expectations.

### Unit-level regressions

| Existing behavior | Why it could break | Protecting test | Expected result |
| --- | --- | --- | --- |
| Package import succeeds. | Data or perception modules may introduce import cycles or heavy imports. | `test_phase1_package_import_remains_stable` | Import succeeds. |
| Phase 1 math dependencies remain usable. | Dependency addition may affect resolution. | Phase 1 dependency tests and Phase 3 dependency test. | Imports succeed. |
| Phase 2 simulation imports remain usable. | Dependency changes may affect environment resolution. | Phase 2 dependency and simulation tests, if present. | Existing Phase 2 behavior remains intact. |

### Integration-level regressions

| Existing behavior | Why it could break | Protecting test | Expected result |
| --- | --- | --- | --- |
| Scripts can import pipeline modules. | Data preparation may introduce pipeline import errors. | Existing import-related tests, if present. | Import path remains valid. |
| Downstream training can expect a dataset root and index. | Index schema or loader contract may become unstable. | Index loading and point-cloud loading tests. | Contract remains stable. |

### Pipeline-level regressions

No full training pipeline behavior is verified yet. Therefore, no pipeline-level regression test can be specified without invention.

If an existing pipeline smoke test exists, it must be rerun. If no such test exists, Phase 3 must not create a broad pipeline smoke test.

---

## 15. Impact on Preceding Phases

### Phase 1 — Foundation & Math Primitives

| Item | Description |
| --- | --- |
| Existing contract | Phase 1 established package importability, dependency availability, static configuration preservation, and pure SE(3) math behavior where verified. |
| What Phase 3 changes | Phase 3 adds data loading and perception behavior and may add NumPy as a runtime dependency. |
| What remains compatible | Package layout, existing dependencies, static configuration files, and Phase 1 math behavior must remain unchanged. |
| Previous-phase tests to rerun | Phase 1 package import test, dependency tests, SE(3) math tests if present, and configuration existence tests if present. |
| Whether Phase 1 needs modification | No modification is expected unless dependency resolution fails. |

### Phase 2 — Simulation & Robotics Core

| Item | Description |
| --- | --- |
| Existing contract | Phase 2 established simulation and robotics behavior where verified. |
| What Phase 3 changes | Phase 3 does not modify simulation or robotics behavior directly. |
| What remains compatible | Simulation and robotics modules must remain independent of data loading and perception. |
| Previous-phase tests to rerun | Phase 2 dependency and simulation tests if present. |
| Whether Phase 2 needs modification | No modification is expected unless a shared dependency conflict appears. |

---

## 16. Impact on Downstream Phases

### Downstream Phase 4 — Generative Grasp Model

| Item | Description |
| --- | --- |
| Dependency | Phase 4 depends heavily on Phase 3. |
| Interface relied on | Dataset index, point-cloud loading, preprocessing output, and batching behavior. |
| New behavior available after Phase 3 | A supervised training pipeline can later consume preprocessed point clouds. |
| Constraints preserved | Data loading must remain separate from model logic. Coordinate frame must not be silently altered. |
| Protecting tests | Point-cloud loading, perception preprocessing, and batching tests. |

### Downstream Phase 5 — Reinforcement Learning Policy

| Item | Description |
| --- | --- |
| Dependency | Phase 5 does not strictly depend on Phase 3 for environment stepping, but observation processing may later reuse perception validation. |
| Interface relied on | Stable point-cloud validation behavior if observation inputs use point clouds. |
| New behavior available after Phase 3 | Point-cloud observations can be validated consistently. |
| Constraints preserved | Perception must not depend on RL policy code. |
| Protecting tests | Perception validation tests. |

### Downstream Phase 6 — End-to-End Orchestration & Evaluation

| Item | Description |
| --- | --- |
| Dependency | Phase 6 may need point-cloud inputs for grasp generation and evaluation. |
| Interface relied on | Stable dataset and perception contracts. |
| New behavior available after Phase 3 | End-to-end pipelines can later load observations from processed datasets. |
| Constraints preserved | Data and perception must remain low-level and explicit. |
| Protecting tests | Index loading and perception preprocessing tests. |

---

## 17. Cross-Phase Contract

Phase 3 establishes the following contract.

### Inputs

1. Dataset preparation:
   - Explicit raw dataset root.
   - Explicit output index path.

2. Index loading:
   - Explicit index path or dataset root containing an index, according to the existing skeleton contract.

3. Point-cloud loading:
   - Explicit point-cloud path from an index record.
   - Optional target path if the existing skeleton contract includes supervised targets.

4. Perception preprocessing:
   - Numeric point-cloud array with shape `N x 3`.
   - Optional explicit preprocessing arguments if present in the existing skeleton.

### Outputs

1. Dataset preparation:
   - JSON index file.

2. Index loading:
   - Parsed record list.

3. Point-cloud loading:
   - Point-cloud array or tensor with shape `N x 3`.
   - Optional target array if present.

4. Perception preprocessing:
   - Validated point-cloud representation.
   - Preferred output is a floating-point Torch tensor.

5. Batching:
   - Batched fixed-shape samples.

### Expected behavior

1. Dataset preparation is deterministic for a fixed raw dataset layout.
2. Index loading does not mutate the index file.
3. Point-cloud loading does not mutate loaded arrays.
4. Perception preprocessing does not silently change coordinate frame.
5. Batching fails explicitly for mismatched shapes.

### Invariants

1. No module-level mutable state.
2. No global configuration state.
3. No dependency on models, training, inference, simulation, or robotics.
4. Point-cloud coordinate channels remain three-dimensional.
5. Point-cloud values remain finite after loading and preprocessing.
6. Existing configuration files remain present.

### Error behavior

1. Missing required files raise explicit filesystem or value errors.
2. Invalid JSON raises value errors.
3. Invalid point-cloud shape raises value errors.
4. Non-finite point-cloud values raise value errors.
5. Mismatched batch shapes raise value errors.

### Configuration assumptions

1. YAML files remain static.
2. Phase 3 does not parse YAML globally.
3. Later pipelines may consume configuration explicitly.

### Data assumptions

1. Point clouds are stored as numeric arrays with shape `N x 3`.
2. The documented `.npy` format is supported.
3. Coordinates are expressed in the frame provided by the dataset.
4. No unit conversion is performed unless explicitly declared by the existing skeleton.

### Performance assumptions

1. Single-process loading is sufficient for Phase 3.
2. Parallel data loading is out of scope.
3. Large dataset caching is out of scope.
4. Tests must avoid large point clouds.

---

## 18. Validation Before Commit

The implementation must not be considered ready for commit until the following sequence is complete.

1. Run Phase 3 dependency tests.
2. Run Phase 3 package import regression tests.
3. Run Phase 3 configuration existence tests.
4. Run dataset preparation tests.
5. Run index loading tests.
6. Run point-cloud loading tests.
7. Run perception preprocessing tests.
8. Run batching tests if batching is part of the verified skeleton.
9. Run no-global-state tests.
10. Rerun Phase 1 tests.
11. Rerun Phase 2 tests if present.
12. Run the broader relevant test suite.
13. Verify that no unrelated module changed.
14. Verify that no new class was introduced.
15. Verify that no helper or utility function was introduced.
16. Verify that no global variable or constant was introduced.
17. Verify that no file-level source description was introduced.
18. Verify that every modified or newly introduced function or method has a Google-style docstring.
19. Verify that no unrelated refactoring is included.
20. Verify that no new dependency other than the NumPy runtime declaration was introduced.
21. Verify that downstream contracts remain compatible.

The final commit gate is:

`Implementation -> Targeted Tests -> Regression Tests -> Integration Tests -> Full Relevant Suite -> Static/Structural Review -> Commit`

---

## 19. Commit Boundary

The Phase 3 commit must contain only the following.

### Allowed in the commit

1. Runtime dependency declaration for NumPy if direct NumPy imports are used.
2. Modifications to verified existing data-preparation skeleton units.
3. Modifications to verified existing dataset loading skeleton units.
4. Modifications to verified existing perception preprocessing skeleton units.
5. Required Phase 3 tests.
6. Minimal test-only temporary data contracts used only inside tests.

### Not allowed in the commit

1. New dataset classes.
2. New helper functions.
3. New utility modules.
4. New global constants.
5. New global variables.
6. New configuration loaders.
7. Unrelated formatting changes.
8. Refactoring of scripts unrelated to Phase 3.
9. Model code.
10. Training loop code.
11. Inference code.
12. Simulation code.
13. Evaluation metric code.
14. Temporary debugging code.
15. Experimental code.

---

## 20. Implementation Risks

| Risk | Severity | Cause | Affected Component | Detection Method | Mitigation |
| --- | --- | --- | --- | --- | --- |
| Exact data skeleton file or symbol is unverified. | High | Repository snapshot did not expose source details under `src/grasping_ai/data/`. | Data pipeline implementation. | Manual source inspection before implementation. | Modify only located existing units. If absent, pause for scope decision. |
| Exact perception skeleton file or symbol is unverified. | High | Repository snapshot did not expose source details under `src/grasping_ai/perception/`. | Perception implementation. | Manual source inspection before implementation. | Modify only located existing units. If absent, pause for scope decision. |
| Raw dataset format is unverified. | High | Documentation does not define exact raw record layout. | Dataset preparation. | Source inspection and test dataset contract. | Follow existing skeleton contract. If absent, define minimal explicit contract and record it. |
| Index schema is unverified. | High | Documentation does not define exact JSON schema. | Dataset loading and downstream training. | Source inspection and index tests. | Preserve existing schema if found. Otherwise define minimal explicit contract. |
| Creating a new dataset class would violate constraints. | High | PyTorch workflows often use Dataset classes. | Data pipeline. | Structural review. | Use existing class if present. Otherwise use function-based loading or pause. |
| NumPy dependency addition affects resolution. | Medium | NumPy is currently transitive rather than explicitly declared. | Packaging. | Dependency resolution and import tests. | Use conservative version constraint and rerun dependency tests. |
| Coordinate-frame mismatch. | High | Preprocessing may center, scale, or transform points. | Downstream grasp generation and evaluation. | Perception tests and contract review. | Preserve source coordinate frame by default. |
| Heavy Open3D imports slow tests. | Medium | Open3D can be heavy to import. | Test runtime. | Test timing and import review. | Import Open3D only where skeleton requires it. |
| Invalid point clouds pass silently. | Medium | Missing validation could corrupt training. | Data pipeline and perception. | Non-finite and shape tests. | Raise explicit errors. |

---

## 21. Design Decisions

| Decision | Evidence | Alternatives Considered | Reason for Selection |
| --- | --- | --- | --- |
| Add NumPy to runtime dependencies if directly imported. | Phase 3 loads `.npy` arrays and validates numeric point clouds. | Keep NumPy transitive only. | Direct imports should be declared explicitly. |
| Preserve thin CLI pattern for `scripts/prepare_data.py`. | Repository documentation describes thin CLI entry points. | Implement all logic in the script. | Preserves existing architecture. |
| Use JSON index output. | `docs/USAGE.md` shows `--output-index data/processed/index.json`. | Use CSV, pickle, or database index. | Matches documented usage and standard library support. |
| Do not parse YAML configuration in Phase 3. | No config loader verified; YAML parser dependency not verified. | Add global config loader. | Avoids global state and unnecessary dependency. |
| Use explicit dataset roots and index paths. | `docs/USAGE.md` CLI arguments use explicit paths. | Infer paths from config implicitly. | Preserves explicit dependency flow and testability. |
| Preserve source coordinate frame by default. | Robotics spatial reasoning requires frame consistency. | Automatically center and scale point clouds. | Prevents downstream frame mismatch. |
| Restrict batching to fixed-shape samples. | Variable-shape batching adds complexity and is not verified. | Implement padding and masking. | Keeps Phase 3 minimal and stable. |
| Do not implement augmentation. | Augmentation is not required for the minimum data contract. | Add random rotation, jitter, dropout. | Avoids unverified research behavior. |

---

## 22. Explicitly Rejected Changes

The following changes were considered and rejected.

### Rejected: creating a new dataset class

A PyTorch Dataset class might simplify integration with DataLoader, but it violates the no-new-class constraint. If an existing class exists, it may be modified. If not, function-based loading must be used or the phase must be paused.

### Rejected: creating a data utility module

Reusable path helpers, index validators, or array converters would be convenient, but they violate the no-helper constraint. Required behavior must be implemented directly in the appropriate existing skeleton unit.

### Rejected: creating a global configuration loader

A config loader would make `configs/data.yaml` easier to use, but Phase 3 does not require global configuration. It would introduce hidden state and unnecessary dependency risk.

### Rejected: adding a YAML parsing dependency

YAML parsing is not required for Phase 3 behavior because explicit paths can be used. Adding a YAML dependency would expand the dependency surface unnecessarily.

### Rejected: implementing data augmentation

Augmentation is research behavior and is not required for the minimum data contract. Adding it would increase nondeterminism and testing complexity.

### Rejected: implementing automatic train/validation/test splitting

Dataset splits are documented, but exact split semantics are not verified. Splitting should be handled by the existing skeleton contract or a later phase.

### Rejected: implementing model-specific normalization

Model-specific normalization belongs to the model input contract in Phase 4 or later. Phase 3 must avoid silently changing coordinate frames.

### Rejected: downloading datasets

Dataset downloading introduces network dependence and reproducibility risk. Phase 3 must work with local explicit dataset roots only.

---

## 23. Verification Evidence

| Claim | Evidence |
| --- | --- |
| Repository uses `hatchling` and `src` layout. | Verified `pyproject.toml`. |
| Python requirement is `>=3.12`. | Verified `pyproject.toml`. |
| Open3D and Torch are existing dependencies. | Verified `pyproject.toml`. |
| Data and perception modules are intended. | README repository layout. |
| Dataset loading and transforms belong to data module. | README repository layout. |
| Point-cloud preprocessing and SE(3) helpers belong to perception module. | README repository layout. |
| Dataset loaders must not contain model or robot logic. | `docs/PROJECT.md`. |
| Data preparation script exists. | Repository script table and `docs/USAGE.md`. |
| Data preparation writes an index. | `docs/USAGE.md` example argument `--output-index`. |
| Training consumes a processed dataset root. | `docs/USAGE.md` example argument `--dataset-root`. |
| Observations use `.npy` files. | `docs/USAGE.md` example observation path. |
| Exact data source files are unverified. | Available snapshot did not expose contents of `src/grasping_ai/data/`. |
| Exact perception source files are unverified. | Available snapshot did not expose contents of `src/grasping_ai/perception/`. |
| Existing tests are unverified. | Available snapshot did not expose a test suite. |

---

## 24. Definition of Done

Phase 3 is considered complete only when:

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

- [ ] Verified exact data-preparation skeleton file and symbol.
- [ ] Verified exact dataset loading skeleton file and symbol.
- [ ] Verified exact perception preprocessing skeleton file and symbol.
- [ ] Added NumPy to runtime dependencies if directly imported.
- [ ] Implemented data-preparation behavior.
- [ ] Implemented index loading behavior.
- [ ] Implemented point-cloud loading behavior.
- [ ] Implemented perception preprocessing behavior.
- [ ] Implemented fixed-shape batching behavior only if already part of the skeleton.
- [ ] Did not create new classes.
- [ ] Did not create helper functions.
- [ ] Did not create utility modules.
- [ ] Did not introduce global variables.
- [ ] Did not introduce global constants.
- [ ] Did not add file-level descriptions.
- [ ] Added Google-style docstrings to modified functions or methods.

### Tests

- [ ] Added NumPy dependency test.
- [ ] Added package import regression test.
- [ ] Added data configuration existence test.
- [ ] Added dataset preparation test.
- [ ] Added missing dataset root failure test.
- [ ] Added empty dataset failure test.
- [ ] Added index loading test.
- [ ] Added invalid index structure test.
- [ ] Added valid point-cloud loading test.
- [ ] Added invalid point-cloud shape test.
- [ ] Added non-finite point-cloud test.
- [ ] Added perception preprocessing shape test.
- [ ] Added perception invalid shape test.
- [ ] Added perception non-finite test.
- [ ] Added perception determinism test.
- [ ] Added fixed-shape batching tests if applicable.
- [ ] Added no-global-state test.

### Regression Verification

- [ ] Phase 1 package import remains successful.
- [ ] Phase 1 dependency tests remain successful.
- [ ] Phase 1 SE(3) tests remain successful if present.
- [ ] Phase 2 tests remain successful if present.
- [ ] No import cycles were introduced.
- [ ] No module-level side effects were introduced.

### Cross-Phase Verification

- [ ] Phase 4 can rely on dataset index and point-cloud loading contracts.
- [ ] Phase 4 can rely on preprocessed point-cloud output.
- [ ] Phase 5 can rely on perception validation if point-cloud observations are used.
- [ ] Phase 6 can rely on explicit data paths and deterministic preprocessing.
- [ ] Data modules do not import models, training, inference, simulation, or robotics.
- [ ] Perception modules do not import models, training, inference, simulation, or robotics.

### Structural Constraints

- [ ] No new dataset class exists.
- [ ] No helper abstraction exists.
- [ ] No utility abstraction exists.
- [ ] No global configuration abstraction exists.
- [ ] No unrelated refactoring exists.
- [ ] No experimental code exists.

### Commit Readiness

- [ ] Targeted Phase 3 tests pass.
- [ ] Regression tests pass.
- [ ] Broader relevant test suite passes.
- [ ] Static and structural review confirms forbidden constructs are absent.
- [ ] Commit contains only Phase 3 changes.