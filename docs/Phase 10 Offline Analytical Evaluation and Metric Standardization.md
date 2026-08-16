# Phase 10: Offline Analytical Evaluation and Metric Standardization

> **Historical design record.** This document captures the Phase 10 plan from the skeleton era. Current architecture, CLIs, and contracts live in [architecture.md](architecture.md), [USAGE.md](USAGE.md), and the repository [README.md](../README.md).

## 1. Phase Overview

**Phase name:** Phase 10: Offline Analytical Evaluation and Metric Standardization

**Objective:** Make the offline grasp evaluation pipeline analytically valid by replacing the dummy contact provider with deterministic analytical contact generation, adding a standardized analytical grasp-quality metric, and making evaluation parameters and report fields explicit. The phase must remain fully offline and must not depend on MuJoCo simulation, generated datasets, network access, or physical hardware.

**Why this phase is necessary based on verified repository state:**

Verified repository evidence shows that the offline evaluation path is structurally present but analytically incomplete.

- `src/grasping_ai/evaluation/collision.py` implements point-cloud collision checking using a KDTree and a clearance threshold.
- `src/grasping_ai/evaluation/force_closure.py` implements a wrench-space force-closure judge and a grasp wrench matrix builder.
- `src/grasping_ai/pipelines/evaluate.py` implements `evaluate_generated_grasps`, `aggregate_evaluation_results`, and `write_evaluation_report`.
- `scripts/evaluate.py` currently defines a local contact provider that returns an empty list for every grasp pose.
- Because the contact provider returns an empty list, `build_force_closure_judge` evaluates every grasp as not force-closed. Therefore offline force-closure evaluation is currently non-functional.
- `src/grasping_ai/pipelines/evaluate.py` hardcodes collision clearance and wrench regularization values inside `evaluate_generated_grasps`.
- `lift_height_threshold` is accepted and validated by `evaluate_generated_grasps`, but it is not used in the offline analytical evaluation. This is an existing interface inconsistency because offline point-cloud evaluation cannot measure lift height.
- `scripts/evaluate.py` accepts an `--object-id`, but currently uses it only as an aggregation key. It loads the grasp file as a plain array, while `scripts/generate_grasps.py` writes generated grasps as a pickled dictionary mapping object identifiers to grasp arrays. This creates an offline evaluation compatibility gap.

**Expected outcome:**

After this phase:

- Offline evaluation can generate analytical contact sets from object point clouds, gripper point clouds, and grasp poses.
- Offline force-closure evaluation no longer depends on an empty dummy contact provider.
- A standardized numerical grasp-quality metric is reported alongside the existing boolean force-closure result.
- Evaluation clearance and regularization parameters are explicit and preserve current default behavior.
- The evaluation script can consume both plain grasp arrays and object-keyed grasp dictionaries.
- Existing report keys remain present and backward compatible.
- No MuJoCo simulation is required.
- No physical robot, sensor, or dataset download is required.

## 2. Verified Current State

### Verified implemented

- **Point-cloud collision checking**
  - File: `src/grasping_ai/evaluation/collision.py`
  - Implemented units:
    - `build_collision_checker`
    - `check_collision`
    - `filter_collision_free_grasps`
  - Behavior: builds a KDTree over the object point cloud, transforms the gripper point cloud by a 4x4 grasp pose, and classifies a grasp as collision-free when all transformed gripper points are at least the clearance distance away from object points.

- **Force-closure analysis**
  - File: `src/grasping_ai/evaluation/force_closure.py`
  - Implemented units:
    - `load_contact_set`
    - `build_force_closure_judge`
    - `evaluate_force_closure`
    - `compute_grasp_wrench_matrix`
  - Behavior: converts contact records into a grasp wrench matrix using a four-sided friction-cone approximation, then uses rank checking and linear programming to decide force closure.

- **Offline evaluation orchestration**
  - File: `src/grasping_ai/pipelines/evaluate.py`
  - Implemented units:
    - `evaluate_generated_grasps`
    - `aggregate_evaluation_results`
    - `write_evaluation_report`
  - Behavior: evaluates collision freedom, requests contacts from a caller-supplied provider, evaluates force closure, aggregates rates, and writes a JSON report.

- **Evaluation CLI**
  - File: `scripts/evaluate.py`
  - Implemented units:
    - `evaluate_main`
    - argument parser block
  - Behavior: loads grasp poses, object point cloud, and gripper point cloud from `.npy` files, calls the evaluation pipeline, aggregates results, and writes a JSON report.

- **Simulation contact reporting**
  - File: `src/grasping_ai/simulation/mujoco_env.py`
  - Implemented behavior: MuJoCo simulation can report contacts through the contact reporter created by `create_simulation`.
  - This is relevant because it confirms contact reporting exists for simulation, but the offline evaluation path does not use it.

### Partially implemented

- **Offline analytical evaluation**
  - File: `src/grasping_ai/pipelines/evaluate.py`
  - The evaluation pipeline can compute collision freedom and force closure if given contacts, but it does not itself generate analytical contacts.
  - The `contact_set_provider` argument exists, but the only script-level provider currently returns an empty list.

- **Evaluation parameterization**
  - File: `src/grasping_ai/pipelines/evaluate.py`
  - Collision clearance and wrench regularization are currently hardcoded inside `evaluate_generated_grasps` instead of being explicit parameters.

### Missing

- **Analytical contact generation**
  - No inspected evaluation function generates a contact set from an object point cloud, gripper point cloud, and grasp pose.
  - Without this, offline force-closure evaluation cannot be valid.

- **Standardized numerical grasp-quality metric**
  - The current force-closure result is boolean.
  - No standardized scalar quality metric, such as a wrench-space robustness margin, is computed or reported.

- **Robust grasp-file loading in the evaluation script**
  - `scripts/evaluate.py` currently loads grasp poses as a plain array.
  - `scripts/generate_grasps.py` writes generated grasps as a dictionary keyed by object identifier.
  - The evaluation script does not currently select a grasp array from such a dictionary.

### Inconsistent

- **Dummy contact provider**
  - File: `scripts/evaluate.py`
  - The script supplies a contact provider that always returns an empty list.
  - This guarantees force-closure failure and makes the offline report misleading.

- **Unused lift-height threshold**
  - File: `src/grasping_ai/pipelines/evaluate.py`
  - `lift_height_threshold` is validated but not used in offline analytical evaluation.
  - Offline point-cloud evaluation cannot observe lift height, so the parameter is semantically inconsistent with the current offline path.

- **Hardcoded evaluation constants**
  - File: `src/grasping_ai/pipelines/evaluate.py`
  - Collision clearance and wrench regularization are hardcoded inside the evaluation pipeline.

- **Generated-grasp persistence mismatch**
  - File: `scripts/generate_grasps.py` writes a dictionary of object-keyed grasp arrays.
  - File: `scripts/evaluate.py` expects a plain grasp array.
  - This creates an interface mismatch between generation and offline evaluation.

### Unused or dead code relevant to this phase

- **`load_contact_set`**
  - File: `src/grasping_ai/evaluation/force_closure.py`
  - Implemented, but the current offline script does not load contact sets from disk.

- **Stability and lift judges**
  - File: `src/grasping_ai/evaluation/metrics.py`
  - `build_stability_judge`, `evaluate_stability`, `build_lift_outcome_judge`, `evaluate_lift_success`, and `aggregate_grasp_success_rate` are implemented.
  - The current offline evaluation script does not use these stability or lift-height judges because it has no simulated velocities or heights.

## 3. Phase Boundary

### Exact scope of what this phase modifies

This phase owns the following work:

1. Add analytical contact generation for offline evaluation.
2. Add a standardized scalar grasp-quality metric derived from contact wrench space.
3. Modify `evaluate_generated_grasps` to support analytical contact generation when no external contact provider is supplied.
4. Make collision clearance and wrench regularization explicit evaluation parameters while preserving current default behavior.
5. Extend evaluation result dictionaries and aggregate reports with standardized quality metrics.
6. Modify `scripts/evaluate.py` to remove the dummy contact provider and support both plain grasp arrays and object-keyed grasp dictionaries.
7. Add tests for analytical contacts, quality metric, offline evaluation behavior, and script-level compatibility.

### Explicitly excluded responsibilities

This phase does not implement:

- MuJoCo simulation-based evaluation.
- Dynamic lift-success evaluation.
- Stability evaluation based on simulated object velocity.
- Contact generation from physics simulation.
- Mesh-based contact sampling.
- YCB object loading changes.
- Gripper mesh generation.
- Training pipeline changes.
- Generative model changes.
- RL environment changes.
- Experiment tracking.
- Configuration-file parsing.
- Physical robot deployment.
- Real sensor processing.
- Changes to `scripts/generate_grasps.py` output format.
- Changes to `scripts/run_simulation.py`.

### What must NOT be touched

The following must remain unchanged unless directly required for backward-compatible optional arguments:

- Existing collision checker behavior in `build_collision_checker`.
- Existing wrench matrix computation in `compute_grasp_wrench_matrix`.
- Existing force-closure judge behavior in `build_force_closure_judge`.
- Existing JSON report writing behavior in `write_evaluation_report`.
- Existing simulation contact reporter behavior in `src/grasping_ai/simulation/mujoco_env.py`.
- Existing evaluation report keys:
  - `success_rate`
  - `collision_free_rate`
  - `force_closure_rate`
- Existing per-grasp keys:
  - `collision_free`
  - `force_closure`
  - `lift_success`

### MECE boundary definition

This phase owns **offline analytical grasp evaluation and metric standardization**.

It does not own:

- Simulation-based grasp execution.
- Dynamic lift or stability validation.
- Data generation.
- Model training.
- RL environment standardization.
- Experiment tracking.
- Deployment.

## 4. Architecture and Dependency Analysis

### Real dependency flow before this phase

Current offline evaluation flow:

- `scripts/evaluate.py`
  - loads grasp poses, object point cloud, and gripper point cloud
  - defines an empty contact provider
  - calls `src/grasping_ai/pipelines/evaluate.py::evaluate_generated_grasps`
    - calls `src/grasping_ai/evaluation/collision.py::build_collision_checker`
    - calls the supplied contact provider, which returns no contacts
    - calls `src/grasping_ai/evaluation/force_closure.py::build_force_closure_judge`
    - evaluates force closure on empty contacts
  - calls `aggregate_evaluation_results`
  - calls `write_evaluation_report`

Because the contact provider returns no contacts, force closure is always false.

### Required dependency flow after this phase

New offline evaluation flow:

- `scripts/evaluate.py`
  - loads grasp poses, supporting both array and dictionary persistence
  - calls `evaluate_generated_grasps` without a dummy provider
    - evaluates collision freedom
    - generates analytical contacts from object and gripper point clouds
    - evaluates force closure from analytical contacts
    - computes a standardized scalar grasp-quality metric
  - aggregates existing rates and new quality metrics
  - writes JSON report

### Input, processing, output chain

#### Input

- Grasp poses:
  - expected shape `(K, 4, 4)` for a single object,
  - or a dictionary mapping object identifiers to grasp arrays,
  - or a batched array with leading batch dimension one.
- Object point cloud:
  - shape `(N, 3)`.
- Gripper point cloud:
  - shape `(M, 3)`.
- Friction coefficient.
- Lift-height threshold, retained for interface compatibility but not used analytically.
- Optional contact clearance.
- Optional wrench regularization.

#### Processing

- Validate array shapes and finiteness.
- For each grasp pose:
  - transform gripper points into the object frame using the grasp pose.
  - find nearby object points within contact clearance.
  - construct analytical contact records with position and inward contact normal.
  - evaluate collision freedom.
  - evaluate force closure.
  - compute standardized grasp quality.
- Aggregate per-grasp results.

#### Output

- Per-grasp evaluation dictionaries containing:
  - existing boolean fields,
  - new scalar grasp-quality field.
- Aggregate report dictionary containing:
  - existing rates,
  - new quality statistics.
- JSON report file.

### Data contracts

#### Grasp pose

- Shape: `(4, 4)`.
- Type: NumPy array.
- Semantics: homogeneous transform placing the gripper in the object frame.
- Rotation block must be finite. No new rotation-orthonormality enforcement is required beyond finite-value validation.

#### Object point cloud

- Shape: `(N, 3)`.
- Type: NumPy array.
- Frame: object frame.
- Units: meters.

#### Gripper point cloud

- Shape: `(M, 3)`.
- Type: NumPy array.
- Frame: gripper frame before applying the grasp pose.
- Units: meters.

#### Contact record

Each analytical contact record must be a mapping containing at least:

- `position`: NumPy array of shape `(3,)`, expressed in the object frame.
- `normal`: NumPy array of shape `(3,)`, expressed in the object frame.

The normal must represent the feasible inward contact-force direction applied by the gripper on the object.

#### Report

- Persistence format: JSON.
- Existing keys must remain present.
- New keys must be numeric floats.

### Cross-module interactions

- `pipelines/evaluate.py` will depend on analytical contact generation in `evaluation/collision.py`.
- `pipelines/evaluate.py` will depend on standardized quality computation in `evaluation/force_closure.py`.
- No simulation module is required.
- No training or inference module is modified.
- No generative model module is modified.

## 5. Source Code Impact Analysis

### `src/grasping_ai/evaluation/collision.py`

**Current responsibility:** Point-cloud collision checking for grasps.

**Exact unit affected:** New analytical contact generation function.

**Current behavior:** The module can check whether transformed gripper points are too close to object points, but it does not expose contact positions or normals.

**Required change:** Add one domain-specific function that generates analytical contact records from an object point cloud, a gripper point cloud, a grasp pose, and a contact clearance.

**Reason for change:** Offline force-closure evaluation requires contact records. No existing function produces them analytically.

**Dependency impact:** Uses existing NumPy and SciPy KDTree behavior already present in the module.

**Regression risk:** Low. Existing collision functions remain unchanged.

### `src/grasping_ai/evaluation/force_closure.py`

**Current responsibility:** Contact-set loading, wrench matrix construction, boolean force-closure judgment.

**Exact unit affected:** New standardized grasp-quality function.

**Current behavior:** The module can decide force closure as a boolean, but it does not produce a scalar quality metric.

**Required change:** Add one domain-specific function that computes a nonnegative scalar grasp-quality metric from a contact set and friction coefficient.

**Reason for change:** Phase 10 requires standardized numerical evaluation, not only a boolean force-closure flag.

**Dependency impact:** Reuses existing `compute_grasp_wrench_matrix` and existing SciPy dependencies.

**Regression risk:** Low. Existing force-closure functions remain unchanged.

### `src/grasping_ai/pipelines/evaluate.py`

**Current responsibility:** Offline evaluation orchestration and reporting.

**Exact functions affected:**

- `evaluate_generated_grasps`
- `aggregate_evaluation_results`

**Current behavior:**

- `evaluate_generated_grasps` requires a contact provider.
- Collision clearance and wrench regularization are hardcoded.
- `lift_height_threshold` is validated but unused.
- Per-grasp results contain boolean fields only.
- `aggregate_evaluation_results` computes existing rates only.

**Required change:**

- Allow the contact provider to be absent.
- When absent, generate analytical contacts.
- Make collision clearance and wrench regularization explicit optional parameters with defaults preserving current behavior.
- Compute and include a scalar grasp-quality metric in per-grasp results.
- Extend aggregate results with quality statistics.
- Preserve all existing keys and boolean behavior.

**Reason for change:** The current offline path is analytically invalid without contacts and lacks standardized metric output.

**Dependency impact:** Depends on new analytical contact generation and quality metric functions.

**Regression risk:** Medium. Existing callers passing explicit contact providers must continue to work.

### `scripts/evaluate.py`

**Current responsibility:** CLI entry point for offline evaluation.

**Exact functions affected:**

- `evaluate_main`
- argument parser block

**Current behavior:**

- Defines a dummy contact provider returning an empty list.
- Loads grasp poses as a plain array.
- Uses `object_id` only as an aggregation key.
- Does not expose contact clearance or wrench regularization.

**Required change:**

- Remove the dummy contact provider.
- Call the evaluation pipeline with analytical contact generation.
- Support grasp files containing:
  - a plain `(K, 4, 4)` array,
  - a batched array with leading batch dimension one,
  - a dictionary mapping object identifiers to grasp arrays.
- Add optional CLI arguments for contact clearance and wrench regularization with defaults preserving current behavior.

**Reason for change:** The script currently guarantees force-closure failure and cannot consume generated-grasp dictionaries.

**Dependency impact:** Depends on updated pipeline signature.

**Regression risk:** Medium. Existing CLI invocations must remain valid because new arguments are optional.

### `src/grasping_ai/evaluation/metrics.py`

**Current responsibility:** Stability and lift-success judges.

**Exact functions affected:** None.

**Current behavior:** Provides dynamic lift and stability helpers that require velocities or heights.

**Required change:** None.

**Reason for change:** Offline analytical evaluation cannot measure dynamic lift or stability. These functions remain available for future simulation-based use.

**Dependency impact:** None.

**Regression risk:** Low.

## 6. New Source Code Units

### Required new function: analytical contact generation

**Location:** `src/grasping_ai/evaluation/collision.py`

**Responsibility:** Generate analytical contact records from an object point cloud, a gripper point cloud, a grasp pose, and a contact clearance.

**Inputs:**

- Object point cloud, shape `(N, 3)`.
- Gripper point cloud, shape `(M, 3)`.
- Grasp pose, shape `(4, 4)`.
- Contact clearance, nonnegative float.

**Outputs:**

- List of contact records.
- Each record contains at least `position` and `normal`.

**Failure behavior:**

- Invalid point-cloud shape raises a value error.
- Invalid grasp-pose shape raises a value error.
- Non-finite inputs raise a value error.
- Negative clearance raises a value error.
- Empty point clouds return an empty contact list.

**Why existing code cannot be extended instead:**

Existing collision checking only returns a boolean collision-free classification. It does not expose contact positions or normals. Force-closure analysis requires contact records. Therefore a dedicated analytical contact-generation function is required.

### Required new function: standardized grasp-quality metric

**Location:** `src/grasping_ai/evaluation/force_closure.py`

**Responsibility:** Compute a nonnegative scalar grasp-quality metric from a contact set and friction coefficient.

**Inputs:**

- Contact set.
- Friction coefficient.

**Outputs:**

- Nonnegative float quality value.
- Zero for empty, rank-deficient, or non-force-closure contact sets.

**Failure behavior:**

- Negative friction coefficient raises a value error.
- Invalid or empty contact sets return zero.
- Numerical failure in quality computation returns zero rather than raising, preserving offline evaluation robustness.

**Why existing code cannot be extended instead:**

The existing force-closure judge returns only a boolean. Phase 10 requires a standardized scalar metric for ranking and reporting. Extending the boolean judge’s return type would break existing callers, so a separate domain function is required.

### No other new source units

No new modules are required.

No new classes are required.

No helper utilities are required.

No configuration objects are required.

## 7. Existing Code Modification Rules

### Preserve existing evaluation invariants

- Existing collision checking semantics must remain unchanged.
- Existing force-closure boolean behavior must remain available.
- Existing report keys must remain present.
- Existing per-grasp keys must remain present.

### Preserve public interfaces

- `evaluate_generated_grasps` must continue to accept a callable contact provider.
- Existing callers that provide a contact provider must continue to work.
- New parameters must be optional and appended to existing signatures.
- Existing CLI arguments must remain valid.

### Preserve tensor and data contracts

- Grasp poses remain `(4, 4)` homogeneous transforms.
- Point clouds remain `(N, 3)` arrays.
- Contact records remain mappings containing `position` and `normal`.
- Aggregate report values remain JSON-serializable floats.

### Preserve numerical behavior unless explicitly required

- Default collision clearance must preserve the current hardcoded value.
- Default wrench regularization must preserve the current hardcoded value.
- Boolean force-closure behavior from the existing judge must remain available.
- The new scalar quality metric is an explicitly required numerical extension.

### No unrelated refactoring

Do not modify simulation, training, inference, generative models, data loading, or robotics modules.

Do not rename existing functions.

Do not extract helpers.

Do not introduce configuration loaders.

## 8. Detailed Behavioral Design

### Analytical contact generation behavior

The analytical contact generator must:

1. Validate object point cloud shape `(N, 3)` and finiteness.
2. Validate gripper point cloud shape `(M, 3)` and finiteness.
3. Validate grasp pose shape `(4, 4)` and finiteness.
4. Validate that contact clearance is nonnegative.
5. Transform gripper points into the object frame using the grasp pose.
6. Build a nearest-neighbor query over the object point cloud.
7. For each transformed gripper point, find the closest object point.
8. Create a contact record when the distance is less than or equal to the contact clearance, allowing a tiny numerical tolerance.
9. Set contact position to the object point.
10. Set contact normal to the inward feasible contact-force direction, derived from the object point and the transformed gripper point.
11. Return an empty list when no contacts satisfy the clearance condition.

The normal must not be taken from the outward vector pointing from object to gripper. It must represent the direction in which the gripper can apply force into the object.

### Standardized grasp-quality behavior

The standardized quality function must:

1. Validate that the friction coefficient is nonnegative.
2. Return zero for an empty contact set.
3. Use the existing grasp wrench matrix computation.
4. Return zero when the wrench matrix has insufficient rank or insufficient wrench samples.
5. Normalize wrench columns by the maximum finite wrench norm to produce a dimensionless quality value.
6. Attempt a convex-hull analysis of the normalized wrench points.
7. If the origin lies outside the convex hull, return zero.
8. If the origin lies inside the convex hull, return the minimum distance from the origin to the hull facets.
9. If convex-hull construction fails, fall back to a deterministic linear-programming margin derived from the same wrench-balance constraints used for force closure.
10. Return a nonnegative float.
11. Return zero instead of raising when numerical computation fails.

The quality metric must not introduce random sampling.

### Offline evaluation pipeline behavior

`evaluate_generated_grasps` must:

1. Preserve existing input validation for grasp poses, point clouds, friction coefficient, and lift-height threshold.
2. Accept optional contact clearance and wrench regularization parameters.
3. Use the current hardcoded values as defaults to preserve backward compatibility.
4. Build the collision checker using the explicit contact clearance.
5. Build the force-closure judge using the explicit wrench regularization.
6. Accept an optional contact provider.
7. If a contact provider is supplied, use it exactly as before.
8. If no contact provider is supplied, generate analytical contacts for each grasp pose.
9. Compute the scalar grasp-quality metric for every contact set.
10. Return per-grasp dictionaries containing:
    - existing `collision_free`,
    - existing `force_closure`,
    - existing `lift_success`,
    - new `grasp_quality`.

The existing `lift_success` field must remain defined as the offline analytical success combination currently used by the pipeline: collision-free and force-closed. The phase does not introduce dynamic lift-height evaluation.

The `lift_height_threshold` argument must remain accepted for interface compatibility. The design must explicitly document that offline analytical evaluation cannot use lift height.

### Aggregation behavior

`aggregate_evaluation_results` must:

1. Preserve existing aggregate keys:
   - `success_rate`
   - `collision_free_rate`
   - `force_closure_rate`
2. Add standardized quality statistics, at minimum:
   - mean grasp quality,
   - minimum grasp quality,
   - maximum grasp quality.
3. Use zero for quality statistics when no grasp-quality values are present.
4. Ensure all aggregate values are plain Python floats suitable for JSON serialization.

### Evaluation script behavior

`scripts/evaluate.py` must:

1. Remove the dummy contact provider.
2. Load the grasp file.
3. Support the following grasp-file shapes:
   - direct `(K, 4, 4)` array,
   - direct `(4, 4)` array,
   - batched array with leading dimension one,
   - dictionary mapping object identifiers to grasp arrays.
4. When the loaded grasp file is a dictionary:
   - select the array whose key matches the supplied object identifier,
   - if no exact match exists and the dictionary contains exactly one entry, use that entry,
   - otherwise raise a value error.
5. Pass the selected grasp array to the evaluation pipeline.
6. Pass analytical contact generation by supplying no contact provider.
7. Expose optional CLI arguments for:
   - contact clearance,
   - wrench regularization.
8. Preserve all existing required CLI arguments.

### Edge cases

- Empty object point cloud: collision checking and contact generation must fail or return empty behavior according to existing validation. The pipeline should raise a value error if the object point cloud is invalid.
- Empty gripper point cloud: contact generation returns no contacts; force closure and quality are zero.
- Clearance equal to zero: only exact or numerically near contacts are accepted.
- All contacts filtered out: force closure false, quality zero.
- Grasp dictionary missing requested object key: script raises a value error.
- Batched grasp array with batch size greater than one: script raises a value error because the CLI evaluates one object at a time.
- Non-finite grasp pose: evaluation raises a value error.
- Non-finite point cloud: evaluation raises a value error.
- Convex-hull failure: quality falls back to deterministic margin or zero.

### Failure cases

- Missing grasp file must raise a filesystem error.
- Missing point-cloud file must raise a filesystem error.
- Invalid NumPy payload must raise a value error.
- Invalid dictionary grasp payload must raise a value error.
- Negative friction coefficient must raise a value error.
- Negative contact clearance must raise a value error.
- Negative wrench regularization must raise a value error.

## 9. Cross-Phase Impact

### Impact on Phase 1: Foundation and Math Primitives

- No math primitive is modified.
- No package import contract is changed.

Required backward compatibility:

- Existing package imports remain valid.

### Impact on Phase 2: Simulation and Robotics Core

- No MuJoCo behavior is modified.
- Simulation contact reporting remains unchanged.
- Offline evaluation does not depend on simulation.

Required regression tests:

- Existing simulation tests must continue to pass.

### Impact on Phase 3: Data Pipeline and Perception

- Point-cloud loading contracts are unchanged.
- Evaluation consumes point clouds as arrays and does not modify data loading.

Required regression tests:

- Existing data and perception tests must continue to pass.

### Impact on Phase 4: Generative Grasp Model

- Generated grasp pose format remains `(K, 4, 4)` for single-object evaluation.
- The evaluation script gains compatibility with dictionary grasp files but does not modify generation.

Required regression tests:

- Existing generative-model tests must continue to pass.

### Impact on Phase 5: Reinforcement Learning Policy

- No RL policy behavior is modified.

Required regression tests:

- Existing RL tests must continue to pass.

### Impact on Phase 6: End-to-End Orchestration and Evaluation

- Offline evaluation becomes analytically valid.
- Simulation-based evaluation remains unchanged.
- Existing report keys remain backward compatible.

Required regression tests:

- Existing evaluation and orchestration tests must continue to pass.
- New tests must prove that offline evaluation no longer depends on the dummy contact provider.

## 10. Regression Protection

### Unit tests

Required unit tests must cover:

- Analytical contact generation with known geometry.
- Analytical contact generation rejection of invalid inputs.
- Contact normal direction consistency.
- Grasp-quality zero for empty contact sets.
- Grasp-quality zero for rank-deficient contact sets.
- Grasp-quality determinism for repeated identical inputs.
- Grasp-quality positive value for a synthetic full-rank contact set.
- Backward compatibility when a callable contact provider is supplied.

### Integration tests

Required integration tests must cover:

- `evaluate_generated_grasps` with no contact provider.
- `evaluate_generated_grasps` with an explicit contact provider.
- Aggregate report containing existing keys and new quality keys.
- `scripts/evaluate.py` loading a plain grasp array.
- `scripts/evaluate.py` loading an object-keyed grasp dictionary.
- `scripts/evaluate.py` producing a JSON report without MuJoCo.

### Simulation tests

No new simulation tests are required.

Existing simulation tests must remain untouched and passing.

### Numerical validation tests

Required numerical validation:

- Contact positions lie on or near the object point cloud.
- Contact normals point in the inward feasible force direction.
- Quality metric is nonnegative.
- Quality metric is deterministic.
- Existing collision behavior remains unchanged for the same clearance.
- Existing boolean force-closure behavior remains available.

### Explicit protection of previous phases

Before committing, the full repository test suite must pass, including:

- Phase 1 foundation tests.
- Phase 2 simulation and robotics tests.
- Phase 3 data and perception tests.
- Phase 4 generative model tests.
- Phase 5 RL tests.
- Phase 6 evaluation and orchestration tests.

## 11. New Test Suite

### `tests/unit/test_phase10_analytical_contacts.py`

**Purpose:** Validate analytical contact generation.

**Input strategy:**

- Small synthetic object point clouds.
- Small synthetic gripper point clouds.
- Identity and rotated grasp poses.
- Varying clearance values.

**Expected output:**

- Contact lists with expected count.
- Contact positions within tolerance of known object points.
- Contact normals oriented inward.
- Empty contact list when gripper points are beyond clearance.
- Value errors for invalid shapes or non-finite inputs.

**Failure condition:**

- Missing contacts for known touching geometry.
- Contact normals oriented outward.
- Non-deterministic outputs.
- Exceptions from valid empty-contact cases.

**Determinism requirement:**

- Fully deterministic NumPy/SciPy computation.
- No random sampling.

**CI suitability:**

- Fast, local, no MuJoCo, no datasets.

### `tests/unit/test_phase10_grasp_quality.py`

**Purpose:** Validate standardized grasp-quality computation.

**Input strategy:**

- Empty contact set.
- Rank-deficient synthetic contact set.
- Full-rank synthetic contact set.
- Invalid friction coefficient.

**Expected output:**

- Zero quality for empty contact set.
- Zero quality for rank-deficient contact set.
- Nonnegative deterministic quality for valid contact set.
- Positive quality for a synthetic force-closure-capable contact set.
- Value error for negative friction.

**Failure condition:**

- Negative quality.
- Non-deterministic quality.
- Positive quality for empty contact set.
- Crash on numerical degeneracy.

**Determinism requirement:**

- Same input produces identical quality.

**CI suitability:**

- Fast and local.

### `tests/unit/test_phase10_offline_evaluation.py`

**Purpose:** Validate modified offline evaluation pipeline behavior.

**Input strategy:**

- Synthetic grasps, object point cloud, and gripper point cloud.
- One call with no contact provider.
- One call with a callable provider returning empty contacts.
- One call with a callable provider returning custom contacts.

**Expected output:**

- Per-grasp dictionaries contain existing keys and `grasp_quality`.
- Analytical contacts are used when no provider is supplied.
- Supplied provider behavior is preserved.
- Existing aggregate keys remain present.
- New quality aggregate keys are present.

**Failure condition:**

- Missing existing keys.
- Missing quality key.
- Provider ignored when supplied.
- Non-JSON-serializable aggregate values.

**Determinism requirement:**

- Deterministic evaluation for fixed inputs.

**CI suitability:**

- Fast and local.

### `tests/integration/test_phase10_evaluate_script.py`

**Purpose:** Validate the evaluation script end to end.

**Input strategy:**

- Temporary `.npy` files for grasps, object point cloud, and gripper point cloud.
- One test with a plain grasp array.
- One test with an object-keyed grasp dictionary.
- Temporary report path.

**Expected output:**

- Script completes without MuJoCo.
- JSON report exists.
- Report contains existing keys and new quality keys.
- Dictionary loading selects the correct grasp array.

**Failure condition:**

- Script crashes on dictionary grasp file.
- Report missing existing keys.
- Force closure always false due to dummy provider.

**Determinism requirement:**

- Static temporary files.
- No randomness.

**CI suitability:**

- Local filesystem only.

## 12. Test Matrix

| Code unit | Affected phase | Test | Regression coverage | Commit gate |
|---|---:|---|---|---|
| Analytical contact generation | Phase 6, Phase 10 | `tests/unit/test_phase10_analytical_contacts.py` | Offline contact availability | Unit tests pass |
| Standardized quality metric | Phase 6, Phase 10 | `tests/unit/test_phase10_grasp_quality.py` | Numeric metric validity | Unit tests pass |
| `evaluate_generated_grasps` | Phase 6, Phase 10 | `tests/unit/test_phase10_offline_evaluation.py` | Existing evaluation contract | Unit tests pass |
| `aggregate_evaluation_results` | Phase 6, Phase 10 | `tests/unit/test_phase10_offline_evaluation.py` | Existing report keys | Unit tests pass |
| `scripts/evaluate.py` | Phase 6, Phase 10 | `tests/integration/test_phase10_evaluate_script.py` | CLI compatibility and grasp-file loading | Integration tests pass |
| Existing collision checker | Phase 6 | Existing collision tests | Collision behavior unchanged | Existing tests pass |
| Existing force-closure judge | Phase 6 | Existing force-closure tests | Boolean force-closure unchanged | Existing tests pass |

## 13. Numerical and Robotics Validation

### SE(3) and grasp poses

- Grasp poses remain 4x4 homogeneous transforms.
- The rotation block is used to transform gripper points.
- No new rotation representation is introduced.
- Non-finite pose values must raise errors.

### Point clouds

- Object and gripper point clouds remain `(N, 3)` arrays.
- Contact generation must not mutate input point clouds.
- Contact generation must reject non-finite points.

### Contact normals

- Contact normals must represent inward feasible contact-force directions.
- Normal vectors must be normalized unless distance is numerically zero.
- Degenerate zero-distance contacts must use a deterministic fallback normal or be skipped.

### Wrench-space quality

- Quality metric must be dimensionless after normalization.
- Quality metric must be nonnegative.
- Quality metric must be zero when the contact set cannot span the required wrench space.
- Quality computation must not use random direction sampling.

### Determinism constraints

- All new behavior must be deterministic for identical inputs.
- No global RNG state may be introduced.
- No time-dependent values may enter metrics.

## 14. Data and Interface Contracts

### Input formats

- Grasp file:
  - `.npy` array with shape `(K, 4, 4)`, or
  - `.npy` array with shape `(4, 4)`, or
  - `.npy` batched array with leading dimension one, or
  - `.npy` pickled dictionary mapping object identifiers to grasp arrays.
- Object point-cloud file:
  - `.npy` array with shape `(N, 3)`.
- Gripper point-cloud file:
  - `.npy` array with shape `(M, 3)`.

### Output format

- JSON report.
- Existing keys preserved.
- New keys are numeric floats.

### Units

- Meters for point clouds and contact positions.
- Radians are not directly used.
- Friction coefficient is dimensionless.
- Quality metric is dimensionless.

### Coordinate systems

- Object point cloud is expressed in the object frame.
- Grasp pose places the gripper in the object frame.
- Contact positions and normals are expressed in the object frame.

### Persistence

- No new persistence format is introduced.
- Existing `.npy` and JSON persistence remain unchanged.

## 15. Configuration and Dependency Impact

### Configuration files

No YAML configuration changes are required.

No YAML parsing is introduced.

No global configuration object is introduced.

### Dependency justification

- NumPy is already declared and required for array operations.
- SciPy is already declared and required for KDTree, linear programming, and convex-hull analysis.
- No new package dependency is introduced.

### Environment constraints

- Local filesystem access is required for input arrays and report output.
- No GPU is required.
- No network access is required.
- No MuJoCo simulation is required.
- No physical robot hardware is required.

## 16. Reproducibility and Local Validation

### How to run locally

1. Create small synthetic `.npy` files for:
   - grasp poses,
   - object point cloud,
   - gripper point cloud.
2. Run `scripts/evaluate.py` with those files and a temporary report path.
3. Inspect the JSON report.
4. Verify that existing keys are present.
5. Verify that new quality metrics are present.
6. Verify that force closure is no longer forced to false by an empty contact provider.

### Deterministic setup

- Use fixed synthetic arrays.
- Use CPU-only computation.
- Avoid randomness.
- Use default clearance and regularization for backward-compatible runs.

### Synthetic data usage

Tests must use small synthetic point clouds and grasp poses.

Tests must not require YCB assets.

### No physical robot dependency

All validation is offline and analytical.

## 17. Implementation Order

### Step 1: Add analytical contact generation

Modify `src/grasping_ai/evaluation/collision.py`.

Add tests for contact generation.

### Step 2: Add standardized quality metric

Modify `src/grasping_ai/evaluation/force_closure.py`.

Add tests for quality computation.

### Step 3: Extend offline evaluation pipeline

Modify `evaluate_generated_grasps` and `aggregate_evaluation_results` in `src/grasping_ai/pipelines/evaluate.py`.

Add pipeline tests.

### Step 4: Update evaluation script

Modify `scripts/evaluate.py` to remove the dummy provider, support dictionary grasp files, and expose optional clearance and regularization arguments.

Add script integration tests.

### Step 5: Run full regression suite

Run all existing repository tests and all new Phase 10 tests.

## 18. Commit Gates

A commit for this phase is allowed only if:

1. All existing tests pass.
2. All new Phase 10 tests pass.
3. The dummy contact provider is removed.
4. Offline evaluation produces analytical contacts without MuJoCo.
5. Existing report keys remain present.
6. Existing callable contact-provider behavior remains supported.
7. No helper functions are introduced beyond the two justified domain functions.
8. No utility modules are introduced.
9. No classes are introduced.
10. No global variables, global constants, or global mutable state are introduced.
11. Every new or modified function has a Google-style docstring.
12. No YAML configuration loading is introduced.
13. No physical robot dependency is introduced.
14. Static checks required by the repository pass.

## 19. Definition of Done

This phase is complete when all of the following are true:

- `scripts/evaluate.py` no longer supplies an empty contact provider.
- Offline evaluation generates analytical contact sets from point clouds and grasp poses.
- Offline force-closure evaluation uses non-empty analytical contacts when geometrically appropriate.
- A scalar grasp-quality metric is computed and included in per-grasp results.
- Aggregate reports include existing keys and new quality metrics.
- The evaluation script can load both plain grasp arrays and object-keyed grasp dictionaries.
- Existing CLI invocations remain valid.
- Existing report consumers remain compatible.
- All repository tests pass.
- Local validation requires no MuJoCo, GPU, network, dataset download, or physical robot.

## 20. Risks and Failure Modes

### Technical risk: approximate contact normals from point clouds

**Risk:** Point-cloud-derived normals may be less accurate than mesh-derived surface normals.

**Mitigation:** Define the phase metric as an offline analytical approximation. Use deterministic inward pair-direction normals. Document that mesh-based contact generation is out of scope.

### Technical risk: high-dimensional convex hull fragility

**Risk:** Six-dimensional convex-hull computation may fail for degenerate contact sets.

**Mitigation:** Return zero for degenerate cases and provide a deterministic linear-programming fallback when hull construction fails. Tests must cover degenerate contact sets.

### Numerical risk: quality metric scale sensitivity

**Risk:** Wrench magnitudes depend on object scale and contact positions.

**Mitigation:** Normalize wrench columns before computing the standardized quality metric. Report quality as dimensionless.

### Integration risk: existing report consumers

**Risk:** Additional report fields could break strict consumers.

**Mitigation:** Preserve all existing keys and value types. Add only new numeric fields.

### Regression risk: contact provider behavior

**Risk:** Existing callers that pass custom contact providers could be affected.

**Mitigation:** If a provider is supplied, the pipeline must use it exactly as before. Add explicit backward-compatibility tests.

### Integration risk: generated grasp dictionary mismatch

**Risk:** Generated grasp dictionaries may contain keys that do not match user-supplied object identifiers.

**Mitigation:** Script must support exact key match, single-entry fallback, and explicit failure for ambiguous cases.

### Performance risk: repeated spatial indexing

**Risk:** Analytical contact generation may rebuild spatial indexes for many grasps.

**Mitigation:** Accept for offline evaluation in this phase. If performance becomes blocking, optimize in a later phase without changing contracts.

## 21. Out of Scope

The following are explicitly out of scope for this phase:

- MuJoCo simulation-based evaluation.
- Dynamic lift-success evaluation.
- Stability evaluation from simulated velocities.
- Contact generation from physics simulation.
- Mesh-based contact generation.
- YCB object attachment.
- Gripper mesh creation.
- Training pipeline changes.
- RL environment changes.
- Generative model changes.
- Experiment tracking.
- Configuration-file parsing.
- Global configuration systems.
- Physical robot deployment.
- Real sensor integration.
- Changing the output format of `scripts/generate_grasps.py`.
- Changing the behavior of `scripts/run_simulation.py`.

## 22. Design Review Checklist

- Repository state verified against collision, force closure, evaluation pipeline, evaluation script, and related persistence behavior.
- Phase boundary restricted to offline analytical evaluation and metric standardization.
- Dummy contact provider removed.
- Analytical contact generation justified and bounded.
- Standardized quality metric justified and bounded.
- Existing report keys preserved.
- Existing callable contact-provider behavior preserved.
- No helper utilities introduced beyond required domain functions.
- No classes introduced.
- No global state introduced.
- No YAML configuration loading introduced.
- Tests cover contacts, quality, pipeline compatibility, and script loading.
- Local validation uses synthetic arrays only.
- No MuJoCo or physical robot dependency introduced.