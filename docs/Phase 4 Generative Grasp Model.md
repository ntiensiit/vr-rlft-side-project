# Phase 4 — Generative Grasp Model

## 1. Phase Objective

Phase 4 establishes the minimum generative grasp-model capability required before generated grasp poses can be executed or evaluated in later phases.

The phase must deliver only the following capabilities:

1. A generative model behavior capable of producing 6-DoF grasp pose candidates from point-cloud input.
2. A supervised training behavior that can consume the Phase 3 dataset contract and save a checkpoint.
3. An inference behavior that can load a checkpoint, sample grasp poses from a point-cloud observation, and write generated grasps to an explicit output file.
4. A stable grasp-pose output contract compatible with later simulation and evaluation phases.
5. Tests that prove model forward behavior, training checkpoint creation, inference output shape, and SE(3) validity without requiring external datasets, GPUs, or unverified model assets.

This phase is intentionally conservative. The repository is a skeleton, and the exact source files inside `src/grasping_ai/models/`, `src/grasping_ai/training/`, and `src/grasping_ai/inference/` were not fully verifiable in the available snapshot. Therefore, this design distinguishes verified facts from unverified repository details. Where an exact file path or symbol cannot be verified, it is marked `Unverified`.

The implementation must not create new classes, helper utilities, global state, or abstraction layers. If the repository does not already contain a skeleton model definition, training function, inference function, or equivalent entry point, the phase must be paused for a scope decision rather than creating a new model class or utility module.

---

## 2. Verified Repository Context

The following facts were verified from the repository snapshot available during analysis.

### Verified packaging facts

- `pyproject.toml` exists.
- The build backend is `hatchling.build`.
- The wheel target uses `packages = ["src"]`.
- Python requirement is `>=3.12`.
- Runtime dependencies already include:
  - `torch`
  - `torchvision`
  - `open3d`
  - `pytransform3d`
  - `scipy`
  - `theseus`
  - `gymnasium`
  - `stable-baselines3`
  - `tensorboard`
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
  - `src/grasping_ai/models/` as containing equivariant encoder, diffusion, flow, and RL policy models.
  - `src/grasping_ai/training/` as containing supervised and RL training loops.
  - `src/grasping_ai/inference/` as containing trained-model loading and grasp/action generation.
- `README.md` defines the grasp-generation flow as:

  `sensors -> perception -> models -> inference -> pipelines.generate_grasps`

and the supervised training flow as:

  `data -> perception -> models -> training -> inference`

- `docs/PROJECT.md` defines the dependency direction:

  `configs -> data -> perception -> models -> inference -> robotics -> simulation -> evaluation`

It also states that training and pipeline modules orchestrate lower-level components rather than being dependencies of them.

### Verified script and usage facts

- The repository script table lists:
  - `scripts/train.py`
  - `scripts/generate_grasps.py`
- `docs/USAGE.md` describes `scripts/train.py` as training the grasp-pose generation model using supervised learning.
- `docs/USAGE.md` describes `scripts/generate_grasps.py` as generating grasp poses from object point clouds.
- `docs/USAGE.md` shows `scripts/train.py` receiving arguments including:
  - dataset root
  - checkpoint path
  - feature dimension
  - hidden dimension
  - number of layers
  - learning rate
  - number of epochs
  - batch size
  - device
- `docs/USAGE.md` shows `scripts/generate_grasps.py` receiving arguments including:
  - checkpoint path
  - observations path
  - output path
  - feature dimension
  - number of diffusion steps
  - number of grasps
  - device
- `docs/USAGE.md` describes `configs/model.yaml` as containing generative grasping model architecture settings, including generative model type, number of steps, hidden dimensions, and RL model settings.

### Non-verified or partially verified areas

The following are not verified in the available snapshot:

- Exact file names under `src/grasping_ai/models/`.
- Exact file names under `src/grasping_ai/training/`.
- Exact file names under `src/grasping_ai/inference/`.
- Exact function names, class names, and signatures inside those directories.
- Existing test files, fixtures, and test conventions.
- Exact YAML key names inside `configs/model.yaml`.
- Whether the existing skeleton uses diffusion, flow, or both.
- Whether the existing skeleton defines model classes or functional model constructors.
- Exact supervised target format expected by the training skeleton.

Because of this, any source-unit design inside those directories is marked `Unverified` where exact paths or symbols cannot be confirmed.

---

## 3. Scope

Phase 4 includes only the following work.

### In scope

1. Implementation of the existing generative grasp-model skeleton.
   - Model behavior conditioned on point-cloud input.
   - Generative denoising or sampling behavior required by the existing skeleton.
   - Output of grasp-pose parameters suitable for conversion to SE(3) homogeneous transforms.

2. Implementation of the existing supervised training skeleton.
   - Load dataset records using the Phase 3 data contract.
   - Convert target grasp poses into the model training representation.
   - Compute a supervised generative loss.
   - Update model parameters.
   - Save a checkpoint.

3. Implementation of the existing grasp inference skeleton.
   - Load a checkpoint.
   - Load point-cloud observations.
   - Sample the requested number of grasps.
   - Convert sampled grasp representations into SE(3) homogeneous transforms.
   - Write generated grasps to an explicit output file.

4. Preservation of static model configuration.
   - Do not remove `configs/model.yaml`.
   - Do not introduce global configuration loading.
   - Do not introduce module-level configuration state.

5. Tests for the Phase 4 contract.
   - Model forward shape tests.
   - Training checkpoint tests.
   - Inference output shape tests.
   - SE(3) validity tests.
   - Regression tests for Phase 1 and Phase 3 behavior.

### Conditional scope

The generative model implementation is conditional on locating existing skeleton units.

If no existing model definition, model constructor, or equivalent model entry point exists, Phase 4 must not create a new model class. The correct action is to stop and request a scope decision because the no-new-class constraint would otherwise be violated.

If no existing supervised training function exists, Phase 4 must not create a new training utility module. The correct action is to stop and request a scope decision.

If no existing grasp inference function exists, Phase 4 must not create a new inference utility module. The correct action is to stop and request a scope decision.

---

## 4. Out of Scope

The following are explicitly out of scope for Phase 4.

1. Reinforcement learning.
   - RL policy training.
   - RL reward design.
   - Stable-Baselines3 integration.
   - Environment rollouts.

2. Simulation execution.
   - MuJoCo stepping.
   - Grasp execution.
   - YCB object loading in simulation.
   - Lift or force-closure evaluation.

3. Advanced generative modeling.
   - Flow matching if the existing skeleton does not already require it.
   - Advanced equivariant neural network architectures beyond the existing skeleton.
   - Multi-object or scene-level grasp generation.
   - Grasp ranking or quality learning.
   - Latent diffusion beyond the existing skeleton.

4. Advanced training infrastructure.
   - Distributed training.
   - Mixed precision.
   - Experiment tracking integration.
   - Hyperparameter sweeps.
   - Early stopping frameworks.
   - Learning-rate schedulers unless already required by the existing skeleton.

5. Advanced inference behavior.
   - Grasp filtering.
   - Collision-aware grasp selection.
   - Grasp ranking.
   - Batched multi-observation streaming.
   - Real-time inference optimization.

6. Dataset expansion.
   - Dataset downloading.
   - Dataset augmentation.
   - Dataset caching.
   - New dataset formats.

7. New architecture.
   - New model base classes.
   - New trainer classes.
   - New inference wrapper classes.
   - New helper functions.
   - New utility modules.
   - New global configuration managers.

---

## 5. Existing Architecture and Patterns

Phase 4 must preserve the following verified or documented repository patterns.

### Pattern 1: Thin CLI scripts delegate to pipeline functions

The repository documentation describes scripts as thin CLI entry points. `scripts/train_rl.py` was verified as a thin script that parses arguments and calls a pipeline function.

Phase 4 must preserve this pattern for `scripts/train.py` and `scripts/generate_grasps.py`. The scripts should parse arguments and delegate to existing pipeline or training/inference functions. They should not contain model, training, or inference business logic.

### Pattern 2: Skeleton functions raise `NotImplementedError`

The README states that the repository contains skeleton code with `NotImplementedError` bodies. Phase 4 should replace `NotImplementedError` only in existing functions, methods, or module execution blocks that belong to the phase scope.

Phase 4 must not create new reusable helper functions to avoid implementing required behavior directly in the appropriate existing skeleton unit.

### Pattern 3: Lower-level modules do not depend on orchestration

The documented dependency direction places `models` below `inference`, `robotics`, `simulation`, and `evaluation`.

Phase 4 model code must not import pipeline functions, training loops, inference loaders, simulation modules, or CLI scripts. Training and inference may orchestrate lower-level modules, but models must remain low-level.

### Pattern 4: AI models do not directly depend on robot hardware

`docs/PROJECT.md` states that AI models must not directly depend on robot hardware.

Phase 4 model behavior must not import robotics, simulation, gripper control, or hardware-related modules.

### Pattern 5: Dataset loaders do not contain model logic

Phase 3 established that dataset loaders load data but do not interpret model semantics. Phase 4 must consume the Phase 3 data contract without modifying dataset loaders to contain model-specific behavior.

### Pattern 6: Configuration remains static

The repository uses YAML configuration files, but no global configuration loader was verified.

Phase 4 must not introduce a global configuration object, module-level config cache, or singleton configuration manager. Model hyperparameters should be accepted explicitly through the existing CLI or pipeline interface.

---

## 6. Implementation Dependencies

Phase 4 requires the following dependencies.

### Existing dependencies that must remain available

| Dependency | Reason |
| --- | --- |
| `torch` | Model parameters, training, inference, tensor operations, checkpoint serialization. |
| `numpy` | Numeric array handling established by Phase 3 for `.npy` observations and generated grasps. |
| `pytransform3d` | Existing spatial math dependency for SE(3) conventions where required. |
| `pytest` | Development dependency for Phase 4 tests. |

### Dependencies explicitly not required

| Dependency | Reason for exclusion |
| --- | --- |
| New generative-model library | Existing Torch dependency is sufficient for the minimum supervised generative behavior. |
| New experiment tracking library | Phase 4 does not require experiment dashboards or remote logging. |
| New configuration library | Global configuration is out of scope. |
| New visualization library | Phase 4 does not require grasp visualization. |

No new third-party dependency may be introduced.

---

## 7. Source-Unit Change Matrix

Rows marked `Unverified` are required phase targets, but the exact file path or symbol must be resolved from the repository before implementation. They are not speculative new files.

| File | Symbol | Change Type | Current Responsibility | Proposed Change | Risk | Required Tests |
| --- | --- | --- | --- | --- | --- | --- |
| `configs/model.yaml` | Documented generative model settings | Configuration preservation | Declares generative model architecture semantics according to documentation. | Preserve file and documented semantics. Do not introduce global loading. If exact keys are verified, keep them unchanged. | Low | Configuration existence test. |
| `scripts/train.py` | Unverified existing CLI entry function or module-level execution block | Modify | Currently expected to raise `NotImplementedError` or delegate to an unimplemented pipeline function. | Implement or complete the supervised training entry behavior while preserving the thin CLI pattern. | High | Training entry and checkpoint tests. |
| Unverified existing pipeline or training module imported by `scripts/train.py` | Unverified existing supervised training function | Modify | Currently expected to raise `NotImplementedError`. | Implement dataset loading, model training, loss computation, parameter updates, and checkpoint saving. | High | Training tests. |
| Unverified existing module under `src/grasping_ai/models/` | Unverified existing generative model constructor, forward function, or existing class method | Modify | Currently expected to raise `NotImplementedError`. | Implement generative model behavior conditioned on point clouds, noisy grasp representations, and diffusion step information. | High | Model forward tests. |
| Unverified existing module under `src/grasping_ai/models/` or training module | Unverified existing diffusion loss or sampling behavior | Modify | Currently expected to raise `NotImplementedError`. | Implement supervised diffusion loss behavior and sampling behavior required by the generative model. | High | Loss and sampling tests. |
| Unverified existing module under `src/grasping_ai/training/` or inference module | Unverified existing checkpoint saving/loading behavior | Modify | Currently expected to raise `NotImplementedError`. | Implement checkpoint serialization and loading with explicit architecture metadata. | High | Checkpoint round-trip tests. |
| `scripts/generate_grasps.py` | Unverified existing CLI entry function or module-level execution block | Modify | Currently expected to raise `NotImplementedError` or delegate to an unimplemented pipeline function. | Implement or complete the grasp inference entry behavior while preserving the thin CLI pattern. | High | Inference entry tests. |
| Unverified existing pipeline or inference module imported by `scripts/generate_grasps.py` | Unverified existing grasp inference function | Modify | Currently expected to raise `NotImplementedError`. | Implement checkpoint loading, observation loading, grasp sampling, SE(3) conversion, and output writing. | High | Inference output tests. |
| Existing or minimum required test file under `tests/unit/` | Phase 4 generative model tests | Test-only change | No verified existing Phase 4 tests. | Add tests for model forward, training, checkpointing, inference, SE(3) validity, and regression behavior. | Medium | All Phase 4 tests pass. |

---

## 8. Detailed Source-Unit Design

### `configs/model.yaml`

#### Current behavior

Documentation states that this file contains generative grasping model architecture settings, including generative model type, number of steps, hidden dimensions, and RL model settings. The exact key names were not verified.

#### Required behavior

Phase 4 must preserve the file and its documented semantics. Phase 4 must not introduce global configuration loading. If model hyperparameters are needed by training or inference, they should be accepted explicitly through the existing CLI or pipeline interface.

#### Design

- Input: none.
- Output: none.
- Control flow: none.
- State changes: none.
- Existing dependencies: none.
- Error behavior: not applicable during Phase 4 because no runtime YAML parsing is introduced.
- Edge cases: missing keys should be handled by later pipeline validation if that pipeline chooses to consume the file.
- Interaction with surrounding code: later orchestration may read this file, but Phase 4 does not require it.
- Existing pattern being followed: static configuration without global state.

No YAML key may be renamed, removed, or added unless exact key verification proves that a documented key is missing.

---

### `scripts/train.py`

#### Current behavior

The script exists and is documented as the entry point for supervised training of the grasp-generation model. The exact function or module-level execution block was not verified. Because the repository is a skeleton, the current behavior is expected to be `NotImplementedError` or delegation to an unimplemented pipeline function.

#### Required behavior

The script must support the documented supervised training entry point. It must accept explicit training arguments and invoke the existing supervised training behavior.

The script must remain thin. It must not contain model construction, dataset loading, loss computation, or checkpoint-writing logic if an existing pipeline or training function is the intended target.

#### Design

##### Input

Expected inputs include:

- Processed dataset root.
- Checkpoint output path.
- Feature dimension.
- Hidden dimension.
- Number of layers.
- Learning rate.
- Number of epochs.
- Batch size.
- Device identifier.

Additional inputs may exist in the skeleton. The implementation must preserve the existing CLI contract.

##### Output

The script itself produces no meaningful return value. Its side effect is the creation of a checkpoint file at the requested path.

##### Control flow

1. Parse CLI arguments using the existing argument parsing behavior.
2. Call the existing supervised training pipeline or training function.
3. Propagate errors naturally.

##### State changes

No module-level state.

##### Existing dependencies

- Existing training pipeline or training module.
- Standard argument parsing if already used by the script.

##### Error behavior

- Missing dataset root raises a filesystem or value error.
- Invalid checkpoint path raises a filesystem or value error.
- Invalid numeric arguments raise value errors.

##### Edge cases

- Dataset root exists but contains no usable records.
- Checkpoint parent directory does not exist.
- Device identifier is unsupported.

##### Interaction with surrounding code

Later inference behavior loads the checkpoint produced by this entry point.

##### Existing pattern being followed

Thin CLI script delegates to lower-level behavior.

---

### Unverified existing supervised training function

#### Current behavior

The exact training symbol was not verified. The current behavior is expected to be `NotImplementedError`.

#### Required behavior

The supervised training function must train the generative grasp model using the Phase 3 dataset contract and save a checkpoint.

#### Design

##### Input

Expected inputs include:

- Dataset root.
- Checkpoint output path.
- Feature dimension.
- Hidden dimension.
- Number of layers.
- Learning rate.
- Number of epochs.
- Batch size.
- Device identifier.

The implementation must follow the existing skeleton signature.

##### Output

The function may return final loss, training metadata, or nothing if the existing skeleton contract defines no return value.

##### Control flow

1. Validate explicit arguments.
2. Load dataset records using the Phase 3 data contract.
3. Validate that target grasp poses are available for supervised training.
4. Construct or initialize the existing generative model skeleton.
5. Construct an optimizer using existing Torch behavior.
6. For each epoch:
   1. Iterate over fixed-shape batches.
   2. Convert target SE(3) grasps into the model training representation.
   3. Sample generative training noise and step information.
   4. Compute the supervised generative loss.
   5. Update model parameters.
7. Save the checkpoint.

##### State changes

Model parameter state changes during training. Module-level state must not be introduced.

##### Existing dependencies

- Phase 3 dataset loading behavior.
- Phase 4 generative model behavior.
- Torch optimization and tensor operations.

##### Error behavior

- Missing dataset raises a filesystem error.
- Dataset without target grasps raises a value error.
- Invalid target transforms raise value errors.
- Non-finite loss raises an error or fails explicitly.

##### Edge cases

- Batch size larger than dataset size.
- Single-record dataset.
- Target grasp count varying across records.
- CPU-only execution.

##### Interaction with surrounding code

The checkpoint is consumed by grasp inference.

##### Existing pattern being followed

Training orchestrates data and models but does not modify low-level data loaders.

---

### Unverified existing generative model unit

#### Current behavior

The exact model symbol was not verified. The current behavior is expected to be `NotImplementedError`.

#### Required behavior

The generative model unit must produce a prediction conditioned on point-cloud input, noisy grasp representation, and diffusion step information.

For Phase 4, the minimum generative contract is a conditional denoising model. If the existing skeleton defines a different generative contract, the existing contract must be preserved.

#### Design

##### Input

Expected inputs include:

- Point-cloud tensor with shape `B x N x 3` or an equivalent existing contract.
- Noisy grasp representation tensor with shape `B x D`.
- Diffusion step or time tensor with shape `B`.

The recommended minimum grasp representation dimension is `D = 9`, consisting of:

- 3 translation values.
- 6 rotation values using a 6D rotation representation.

If the existing skeleton already defines another representation, that representation must be preserved.

##### Output

The output must match the existing skeleton contract. For the minimum denoising contract, the output is a predicted noise or velocity tensor with shape `B x D`.

##### Control flow

1. Validate input shapes.
2. Extract a point-cloud feature representation.
3. Encode diffusion step information.
4. Combine point-cloud features, noisy grasp representation, and step information.
5. Predict the generative target.
6. Return the prediction.

##### State changes

Model parameters are read during forward execution. No module-level state may be modified.

##### Existing dependencies

- Torch modules or tensor operations already permitted by the skeleton.
- Phase 1 SE(3) math behavior where rotation conversion is required.

##### Error behavior

- Invalid point-cloud shape raises a value error.
- Invalid grasp representation shape raises a value error.
- Non-finite inputs raise a value error.

##### Edge cases

- Batch size one.
- Small point clouds.
- CPU-only execution.
- Inference mode versus training mode.

##### Interaction with surrounding code

Training and inference call this unit. It must not call training, inference, or pipeline code.

##### Existing pattern being followed

Models remain low-level and independent of orchestration.

---

### Unverified existing diffusion loss or sampling behavior

#### Current behavior

The exact diffusion symbol was not verified. The current behavior is expected to be `NotImplementedError`.

#### Required behavior

The diffusion behavior must support the minimum supervised training loss and inference sampling required by the generative model.

#### Design

##### Training behavior

1. Convert target SE(3) grasps into the model training representation.
2. Sample diffusion step indices.
3. Sample noise compatible with the grasp representation.
4. Produce noisy grasp representations.
5. Call the generative model unit.
6. Compute a mean-squared-error loss between predicted and target generative quantities.

##### Inference behavior

1. Start from random noise for each requested grasp.
2. Iteratively denoise using the trained model for the requested number of diffusion steps.
3. Produce final grasp representations.
4. Convert final grasp representations to SE(3) homogeneous transforms.

##### State changes

No module-level state.

##### Existing dependencies

- Torch random tensor generation.
- Existing generative model unit.
- Phase 1 or Phase 3 SE(3) validation behavior where applicable.

##### Error behavior

- Invalid diffusion step count raises a value error.
- Invalid requested grasp count raises a value error.
- Non-finite sampled values raise an error or are explicitly rejected.

##### Edge cases

- One diffusion step.
- Large diffusion step count.
- One requested grasp.
- Batched observations.

##### Interaction with surrounding code

Training uses the loss behavior. Inference uses the sampling behavior.

##### Existing pattern being followed

Generative behavior remains inside model or inference units and does not depend on simulation or robotics.

---

### Unverified existing checkpoint saving/loading behavior

#### Current behavior

The exact checkpoint symbol was not verified. The current behavior is expected to be `NotImplementedError`.

#### Required behavior

The checkpoint behavior must save and load the generative model state together with the explicit architecture metadata needed for inference.

#### Design

##### Checkpoint content contract

The checkpoint must contain at least:

- Model parameter state.
- Feature dimension.
- Hidden dimension.
- Number of layers.
- Generative model type identifier.
- Grasp representation dimension.

If the existing skeleton already defines a checkpoint contract, that contract must be preserved.

##### Save behavior

1. Validate model state and architecture metadata.
2. Create a JSON-compatible or Torch-serializable mapping.
3. Write the checkpoint to the requested path.

##### Load behavior

1. Read the checkpoint file.
2. Validate required fields.
3. Validate architecture compatibility with inference arguments where provided.
4. Return model state and metadata.

##### State changes

Filesystem side effect during saving. No module-level state.

##### Existing dependencies

- Torch serialization.
- Standard filesystem utilities.

##### Error behavior

- Missing checkpoint raises a filesystem error.
- Missing required fields raises a value error.
- Architecture mismatch raises a value error.

##### Edge cases

- Checkpoint parent directory missing during save.
- Corrupted checkpoint file.
- Checkpoint created with a different feature dimension.

##### Interaction with surrounding code

Training writes checkpoints. Inference reads checkpoints.

##### Existing pattern being followed

Explicit checkpoint metadata avoids hidden runtime assumptions.

---

### `scripts/generate_grasps.py`

#### Current behavior

The script exists and is documented as the entry point for generating grasp poses from object point clouds. The exact function or module-level execution block was not verified. Because the repository is a skeleton, the current behavior is expected to be `NotImplementedError` or delegation to an unimplemented pipeline function.

#### Required behavior

The script must support the documented grasp inference entry point. It must accept explicit inference arguments and invoke the existing grasp inference behavior.

The script must remain thin. It must not contain model loading, sampling, or SE(3) conversion logic if an existing pipeline or inference function is the intended target.

#### Design

##### Input

Expected inputs include:

- Checkpoint path.
- Observations path.
- Output path.
- Feature dimension.
- Number of diffusion steps.
- Number of grasps.
- Device identifier.

Additional inputs may exist in the skeleton. The implementation must preserve the existing CLI contract.

##### Output

The script itself produces no meaningful return value. Its side effect is the creation of a generated-grasps file at the requested output path.

##### Control flow

1. Parse CLI arguments using the existing argument parsing behavior.
2. Call the existing grasp inference pipeline or inference function.
3. Propagate errors naturally.

##### State changes

No module-level state.

##### Existing dependencies

- Existing inference pipeline or inference module.
- Standard argument parsing if already used by the script.

##### Error behavior

- Missing checkpoint raises a filesystem error.
- Missing observation file raises a filesystem error.
- Invalid output path raises a filesystem or value error.

##### Edge cases

- Observation file contains a single point cloud.
- Observation file contains a batch of point clouds.
- Output parent directory does not exist.

##### Interaction with surrounding code

Later simulation execution consumes the generated grasps file.

##### Existing pattern being followed

Thin CLI script delegates to lower-level behavior.

---

### Unverified existing grasp inference function

#### Current behavior

The exact inference symbol was not verified. The current behavior is expected to be `NotImplementedError`.

#### Required behavior

The grasp inference function must load a trained generative model, sample the requested number of grasps from point-cloud observations, and write valid SE(3) grasp poses to the output path.

#### Design

##### Input

Expected inputs include:

- Checkpoint path.
- Observation path.
- Output path.
- Feature dimension.
- Number of diffusion steps.
- Number of grasps.
- Device identifier.

The implementation must follow the existing skeleton signature.

##### Output

The function may return the output path or the generated grasp array if the existing skeleton contract defines a return value.

The generated grasp output contract is:

- For a single observation: shape `K x 4 x 4`.
- For a batch of observations: shape `B x K x 4 x 4`.

`K` is the requested number of grasps.

##### Control flow

1. Load and validate the checkpoint.
2. Load and validate the observation array.
3. Preprocess the observation using Phase 3 perception behavior if required by the existing skeleton.
4. Replicate observations for the requested number of grasps.
5. Sample initial noise.
6. Iteratively denoise using the trained model.
7. Convert final grasp representations into SE(3) homogeneous transforms.
8. Validate rotation matrices.
9. Write the output array.

##### State changes

Filesystem side effect: writes generated grasps. No module-level state.

##### Existing dependencies

- Phase 4 checkpoint loading behavior.
- Phase 4 generative sampling behavior.
- Phase 3 perception preprocessing if required.
- Phase 1 SE(3) math behavior where applicable.

##### Error behavior

- Missing checkpoint raises a filesystem error.
- Invalid observation shape raises a value error.
- Checkpoint architecture mismatch raises a value error.
- Non-finite sampled grasps raise an error or are explicitly rejected.

##### Edge cases

- One requested grasp.
- One diffusion step.
- Single-point observation.
- Batched observations.
- CPU-only execution.

##### Interaction with surrounding code

Simulation and evaluation phases consume the generated grasps file.

##### Existing pattern being followed

Inference orchestrates model loading and sampling but does not depend on simulation or robotics.

---

## 9. Data Flow

Phase 4 introduces the following data flows.

### Supervised training data flow

`Phase 3 dataset root -> Phase 3 index and record loading -> Phase 4 training function -> Generative model -> Loss -> Optimizer -> Checkpoint file`

Input source:

- Processed dataset root produced by Phase 3.

Transformation:

- Point clouds are loaded.
- Target SE(3) grasps are loaded.
- Target grasps are converted into the model training representation.
- The generative model is trained.

Validation:

- Dataset existence.
- Target grasp availability.
- SE(3) target validity.
- Finite loss values.

Output:

- Checkpoint file containing model state and architecture metadata.

Consumers:

- Grasp inference.
- Later evaluation pipelines.

### Inference data flow

`Observation file -> Phase 3 perception preprocessing if required -> Phase 4 checkpoint loading -> Generative sampling -> SE(3) conversion -> Generated grasps file`

Input source:

- Point-cloud observation file.
- Trained checkpoint.

Transformation:

- Observation is validated and preprocessed.
- Model samples grasp representations.
- Grasp representations are converted to homogeneous transforms.

Validation:

- Observation shape.
- Checkpoint compatibility.
- Rotation validity.

Output:

- Generated grasps file containing homogeneous transforms.

Consumers:

- Simulation execution in later phases.
- Evaluation in later phases.

---

## 10. Execution Flow

Phase 4 uses the existing script entry points.

### Training execution path

`scripts/train.py -> Unverified existing supervised training pipeline or training function -> Phase 3 data loading -> Phase 4 model -> Checkpoint file`

Phase 4 implements the missing behavior in the appropriate existing unit. It does not implement simulation or evaluation.

### Inference execution path

`scripts/generate_grasps.py -> Unverified existing grasp inference pipeline or inference function -> Phase 4 model -> Generated grasps file`

Phase 4 implements the missing behavior in the appropriate existing unit. It does not execute grasps in simulation.

### Direct validation path

For Phase 4 validation, tests may directly call the located model, training, and inference skeleton units. This avoids depending on unimplemented downstream behavior.

### Error propagation

Errors must propagate naturally:

- Missing dataset files propagate filesystem errors.
- Invalid target transforms propagate value errors.
- Invalid checkpoint metadata propagate value errors.
- Invalid observation shapes propagate value errors.

Phase 4 must not swallow errors silently.

---

## 11. Configuration Changes

Phase 4 requires no YAML configuration additions.

### Existing configuration preserved

| File | Key area | Existing semantics | Required Phase 4 treatment | Consumed by |
| --- | --- | --- | --- | --- |
| `configs/model.yaml` | Generative model type, steps, hidden dimensions | Documented model architecture settings. | Preserve file and documented semantics. Do not introduce global loading. | Future orchestration if it explicitly consumes configuration. |

### Configuration changes not allowed

Phase 4 must not:

- Add a new YAML schema.
- Add environment-variable configuration.
- Add global config state.
- Add config parser utilities.
- Add default values in code that shadow configuration values unless the existing skeleton already defines such defaults.

Hyperparameters must be passed explicitly through the existing CLI or pipeline contract.

---

## 12. Test Strategy

Testing must use `pytest`, which is already a development dependency.

No existing test suite was verified. Therefore, the implementation engineer must first check whether a test directory and existing conventions are present.

If tests already exist, Phase 4 tests must follow the existing conventions for:

- File naming.
- Test function naming.
- Fixture usage.
- Assertion style.
- Parameterization.
- Mocking, if already used.

If no tests exist, Phase 4 may add the minimum required test file under a conventional unit-test location. This is a test-only addition.

### Testing principles

1. Tests must be deterministic enough for CI.
2. Tests must not require network access.
3. Tests must not require downloaded datasets.
4. Tests must not require YCB assets.
5. Tests must not require GPUs.
6. Tests must not require simulation.
7. Tests must not require real robot hardware.
8. Tests must not introduce global state.
9. Tests must not create helper functions or helper classes.
10. Tests must use tiny model dimensions and tiny datasets.

### Asset strategy

Because external datasets are not guaranteed, tests must use temporary synthetic data.

The preferred approach is:

- Create a temporary dataset root.
- Create minimal point-cloud arrays inside the test scope.
- Create minimal target SE(3) grasp arrays inside the test scope.
- Use the Phase 3 index contract.
- Avoid committing binary assets.
- Avoid creating shared helper fixture code unless an existing fixture already provides the same behavior.

---

## 13. Test Suite

The following tests are required. Rows that depend on unverified skeleton symbols are marked accordingly.

| Test | File | Target | Scenario | Expected Result | Regression Risk |
| --- | --- | --- | --- | --- | --- |
| `test_phase1_package_import_remains_stable` | Existing or minimum required Phase 4 unit test file | `src/grasping_ai/__init__.py` | Import `grasping_ai`. | Import succeeds. | High if Phase 4 changes package initialization. |
| `test_model_config_file_exists` | Existing or minimum required Phase 4 unit test file | `configs/model.yaml` | Check file existence. | File exists. | Low. |
| `test_generative_model_forward_shape` | Existing or minimum required Phase 4 unit test file | Unverified generative model unit | Provide tiny point-cloud batch, noisy grasp batch, and step batch. | Output shape matches expected grasp representation dimension. | High. |
| `test_generative_model_rejects_invalid_point_cloud_shape` | Existing or minimum required Phase 4 unit test file | Unverified generative model unit | Provide invalid point-cloud shape. | Raises value or type error. | Medium. |
| `test_generative_model_rejects_non_finite_input` | Existing or minimum required Phase 4 unit test file | Unverified generative model unit | Provide NaN or infinite values. | Raises value error. | Medium. |
| `test_supervised_training_loss_is_finite` | Existing or minimum required Phase 4 unit test file | Unverified training function or diffusion loss behavior | Run one training step on a tiny synthetic batch. | Loss is finite. | High. |
| `test_training_creates_checkpoint` | Existing or minimum required Phase 4 unit test file | Unverified training function | Train for one tiny epoch on synthetic data. | Checkpoint file exists and contains required metadata. | High. |
| `test_training_rejects_missing_dataset` | Existing or minimum required Phase 4 unit test file | Unverified training function | Provide missing dataset root. | Raises filesystem or value error. | Medium. |
| `test_training_rejects_dataset_without_targets` | Existing or minimum required Phase 4 unit test file | Unverified training function | Provide point clouds without target grasps. | Raises value error. | Medium. |
| `test_checkpoint_roundtrip` | Existing or minimum required Phase 4 unit test file | Unverified checkpoint saving/loading behavior | Save a tiny checkpoint and load it. | Loaded metadata and parameter state are compatible. | High. |
| `test_generate_grasps_output_shape_single_observation` | Existing or minimum required Phase 4 unit test file | Unverified inference function | Generate grasps from a single synthetic observation. | Output array shape is `K x 4 x 4`. | High. |
| `test_generate_grasps_rotations_are_valid` | Existing or minimum required Phase 4 unit test file | Unverified inference function | Generate grasps and inspect rotation matrices. | Rotation matrices are orthonormal with determinant near one. | High. |
| `test_generate_grasps_rejects_invalid_checkpoint` | Existing or minimum required Phase 4 unit test file | Unverified inference function | Provide missing or corrupted checkpoint. | Raises filesystem or value error. | Medium. |
| `test_generate_grasps_rejects_invalid_observation_shape` | Existing or minimum required Phase 4 unit test file | Unverified inference function | Provide invalid observation array shape. | Raises value error. | Medium. |
| `test_model_inference_is_repeatable_without_global_state` | Existing or minimum required Phase 4 unit test file | Unverified model and inference units | Run inference twice with the same seeded local state. | Outputs are consistent under the same local seeding contract. | Medium. |

---

## 14. Regression Test Plan

Phase 4 can introduce regressions in package imports, data contracts, configuration, and downstream simulation input expectations.

### Unit-level regressions

| Existing behavior | Why it could break | Protecting test | Expected result |
| --- | --- | --- | --- |
| Package import succeeds. | Model, training, or inference modules may introduce import cycles or heavy imports. | `test_phase1_package_import_remains_stable` | Import succeeds. |
| Phase 3 data loading remains stable. | Training may pressure changes into dataset loading. | Phase 3 data tests and Phase 4 training tests. | Phase 3 behavior remains unchanged. |
| SE(3) math remains pure. | Grasp conversion may introduce hidden global state. | Phase 1 SE(3) tests, if present. | Existing Phase 1 behavior remains intact. |

### Integration-level regressions

| Existing behavior | Why it could break | Protecting test | Expected result |
| --- | --- | --- | --- |
| Scripts can import pipeline modules. | Training or inference modules may be imported indirectly and cause errors. | Existing import-related tests, if present. | Import path remains valid. |
| Downstream simulation expects generated grasps as `.npy` homogeneous transforms. | Inference output contract may change. | Generated grasp shape and rotation validity tests. | Output contract remains stable. |

### Pipeline-level regressions

No full evaluation or simulation pipeline behavior is verified yet. Therefore, no pipeline-level regression test can be specified without invention.

If an existing pipeline smoke test exists, it must be rerun. If no such test exists, Phase 4 must not create a broad pipeline smoke test.

---

## 15. Impact on Preceding Phases

### Phase 1 — Foundation & Math Primitives

| Item | Description |
| --- | --- |
| Existing contract | Phase 1 established package importability, dependency availability, static configuration preservation, and pure SE(3) math behavior where verified. |
| What Phase 4 changes | Phase 4 adds model, training, and inference behavior that may consume SE(3) math and Torch dependencies. |
| What remains compatible | Package layout, existing dependencies, static configuration files, and Phase 1 math behavior must remain unchanged. |
| Previous-phase tests to rerun | Phase 1 package import test, dependency tests, SE(3) math tests if present. |
| Whether Phase 1 needs modification | No modification is expected unless dependency resolution fails. |

### Phase 2 — Simulation & Robotics Core

| Item | Description |
| --- | --- |
| Existing contract | Phase 2 established simulation and robotics behavior where verified. |
| What Phase 4 changes | Phase 4 does not modify simulation or robotics behavior directly. |
| What remains compatible | Simulation and robotics modules must remain independent of generative model code. |
| Previous-phase tests to rerun | Phase 2 dependency and simulation tests if present. |
| Whether Phase 2 needs modification | No modification is expected. |

### Phase 3 — Data Pipeline & Perception

| Item | Description |
| --- | --- |
| Existing contract | Phase 3 established dataset indexing, point-cloud loading, and perception preprocessing. |
| What Phase 4 changes | Phase 4 consumes Phase 3 data contracts for supervised training and inference observations. |
| What remains compatible | Phase 3 data loaders and perception behavior must remain low-level and unchanged unless a verified incompatibility exists. |
| Previous-phase tests to rerun | Phase 3 data loading, index loading, perception preprocessing, and batching tests if present. |
| Whether Phase 3 needs modification | No modification is expected unless the dataset target contract is verified to be incompatible. |

---

## 16. Impact on Downstream Phases

### Downstream Phase 5 — Reinforcement Learning Policy

| Item | Description |
| --- | --- |
| Dependency | Phase 5 may later use generated grasps as candidate actions or initialization states. |
| Interface relied on | Generated grasp output shape and SE(3) validity. |
| New behavior available after Phase 4 | Valid grasp candidates can be generated from point clouds. |
| Constraints preserved | Generated grasps must remain valid homogeneous transforms. |
| Protecting tests | Generated grasp shape and rotation validity tests. |

### Downstream Phase 6 — End-to-End Orchestration & Evaluation

| Item | Description |
| --- | --- |
| Dependency | Phase 6 depends on generated grasps for simulation execution and evaluation. |
| Interface relied on | Generated grasps file, checkpoint contract, and explicit inference arguments. |
| New behavior available after Phase 4 | Generated grasps can be passed to later simulation execution. |
| Constraints preserved | Output must be `.npy` homogeneous transforms with stable shape. |
| Protecting tests | Inference output tests and SE(3) validity tests. |

---

## 17. Cross-Phase Contract

Phase 4 establishes the following contract.

### Inputs

1. Training:
   - Processed dataset root.
   - Checkpoint output path.
   - Explicit model hyperparameters.
   - Explicit training hyperparameters.
   - Device identifier.

2. Inference:
   - Checkpoint path.
   - Observation path.
   - Output path.
   - Number of diffusion steps.
   - Number of grasps.
   - Device identifier.

3. Model:
   - Point-cloud tensor.
   - Noisy grasp representation tensor.
   - Diffusion step tensor.

### Outputs

1. Training:
   - Checkpoint file containing model state and architecture metadata.

2. Inference:
   - Generated grasps file.
   - Single observation output shape: `K x 4 x 4`.
   - Batched observation output shape: `B x K x 4 x 4`.

3. Model:
   - Predicted generative quantity with shape compatible with the grasp representation.

### Expected behavior

1. Training consumes Phase 3 data without modifying Phase 3 modules.
2. Inference loads checkpoints explicitly.
3. Generated grasps are valid SE(3) homogeneous transforms.
4. Model behavior does not depend on simulation or robotics.
5. Training and inference do not introduce global state.

### Invariants

1. No module-level mutable state.
2. No global configuration state.
3. No dependency on simulation, robotics, or evaluation inside model code.
4. Rotation matrices in generated grasps are orthonormal.
5. Checkpoint metadata is sufficient to reconstruct inference-compatible model settings.

### Error behavior

1. Missing dataset files raise explicit filesystem or value errors.
2. Missing target grasps raise value errors.
3. Invalid target SE(3) transforms raise value errors.
4. Invalid checkpoint metadata raises value errors.
5. Invalid observation shapes raise value errors.

### Configuration assumptions

1. YAML files remain static.
2. Phase 4 does not parse YAML globally.
3. Hyperparameters are explicit.

### Data assumptions

1. Point clouds are numeric arrays with shape `N x 3` or batched equivalents.
2. Target grasps are SE(3) homogeneous transforms.
3. Observations are stored as `.npy` arrays.
4. Generated grasps are stored as `.npy` arrays.

### Performance assumptions

1. CPU execution must be supported for tiny tests.
2. GPU execution may be supported but is not required for validation.
3. Distributed training is out of scope.
4. Tests must use tiny dimensions and tiny datasets.

---

## 18. Validation Before Commit

The implementation must not be considered ready for commit until the following sequence is complete.

1. Run Phase 4 package import regression tests.
2. Run Phase 4 configuration existence tests.
3. Run model forward shape tests.
4. Run model invalid input tests.
5. Run supervised training loss tests.
6. Run training checkpoint creation tests.
7. Run checkpoint round-trip tests.
8. Run inference output shape tests.
9. Run generated grasp SE(3) validity tests.
10. Run inference failure tests.
11. Rerun Phase 1 tests.
12. Rerun Phase 3 tests.
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

The Phase 4 commit must contain only the following.

### Allowed in the commit

1. Modifications to verified existing generative model skeleton units.
2. Modifications to verified existing supervised training skeleton units.
3. Modifications to verified existing grasp inference skeleton units.
4. Modifications to verified existing script entry behavior where required to preserve thin CLI delegation.
5. Required Phase 4 tests.
6. Minimal test-only synthetic dataset artifacts created inside test scope.

### Not allowed in the commit

1. New model base classes.
2. New trainer classes.
3. New inference wrapper classes.
4. New helper functions.
5. New utility modules.
6. New global constants.
7. New global variables.
8. New configuration loaders.
9. Unrelated formatting changes.
10. Refactoring of scripts unrelated to Phase 4.
11. Simulation code.
12. Evaluation code.
13. RL training code.
14. Temporary debugging code.
15. Experimental code.

---

## 20. Implementation Risks

| Risk | Severity | Cause | Affected Component | Detection Method | Mitigation |
| --- | --- | --- | --- | --- | --- |
| Exact model skeleton file or symbol is unverified. | High | Repository snapshot did not expose source details under `src/grasping_ai/models/`. | Generative model implementation. | Manual source inspection before implementation. | Modify only located existing units. If absent, pause for scope decision. |
| Exact training skeleton file or symbol is unverified. | High | Repository snapshot did not expose source details under `src/grasping_ai/training/`. | Supervised training implementation. | Manual source inspection before implementation. | Modify only located existing units. If absent, pause for scope decision. |
| Exact inference skeleton file or symbol is unverified. | High | Repository snapshot did not expose source details under `src/grasping_ai/inference/`. | Grasp inference implementation. | Manual source inspection before implementation. | Modify only located existing units. If absent, pause for scope decision. |
| Creating a new model class would violate constraints. | High | PyTorch models often use classes. | Model implementation. | Structural review. | Use existing class or model constructor. If none exists, escalate. |
| Grasp rotation representation mismatch. | High | SE(3) output must be valid for later simulation. | Inference and downstream evaluation. | Rotation validity tests. | Use explicit 6D rotation conversion or preserve existing skeleton contract. |
| Checkpoint metadata mismatch. | High | Inference may not know hidden dimensions or model type. | Inference. | Checkpoint round-trip tests. | Store explicit architecture metadata in checkpoint. |
| Dataset target format mismatch. | High | Phase 3 target loading may be optional or unverified. | Training. | Training tests with synthetic targets. | Require explicit target SE(3) arrays for supervised training. |
| Diffusion step contract mismatch between training and inference. | Medium | Training arguments and inference arguments differ in documentation. | Generative model. | Training and inference tests. | Store generative metadata in checkpoint and validate at inference. |
| Tests become slow due to Torch training. | Medium | Model training can be heavy. | Test suite. | Use tiny dimensions and tiny datasets. | Keep tests minimal and CPU-only. |

---

## 21. Design Decisions

| Decision | Evidence | Alternatives Considered | Reason for Selection |
| --- | --- | --- | --- |
| Use diffusion as the minimum generative contract if the skeleton does not specify otherwise. | `docs/USAGE.md` includes `--num-diffusion-steps` and `configs/model.yaml` mentions diffusion. | Implement flow matching first. | Diffusion has clearer documented evidence for MVP. |
| Keep scripts thin and delegate to existing pipeline or training/inference functions. | Repository documentation and verified `scripts/train_rl.py` pattern. | Implement logic directly in scripts. | Preserves existing architecture. |
| Use explicit CLI arguments for model and training hyperparameters. | `docs/USAGE.md` shows explicit arguments. | Parse YAML configuration globally. | Avoids global configuration and unverified YAML parsing. |
| Store explicit architecture metadata in checkpoints. | Inference arguments include feature dimension but not all architecture details. | Infer architecture from checkpoint only. | Prevents inference mismatch. |
| Output generated grasps as homogeneous transforms. | Later simulation and evaluation require SE(3) grasp poses. | Output quaternions or raw network parameters. | Stable robotics contract. |
| Use 6D rotation representation for internal model output if no existing representation is verified. | 6D representations avoid quaternion normalization issues and support deterministic conversion to rotation matrices. | Output quaternions or Euler angles. | More stable for training and conversion. |
| Do not parse YAML configuration in Phase 4. | No config loader verified; YAML parser dependency not verified. | Add global config loader. | Avoids global state and unnecessary dependency. |
| Do not implement flow matching unless already required by the skeleton. | Flow is documented but not fixed. | Implement both diffusion and flow. | Keeps phase minimal. |

---

## 22. Explicitly Rejected Changes

The following changes were considered and rejected.

### Rejected: creating a new model class

A new model class might simplify PyTorch implementation, but it violates the no-new-class constraint. If an existing model class or model constructor exists, it must be used. If none exists, the phase must be paused.

### Rejected: creating a trainer class

A trainer class would organize epochs, logging, and checkpointing, but it violates the no-new-class constraint and is unnecessary for the minimum supervised behavior.

### Rejected: creating helper functions for rotation conversion

Rotation conversion is required, but creating separate helper utilities would violate the no-helper constraint. Required conversion behavior must be implemented directly inside the appropriate existing training or inference unit.

### Rejected: adding a global configuration loader

A config loader would make `configs/model.yaml` easier to use, but Phase 4 does not require global configuration. It would introduce hidden state and unnecessary dependency risk.

### Rejected: adding experiment tracking integration

TensorBoard and W&B dependencies exist, but Phase 4 does not require experiment dashboards. Adding tracking would expand scope and introduce side effects.

### Rejected: implementing grasp ranking or filtering

Phase 4 only generates candidate grasps. Ranking, filtering, and collision checking belong to later evaluation or simulation phases.

### Rejected: implementing flow matching in parallel with diffusion

Flow matching is documented as a possible approach, but implementing both would expand scope. The phase should implement the existing skeleton contract or the minimum evidenced generative behavior.

### Rejected: implementing distributed training

Distributed training is an optimization and infrastructure concern. It is not required for Phase 4 correctness.

---

## 23. Verification Evidence

| Claim | Evidence |
| --- | --- |
| Repository uses `hatchling` and `src` layout. | Verified `pyproject.toml`. |
| Python requirement is `>=3.12`. | Verified `pyproject.toml`. |
| Torch is a runtime dependency. | Verified `pyproject.toml`. |
| Models, training, and inference modules are intended. | README repository layout. |
| Generative model types include diffusion and flow. | README project description and `docs/PROJECT.md`. |
| Supervised training script exists. | Repository script table and `docs/USAGE.md`. |
| Grasp generation script exists. | Repository script table and `docs/USAGE.md`. |
| Training arguments include dataset root, checkpoint, feature dimension, hidden dimension, layers, learning rate, epochs, batch size, and device. | `docs/USAGE.md`. |
| Inference arguments include checkpoint, observations, output, feature dimension, diffusion steps, number of grasps, and device. | `docs/USAGE.md`. |
| Generated grasps are consumed by simulation execution. | Repository script table lists `scripts/run_simulation.py` executing generated grasps. |
| Exact model source files are unverified. | Available snapshot did not expose contents of `src/grasping_ai/models/`. |
| Exact training source files are unverified. | Available snapshot did not expose contents of `src/grasping_ai/training/`. |
| Exact inference source files are unverified. | Available snapshot did not expose contents of `src/grasping_ai/inference/`. |
| Existing tests are unverified. | Available snapshot did not expose a test suite. |

---

## 24. Definition of Done

Phase 4 is considered complete only when:

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

- [ ] Verified exact generative model skeleton file and symbol.
- [ ] Verified exact supervised training skeleton file and symbol.
- [ ] Verified exact grasp inference skeleton file and symbol.
- [ ] Verified checkpoint saving/loading behavior location.
- [ ] Implemented generative model forward behavior.
- [ ] Implemented supervised training behavior.
- [ ] Implemented diffusion loss or equivalent generative loss behavior.
- [ ] Implemented checkpoint saving behavior.
- [ ] Implemented checkpoint loading behavior.
- [ ] Implemented grasp sampling behavior.
- [ ] Implemented SE(3) conversion behavior inside the appropriate existing unit.
- [ ] Preserved thin CLI behavior for training and inference scripts.
- [ ] Did not create new classes.
- [ ] Did not create helper functions.
- [ ] Did not create utility modules.
- [ ] Did not introduce global variables.
- [ ] Did not introduce global constants.
- [ ] Did not add file-level descriptions.
- [ ] Added Google-style docstrings to modified functions or methods.

### Tests

- [ ] Added package import regression test.
- [ ] Added model configuration existence test.
- [ ] Added model forward shape test.
- [ ] Added model invalid point-cloud shape test.
- [ ] Added model non-finite input test.
- [ ] Added supervised training loss test.
- [ ] Added training checkpoint creation test.
- [ ] Added missing dataset failure test.
- [ ] Added missing target failure test.
- [ ] Added checkpoint round-trip test.
- [ ] Added inference output shape test.
- [ ] Added generated grasp rotation validity test.
- [ ] Added invalid checkpoint failure test.
- [ ] Added invalid observation shape failure test.
- [ ] Added repeatable inference test under local seeding.

### Regression Verification

- [ ] Phase 1 package import remains successful.
- [ ] Phase 1 dependency tests remain successful.
- [ ] Phase 1 SE(3) tests remain successful if present.
- [ ] Phase 3 data loading tests remain successful.
- [ ] Phase 3 perception tests remain successful.
- [ ] No import cycles were introduced.
- [ ] No module-level side effects were introduced.

### Cross-Phase Verification

- [ ] Phase 5 can rely on generated SE(3) grasp candidates.
- [ ] Phase 6 can rely on generated grasp `.npy` output contract.
- [ ] Model modules do not import simulation, robotics, or evaluation.
- [ ] Training modules orchestrate data and models without modifying Phase 3 loaders.
- [ ] Inference modules load checkpoints explicitly and do not depend on training state.

### Structural Constraints

- [ ] No new model base class exists.
- [ ] No new trainer class exists.
- [ ] No new inference wrapper class exists.
- [ ] No helper abstraction exists.
- [ ] No utility abstraction exists.
- [ ] No global configuration abstraction exists.
- [ ] No unrelated refactoring exists.
- [ ] No experimental code exists.

### Commit Readiness

- [ ] Targeted Phase 4 tests pass.
- [ ] Regression tests pass.
- [ ] Broader relevant test suite passes.
- [ ] Static and structural review confirms forbidden constructs are absent.
- [ ] Commit contains only Phase 4 changes.