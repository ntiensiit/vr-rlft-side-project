# ADR-0004 — Dead-helper wiring, deduplication, and dependency retention

## Status

Accepted (2026-08-12). Updated 2026-08-16 — explicit dead-code removal
supersedes the earlier helper-retention policy.

## Context

A refactoring audit identified four classes of cleanup work:

1. **Dead helpers** — functions exported or tested but not called from production paths.
2. **Duplicate code** — identical SE(3) conversions, training-pair construction,
   inference conditioning blocks, and checkpoint I/O repeated across pipelines.
3. **Refactor cleanup** — unused loss-builder parameters, scene XML output paths
   under `data/interim`, and bloated submodule re-exports.
4. **Dependencies** — `theseus` had no production imports; `pytransform3d` was
   flagged for dev-only relocation despite runtime use.

The earlier direction was to retain audited helpers and wire them into
production. The current cleanup request explicitly authorizes removing helpers
that remain test-only after the wiring audit.

## Decision

### Dead-helper wiring

Wire semantically equivalent production call sites to existing helpers. No
behavior change; no tests weakened or removed.

**First wiring pass**

| Helper | Module | Production caller(s) |
| --- | --- | --- |
| `build_kdtree` | `perception/pointcloud.py` | `data/pointcloud_dataset.py::generate_analytical_grasps`, `evaluation/collision.py::{build_collision_checker, generate_analytical_contacts}` |
| `transform_between_frames` | `robotics/transforms.py` | `evaluation/collision.py::{build_collision_checker, generate_analytical_contacts}` |
| `make_transform` | `perception/geometry.py` | `robotics/kinematics.py::build_forward_kinematics`, `simulation/mujoco_env.py::read_body_pose`, `data/pointcloud_dataset.py::generate_analytical_grasps`, `data/transforms.py::make_random_rotation_jitter` |
| `transform_grasp_pose` | `robotics/transforms.py` | `data/transforms.py::make_random_rotation_jitter` |
| `generate_candidate_grasps` | `inference/grasp_generator.py` | `scripts/run_grasp_inference.py` |
| `acquire_point_cloud_stream` | `sensors/pointcloud_sensor.py` | `scripts/generate_grasps.py` |

**Second wiring pass**

| Area | Helpers wired |
| --- | --- |
| Simulation | `step_scene`, `load_gripper_model`, `make_open_command`, `make_close_command`, `build_forward_kinematics` → `pipelines/simulate_grasp.py` |
| Generation | `generate_candidate_grasps` → `pipelines/generate_grasps.py`, `scripts/generate_grasps.py` |
| Evaluation | `filter_collision_free_grasps`, `load_contact_set`, `compute_grasp_wrench_matrix`, `aggregate_grasp_success_rate` → `pipelines/evaluate.py`, `scripts/evaluate.py`, `run_workflow.py` |
| Training | `load_pretrained_encoder`, `invert_rigid_transform` → `pipelines/train_flow.py` |
| Perception/data | `sample_point_cloud`, `farthest_point_sampling`, `voxel_downsample` → `scripts/prepare_data.py`; `merge_point_clouds`, `normalize_point_cloud` → `scripts/prepare_observations.py`; transform helpers → `data/transforms.py`; `invert_transform` → `models/equivariant_encoder.py` |
| Frames | `convert_grasps_to_world_frame`, `identity_transform` → `scripts/run_simulation.py` |
| RL | `select_action` → `inference/policy_runner.py`, `scripts/run_rl_evaluation.py` |

### Duplicate code extraction

| Extraction | Module |
| --- | --- |
| `se3_to_vec`, `vec_to_se3` | `data/grasp_vector.py` |
| `SupervisedGraspDataset`, `validate_grasp_dataset` | `data/training_pairs.py` |
| `prepare_point_cloud_tensor`, `encode_grasp_conditioning`, `sample_to_world_frame` | `inference/grasp_sampling.py` |
| `load_torch_checkpoint`, `read_model_checkpoint_metadata`, `checkpoint_scalar_int`, `checkpoint_dict_int` | `training/checkpoint_io.py` |
| `SupervisedTrainingDataloader`, `ConditionedTrainingDataloader` | `pipelines/supervised_training.py` |
| `batch_conditioned_grasp_samples` | `models/grasp_sampling_batch.py` |

Callers updated: `pipelines/train_diffusion.py`, `pipelines/train_flow.py`,
`inference/grasp_generator.py`, `training/trainer.py`, `inference/policy_runner.py`,
`models/diffusion.py`, `models/flow.py`.

### Refactor cleanup

- **Loss builders:** `build_diffusion_score_loss()` and
  `build_flow_matching_loss()` no longer take unused model arguments.
- **Scene XML output:** `build_scene_xml`, `attach_object_to_scene`, and
  `MuJoCoScene` accept optional `output_dir` / `scene_output_dir`; default is
  `$TMPDIR/grasping_ai_scenes` instead of repo-relative `data/interim`.
- **`__init__.py` re-exports:** consolidated; augmentation and dataset
  iterators remain available only from their owning modules.
- **SE(3) helpers:** `invert_rigid_transform` delegates to `invert_transform`;
  `transform_between_frames` delegates to `apply_transform`.
- **Removed dead surface:** `FrameConversion` alias, `flow_field` compat dict key,
  metadata-only regression baseline in `pipelines/train.py`, redundant final
  checkpoint write in `train_flow.py`, obsolete notebooks (see
  `notebooks/archive/README.md`), and the remaining test-only
  `build_gripper_controller`, `build_score_network`, and `build_flow_field`
  factories.

### 2026-08-13 follow-up (code audit completion)

The helpers below were wired into production rather than deleted under the
original policy:

| Helper / flag | Production caller |
| --- | --- |
| `make_open_command` | `pipelines/simulate_grasp.py` (pre-grasp open phase) |
| Augmentation transforms | `SupervisedGraspDataset(augment=True)` + `training.augment` on train CLIs |
| `iterate_grasp_dataset` | `validate_grasp_dataset()` in diffusion/flow pipelines |
| `load_training_checkpoint` | `training.resume` on `scripts/train_diffusion.py` / `scripts/train_flow.py` |

Flow inference unified on `FlowGeneratorModel` via `load_flow_model_from_state`
(`models/flow.py`). All 30 code-audit checklist items closed; open work is CI
first-run and research validation only (`CHECKLIST.md`).

### Dependency retention and wiring

Do **not** remove `theseus` or demote `pytransform3d` to dev-only.

- **`theseus` (PyPI 0.0.5):** a text-vocabulary library (`Node`,
  `default_tokenizer`), not Meta's robotics optimizer. Wired for YCB object-name
  alias resolution in `simulation/ycb.py` (`tokenize_ycb_object_name`,
  `build_ycb_object_name_classifier`); `data/pointcloud_dataset.py` delegates
  directory resolution to shared YCB helpers.
- **`pytransform3d`:** expanded runtime use — IK pose error in
  `robotics/kinematics.py` (`pt.concat`, `pr.compact_axis_angle_from_matrix`);
  point-frame transforms in `robotics/transforms.py` (`pt.transform`,
  `pt.vectors_to_points`). Grasp-pose composition keeps matrix `@` because
  learned 4×4 outputs are not guaranteed proper rotation matrices.

## Rationale

- Wiring preserves tested API surface and avoids silent deletion of helpers
  referenced by phase tests and docs.
- Shared modules reduce drift between diffusion and flow training/inference paths.
- Scene output to `$TMPDIR` avoids polluting the repo with temporary MJCF files.
- Dependencies declared in phase docs remain justified by production import chains.

## Consequences

- Refactoring implementation work is closed; open items are research validation
  (see `CHECKLIST.md`).
- Wired helpers remain part of the production API. Test-only helpers without a
  production caller should be removed during cleanup rather than retained
  solely for isolated tests.

## Verification state (2026-08-13)

| Stage | Status |
| --- | --- |
| Software pipeline correctness | **Verified** — `scripts/run_artifacts.py` end-to-end; `tests/test_artifact_chain.py` (slow). |
| Learning pipeline execution | **Verified mechanically** — training, checkpoints, inference, resume, augmentation paths execute. |
| Robotics / research outcome | **Not verified** — Panda reachability and undertrained diffusion remain research-stage limits. |

Gate: see `CHECKLIST.md` (2026-08-16 local: 320 fast tests passed; coverage 84% vs 97% gate; mypy not clean).

### 2026-08-16 follow-up

- CLI argparse wrappers removed; Hydra `@hydra.main` + `configs/scripts/`.
- Dataset records are pickle-free `.npz`.
- Package re-exports slimmed; supervised CLI helper is `pipelines/supervised_training_script.py`.
- Robot viewer uses MuJoCo's built-in UI (no UDP keyboard TUI).
- Shared `models/mlp.py` and `models/grasp_sampling_batch.py` remain the MLP / sampling builders.

## Follow-up review triggers

Revisit when:

- a removed helper gains a production caller and must be restored or replaced;
- a dependency is replaced (e.g. swap PyPI `theseus` for a robotics optimizer);
- duplicate patterns reappear across new pipelines without using shared modules.
