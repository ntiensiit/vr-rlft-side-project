# ADR-0004 — Dead-helper wiring, deduplication, and dependency retention

## Status

Accepted (2026-08-12).

## Context

A refactoring audit identified four classes of cleanup work:

1. **Dead helpers** — functions exported or tested but not called from production paths.
2. **Duplicate code** — identical SE(3) conversions, training-pair construction,
   inference conditioning blocks, and checkpoint I/O repeated across pipelines.
3. **Refactor cleanup** — unused loss-builder parameters, scene XML output paths
   under `data/interim`, and bloated submodule re-exports.
4. **Dependencies** — `theseus` had no production imports; `pytransform3d` was
   flagged for dev-only relocation despite runtime use.

User direction: **"check where them should be used, dont delete them"** — retain
audited helpers and wire them into production rather than deleting dead code.

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
| Simulation | `step_scene`, `load_gripper_model`, `make_close_command`, `build_forward_kinematics` → `pipelines/simulate_grasp.py` |
| Generation | `build_generation_pipeline` → `pipelines/generate_grasps.py` |
| Evaluation | `filter_collision_free_grasps`, `load_contact_set`, `compute_grasp_wrench_matrix`, `aggregate_grasp_success_rate` → `pipelines/evaluate.py`, `scripts/evaluate.py`, `run_workflow.py` |
| Training | `load_pretrained_encoder`, `invert_rigid_transform`, `build_grasp_pose_regression_loss` → `pipelines/train.py`, `pipelines/train_flow.py` |
| Perception/data | `sample_point_cloud`, `farthest_point_sampling`, `voxel_downsample` → `scripts/prepare_data.py`; `merge_point_clouds`, `normalize_point_cloud` → `scripts/prepare_observations.py`; transform helpers → `data/transforms.py`; `invert_transform` → `models/equivariant_encoder.py` |
| Frames | `convert_grasps_to_world_frame`, `identity_transform` → `scripts/run_simulation.py` |
| RL | `build_value_network` → `pipelines/train_rl.py`; `select_action` → `inference/policy_runner.py`, `scripts/run_rl_evaluation.py` |

### Duplicate code extraction

| Extraction | Module |
| --- | --- |
| `se3_to_vec`, `vec_to_se3` | `data/grasp_vector.py` |
| `build_supervised_training_pairs` | `data/training_pairs.py` |
| `prepare_point_cloud_tensor`, `encode_grasp_conditioning`, `sample_to_world_frame` | `inference/grasp_sampling.py` |
| `load_torch_checkpoint`, `read_model_checkpoint_metadata` | `training/checkpoint_io.py` |

Callers updated: `pipelines/train.py`, `pipelines/train_flow.py`,
`inference/grasp_generator.py`, `training/trainer.py`, `inference/policy_runner.py`.

### Refactor cleanup

- **Loss builders:** `build_diffusion_score_loss()` and
  `build_flow_matching_loss()` no longer take unused model arguments.
- **Scene XML output:** `build_scene_xml`, `attach_object_to_scene`, and
  `MuJoCoScene` accept optional `output_dir` / `scene_output_dir`; default is
  `$TMPDIR/grasping_ai_scenes` instead of repo-relative `data/interim`.
- **`__init__.py` re-exports:** consolidated; removed test-only public exports
  `build_gripper_controller` and `make_open_command` from `robotics/__init__.py`
  (helpers remain in `robotics/gripper.py`).

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
- **Do not delete** wired helpers unless this ADR is explicitly superseded.
- New helpers should follow the same pattern: wire into a production caller or
  document why they remain test-only.

## Verification state (2026-08-12)

Three-stage distinction:

| Stage | Status |
| --- | --- |
| Software pipeline correctness | **Verified** — `scripts/run_artifacts.py` end-to-end on a clean tree; `tests/test_artifact_chain.py` (slow). |
| Learning pipeline execution | **Verified mechanically** — training, checkpoints, inference execute; flow joint-encoder contract verified structurally (`tests/test_phase4_flow_training.py`). |
| Robotics / research outcome | **Not verified** — recorded run had 8/8 IK failures and 0/0 physical grasp successes (2-DOF robot reachability + undertrained diffusion). Research-stage, not a pipeline defect. |

Gate: `uv run pytest -q`, `uv run ruff check src tests scripts`, `uv run mypy src`
— 219 passed, ~87% coverage (2026-08-12).

## Follow-up review triggers

Revisit when:

- the wiring policy is reversed (explicit deletion of helpers is requested);
- a dependency is replaced (e.g. swap PyPI `theseus` for a robotics optimizer);
- duplicate patterns reappear across new pipelines without using shared modules.
