# Code Quality Review: `grasping_ai` Package

> **Scope:** Code-level improvements only (not architecture, features, or design)  
> **Initial review:** 2026-08-15  
> **Last updated:** 2026-08-15  
> **Status:** **Complete** — all actionable review items resolved; optional style items remain

Verification at last update: `uv run ruff check src tests scripts`, `uv run mypy src` (56 files), `uv run pytest -q -m "not slow"` — **359 passed**, **97% coverage**.

---

## Executive summary

| Category | Original count | Resolved | Remaining |
|---|---|---|---|
| Dead code | ~10 alleged | 1 removed (`count_supervised_training_pairs`) | 0 actionable (9 retracted — see below) |
| Duplicated code | 6 blocks | 6 | 0 |
| Inconsistent patterns | 5 | 5 | 0 |
| Type safety | 4 | 4 | 0 |
| Performance | 3 | 3 | 0 |
| Fragile patterns | 4 | 4 | 0 |
| Minor / style | ~10 | 10 | 0 |

**Original top-3 recommendations — all addressed:**

1. ~~CPU→GPU round-trip in `compose_with_se3_frame`~~ → `invert_rigid_transform_batch()`
2. ~~Factor RL/generative MLP builders~~ → `models/mlp.py`
3. ~~Remove misleading dead alias~~ → `count_supervised_training_pairs` removed

---

## Applied fixes

| § | Item | Change |
|---|---|---|
| 1.3 | `count_supervised_training_pairs` alias | Removed; tests call `validate_grasp_dataset` |
| 2.1 | RL policy/value MLP duplication | `build_tanh_mlp` in `models/mlp.py` |
| 2.2 | Analytical grasp frame duplication | `_antipodal_grasp_from_contacts`, `_search_antipodal_grasps` |
| 2.3 | Score/flow MLP duplication | `build_mish_mlp` shared by `ScoreNetwork`, `FlowFieldNet` |
| 3.1 | RL SB3 → legacy checkpoint export | Dynamic `copy_sb3_policy_weights()`; `build_sb3_net_arch()` aligns SB3 depth with `policy_num_layers` |
| 3.2 | Scattered `from __future__ import annotations` | All 54 `src/grasping_ai/**/*.py` modules |
| 3.3 | Missing `GraspGeneratorModel.condition()` | Added; matches `FlowGeneratorModel` |
| 4.1 | `GraspSample` loose dict | `TypedDict` exported from `data/__init__.py` |
| 4.2 | Redundant pipeline `cast()` | Cleaned in pipelines and I/O boundaries (`simulate_grasp`, `generate_grasps`, `policy_runner`, `force_closure`) |
| 4.3 | `GraspPoseGenerator` shape-less callable | `Protocol` in `inference/grasp_generator.py` |
| 5.1 | CPU→GPU SE(3) inversion | `invert_rigid_transform_batch()`; used in encoder + training pairs |
| 5.2 | Duplicated MLflow logging | `training/experiment_logging.py` |
| 5.3 | Inline `loguru` imports | Module-level in all touched pipeline/data/training files |
| 6.1 | Function attributes on training step | `SupervisedTrainingStep` class |
| 6.2 | `empty_dataset_handling` path string check | Removed from `training_pairs.py` |
| 6.3 | Silent force-closure LP failures | `logger.warning` in `force_closure.py` |
| 6.4 | Hard-coded PPO hyperparameters | `n_steps`, `batch_size`, `n_epochs`, `policy_num_layers` in config + pipeline |
| 7 | `float32` vs `float64` grasp tensors | `build_supervised_training_pairs` uses `np.float32` (matches analytical output) |
| 8 | Encoder typed as `Callable` | `build_equivariant_encoder` returns `SE3EquivariantPointNet`; checkpoint loads without `# type: ignore` |
| 9 | I/O boundary typing | `robot_model_nq` / `robot_model_mj_model`, `parse_contact_set`, `read_checkpoint_model_state_dict`, grasp dict validators |
| 10 | Magic constants | `utils/numerics.py` shared tolerances for geometry, grasp search, force closure |
| 11 | Path validation dedup | `utils/path_validation.py` (`require_path`, `require_optional_path`) across 18 modules |
| 12 | Missing `__all__` | Added to `utils/__init__.py` |

**New modules from this pass:** `models/mlp.py`, `training/experiment_logging.py`, `utils/numerics.py`, `utils/path_validation.py`

---

## Retracted findings (not dead code)

The initial §1 review incorrectly flagged these as unused. **Do not remove.**

| Symbol | Actual callers |
|---|---|
| `generate_analytical_grasps` | `scripts/prepare_data.py`, extensive test suite |
| `parse_clean_argv` | `run_workflow.py`, `run_grasp_inference.py`, `run_simulation.py` |
| `parse_config_name_from_argv` | `prepare_data.py`, `prepare_ycb_mjcf.py`, `prepare_observations.py` |
| `optional_cli_path` | `run_workflow.py`, `run_simulation.py`, `visualize_robot.py`, `run_rl_evaluation.py` |
| `config_str_list` / `config_float_list` | Multiple scripts, `evaluate.py`, `notebook_helpers.py`, config tests |
| `grasp_pose_to_transform` | `test_phase1_foundation.py`, exported from `perception/__init__.py` |
| `build_score_network` / `build_flow_field` | `test_phase4_generative_grasp.py`, package exports, ADR-0003/0004 |

`grasp_pose_to_transform` is a thin wrapper over `make_transform` but remains part of the public perception API and test contract.

---

## Remaining open items

None — optional style notes below are informational only.

### Style / consistency (informational)

---

## Finding reference (original → current status)

Detailed original write-up preserved for audit trail. Status key: **Fixed**, **Retracted**, **Open**.

### §1 Dead code

| ID | Finding | Status |
|---|---|---|
| 1.1 | `generate_analytical_grasps` dead | **Retracted** — active in data prep + tests; dedup applied via helpers |
| 1.2 | Config helpers unused | **Retracted** — used across scripts |
| 1.3 | `count_supervised_training_pairs` alias | **Fixed** — removed |
| 1.4 | `grasp_pose_to_transform` wrapper | **Open** — kept for API/tests |
| 1.5 | `build_score_network` factory | **Retracted** — used in tests/exports |
| 1.6 | `build_flow_field` factory | **Retracted** — used in tests/exports |
| 1.7 | `GraspSample` unexported | **Fixed** — `TypedDict` + export |

### §2 Duplicated code

| ID | Finding | Status |
|---|---|---|
| 2.1 | `build_policy_network` / `build_value_network` | **Fixed** — `build_tanh_mlp` |
| 2.2 | Analytical grasp strict/relaxed blocks | **Fixed** — `_antipodal_grasp_from_contacts`, `_search_antipodal_grasps` |
| 2.3 | Score/flow Mish MLP stacks | **Fixed** — `build_mish_mlp` |

### §3 Inconsistent patterns

| ID | Finding | Status |
|---|---|---|
| 3.1 | Two RL save mechanisms | **Fixed** — dynamic weight copy; SB3 `net_arch` tracks `policy_num_layers`; inference still loads legacy export |
| 3.2 | Scattered `from __future__ import annotations` | **Fixed** — all modules |
| 3.3 | `condition()` only on flow model | **Fixed** — added to `GraspGeneratorModel` |

### §4 Type safety

| ID | Finding | Status |
|---|---|---|
| 4.1 | `GraspSample` loose dict | **Fixed** — `TypedDict` |
| 4.2 | Excessive `cast()` in pipelines | **Fixed** — pipelines and I/O boundaries |
| 4.3 | `GraspPoseGenerator` without shapes | **Fixed** — `Protocol` |

### §5 Efficiency

| ID | Finding | Status |
|---|---|---|
| 5.1 | CPU→GPU in `compose_with_se3_frame` | **Fixed** — `invert_rigid_transform_batch` |
| 5.2 | Duplicated MLflow logging | **Fixed** — `experiment_logging.py` |
| 5.3 | Inline logger imports | **Fixed** in all identified files |

### §6 Fragile patterns

| ID | Finding | Status |
|---|---|---|
| 6.1 | Function attribute on training step | **Fixed** — `SupervisedTrainingStep` |
| 6.2 | `empty_dataset_handling` string check | **Fixed** — removed |
| 6.3 | Silent force-closure exceptions | **Fixed** — warning logs |
| 6.4 | Hard-coded PPO hyperparameters | **Fixed** — configurable |

### §7 Minor

| ID | Finding | Status |
|---|---|---|
| 7.1 | Magic constants | **Fixed** — `utils/numerics.py` |
| 7.2 | Pre-3.9 `List` hints | **Fixed** — none found in package (uses `list`) |
| 7.3 | float32/float64 grasp tensors | **Fixed** — float32 in training pairs + analytical grasps |
| 7.4 | Wrong exception type in yaml_loader | **N/A** — already `FileNotFoundError` |
| 7.5 | Overspecified Path validation | **Fixed** — `require_path` / `require_optional_path` |
| 7.6 | Missing `__all__` | **Fixed** — `utils/__init__.py` |

---

## Suggested next steps (if continuing)

1. Document RL dual-checkpoint contract in `docs/architecture.md` (SB3 train → legacy export → `policy_runner` load).
