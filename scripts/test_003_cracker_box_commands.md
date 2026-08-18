# `003_cracker_box` single-object test commands

Run these commands from `vr-rlft-side-project`. They intentionally use only
the `003_cracker_box` object and keep object-specific outputs separate.

## Environment

```powershell
uv sync --frozen --group dev
uv run --frozen python -c "import grasping_ai; print('grasping_ai ok')"
uv run --frozen ruff check src tests scripts
uv run --frozen pytest -c pytest.toml -q -m "not slow"
```

## Prepare the object asset

```powershell
uv run python scripts/prepare_ycb_mjcf.py `
  objects.ids=[003_cracker_box]
```

## Generate simulation-validated grasps

```powershell
uv run python scripts/prepare_data.py `
  script.mode=synthetic `
  script.output_dir=data/processed `
  script.quality_report=artifacts/reports/003_synthetic_quality.json `
  "script.object_ids=[003_cracker_box]" `
  script.num_grasps=8 `
  script.candidate_multiplier=20 `
  script.search_multiplier=500 `
  script.sim_validate=true `
  script.sim_validate_require_ik=true `
  script.sim_validate_require_lift=true `
  script.sim_validate_min_contacts=2 `
  script.sim_validate_fallback_analytical=false `
  script.num_simulation_steps=2000 `
  script.allow_relaxed=false
```

Audit only the generated 003 record:

```powershell
uv run python scripts/audit_synthetic_labels.py `
  script.output=artifacts/reports/003_synthetic_label_audit.json `
  paths.dataset_root=data/processed/003_cracker_box.npz
```

## Visualize a validated grasp

Automatically select a reachable object-frame candidate from the 003 archive.
The table is enabled, the gripper is closed, and `animation_duration=0` opens
the selected static pose immediately (with no lift trajectory):

```powershell
uv run python scripts/visualize_robot.py `
  script.grasp_file=data/processed/003_cracker_box.npz `
  script.object_id=003_cracker_box `
  script.grasp_pose_format=object `
  script.auto_select_reachable=true `
  script.allow_ik_failure=false `
  script.close_gripper=true `
  script.lift_object=true `
  script.lift_height=0.1 `
  script.animation_duration=1.0 `
  script.table_xml=deploy/table.xml
```

## Prepare the 003 observation

```powershell
uv run python scripts/prepare_observations.py `
  objects.ids=[003_cracker_box]
```

## Train Flow and run single-object inference

The corrected pose pipeline uses absolute-pose left-action canonicalization.
Train a fresh checkpoint from the 003 record only; older checkpoints are
rejected because they used the incompatible conjugation representation.

```powershell
uv run python scripts/train_flow.py `
  paths.dataset_root=data/processed/003_cracker_box.npz `
  model.checkpoint=artifacts/checkpoints/003_cracker_box_flow_left_action.pt `
  supervised.num_epochs=5000 `
  supervised.batch_size=2 `
  supervised.learning_rate=0.0005 `
  supervised.min_grasp_score=0.001 `
  training.augment=true `
  training.flow_noise_samples=4
```

```powershell
uv run python scripts/run_grasp_inference.py `
  model=flow `
  script.checkpoint=artifacts/checkpoints/003_cracker_box_flow_left_action.pt `
  script.observation=data/observations/003_cracker_box.npy `
  script.object_id=003_cracker_box `
  script.output=artifacts/exports/003_cracker_box_grasp_candidates.npy
```

## Physically validate the exact inferred candidates

Inference produces raw object-frame `.npy` candidates only. They must pass
the MuJoCo IK, collision, contact, and dynamic-lift checks before they are
used for visualization or lifting.

```powershell
uv run python scripts/validate_inference_candidates.py `
  script.candidate_file=artifacts/exports/003_cracker_box_grasp_candidates.npy `
  script.observation=data/observations/003_cracker_box.npy `
  script.output=artifacts/validated/003_cracker_box_flow_validated.npz `
  script.object_id=003_cracker_box `
  script.table_xml=deploy/table.xml `
  script.require_ik=true `
  script.require_lift=true `
  script.min_contacts=2
```

The validator prints a per-reason rejection summary. If it reports `0/N`,
the viewer must stop instead of displaying an unverified pose.

Visualize only physically validated 003 candidates:

```powershell
uv run python scripts/visualize_robot.py `
  script.grasp_file=artifacts/validated/003_cracker_box_flow_validated.npz `
  script.object_id=003_cracker_box `
  script.auto_select_reachable=true `
  script.allow_ik_failure=false `
  script.close_gripper=true `
  script.lift_object=true `
  script.table_xml=deploy/table.xml
```

## Evaluate only object 003

```powershell
uv run python scripts/evaluate.py `
  evaluation.single_object_key=003_cracker_box `
  script.multi_object=false `
  script.grasps=artifacts/exports/003_cracker_box_grasp_candidates.npy `
  script.object_point_cloud=data/observations/003_cracker_box.npy `
  evaluation.analytical_report=artifacts/reports/003_analytical_evaluation.jsonl
```

Optional end-to-end single-object Flow workflow:

```powershell
uv run python scripts/run_workflow.py `
  model=flow `
  evaluation=flow `
  script.observation=data/observations/003_cracker_box.npy `
  script.object_id=003_cracker_box
```

## Train the 003 RL policy

Train an object-specific RL checkpoint using the same table asset used by the
simulation and grasp experiments:

```powershell
uv run python scripts/train_rl.py `
  objects.ids=[003_cracker_box] `
  script.table_xml=deploy/table.xml `
  script.grasp_file=data/processed/003_cracker_box.npz `
  script.grasp_index=0 `
  rl.checkpoint=artifacts/checkpoints/003_cracker_box_rl_policy.pt `
  rl.num_updates=200 `
  rl.n_steps=256
```

## Compare pose execution and RL

This runs three experiments in the same robot, table, and object scene:
validated pose execution, RL from reset, and RL from the selected pose's IK
configuration. The object remains a free MuJoCo body on the table throughout.

```powershell
uv run python scripts/run_grasp_experiments.py `
  script.grasp_file=data/processed/003_cracker_box.npz `
  script.object_id=003_cracker_box `
  script.grasp_index=0 `
  script.policy_checkpoint=artifacts/checkpoints/003_cracker_box_rl_policy.pt `
  script.episodes=20 `
  script.max_steps=500 `
  script.table_xml=deploy/table.xml `
  script.output=artifacts/reports/003_grasp_experiments.jsonl
```
