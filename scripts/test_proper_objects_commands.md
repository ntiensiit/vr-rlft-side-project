# Full YCB object-set test commands

Run these commands from `vr-rlft-side-project`. The object list is loaded from
`configs/object/default.yaml`; do not duplicate it in the commands below.
Outputs are kept under shared full-dataset locations.

## Environment

```powershell
uv sync --frozen --group dev
uv run --frozen python -c "import grasping_ai; print('grasping_ai ok')"
uv run --frozen ruff check src tests scripts
uv run --frozen pytest -c pytest.toml -q -m "not slow"
```

## Prepare all configured MuJoCo object assets

```powershell
uv run python scripts/prepare_ycb_mjcf.py
```

## Generate physically validated grasp records

This runs IK, robot/table collision, bilateral-contact, and dynamic-lift
validation for every configured object.

```powershell
uv run python scripts/prepare_data.py `
  script.mode=synthetic `
  script.output_dir=data/processed `
  script.max_workers=4 `
  script.quality_report=artifacts/reports/full_synthetic_quality_large.json `
  script.num_samples=1024 `
  script.num_grasps=32 `
  script.candidate_multiplier=10 `
  script.search_multiplier=300 `
  script.oversample_factor=4 `
  script.oversample_extra=1024 `
  script.sim_validate=true `
  script.sim_validate_require_ik=true `
  script.sim_validate_require_lift=true `
  script.sim_validate_min_contacts=2 `
  script.sim_validate_fallback_analytical=false `
  script.num_simulation_steps=2000 `
  script.allow_relaxed=false
```

Audit all generated records:

```powershell
uv run python scripts/audit_synthetic_labels.py `
  script.output=artifacts/reports/full_synthetic_label_audit.json `
  paths.dataset_root=data/processed
```

## Prepare observations for all objects

```powershell
uv run python scripts/prepare_observations.py
```

## Visualize processed grasp records

Use the `.npz` records in `data/processed` directly. This does not use Flow
inference or the `.npy` candidate artifacts.

```powershell
$objectIds = @(
  uv run python -c "from grasping_ai.config import FLATTENED_YAML_CONFIG; print(' '.join(FLATTENED_YAML_CONFIG.get('objects.ids', [])))"
) -split '\s+' | Where-Object { $_ }

foreach ($objectId in $objectIds) {
    uv run python scripts/visualize_robot.py `
    script.grasp_file=data/processed/$($objectId).npz `
    script.object_id=$objectId `
    script.grasp_pose_format=object `
    script.auto_select_reachable=true `
    script.allow_ik_failure=false `
    script.close_gripper=true `
    script.lift_object=true `
    script.lift_height=0.1 `
    script.animation_duration=1.0 `
    script.table_xml=deploy/table.xml
}
```

For a static pose without the lift trajectory:

```powershell
script.lift_object=false `
script.animation_duration=0
```

Check which configured processed records exist before launching the viewer:

```powershell
$objectIds | ForEach-Object {
  [PSCustomObject]@{
    Object = $_
    Record = "data/processed/$($_).npz"
    Exists = Test-Path "data/processed/$($_).npz"
  }
}
```

## Train Flow on the full processed dataset

```powershell
uv run python scripts/train_flow.py `
  paths.dataset_root=data/processed `
  model.checkpoint=artifacts/checkpoints/full_objects_flow_left_action.pt `
  supervised.num_epochs=5000 `
  supervised.batch_size=16 `
  supervised.learning_rate=0.0005 `
  supervised.min_grasp_score=0.001 `
  training.augment=true `
  training.flow_noise_samples=4
```

## Train Diffusion on the full processed dataset

Keep the Diffusion checkpoint separate from the Flow checkpoint:

```powershell
uv run python scripts/train_diffusion.py `
  paths.dataset_root=data/processed `
  model.checkpoint=artifacts/checkpoints/full_objects_diffusion.pt `
  supervised.num_epochs=5000 `
  supervised.batch_size=16 `
  supervised.learning_rate=0.0005 `
  supervised.min_grasp_score=0.001
```

## Generate and validate Flow candidates per object

Inference produces raw `.npy` candidates. Validate each object’s candidates
before using them for visualization or physical lifting.

```powershell
$objectIds = @(
  uv run python -c "from grasping_ai.config import FLATTENED_YAML_CONFIG; print(' '.join(FLATTENED_YAML_CONFIG.get('objects.ids', [])))"
) -split '\s+' | Where-Object { $_ }

foreach ($objectId in $objectIds) {
  uv run python scripts/run_grasp_inference.py `
    model=flow `
    script.checkpoint=artifacts/checkpoints/full_objects_flow_left_action.pt `
    script.observation=data/observations/$objectId.npy `
    script.object_id=$objectId `
    script.output=artifacts/exports/$($objectId)_flow_candidates.npy

  uv run python scripts/validate_inference_candidates.py `
    script.candidate_file=artifacts/exports/$($objectId)_flow_candidates.npy `
    script.observation=data/observations/$objectId.npy `
    script.output=artifacts/validated/$($objectId)_flow_validated.npz `
    script.object_id=$objectId `
    script.table_xml=deploy/table.xml `
    script.require_ik=true `
    script.require_lift=true `
    script.min_contacts=2
}
```

## Generate and validate Diffusion candidates per object

Diffusion produces a separate raw `.npy` file and validated `.npz` archive
for every object. These artifacts must not be mixed with Flow artifacts.

```powershell
foreach ($objectId in $objectIds) {
  uv run python scripts/run_grasp_inference.py `
    model=diffusion `
    script.checkpoint=artifacts/checkpoints/full_objects_diffusion.pt `
    script.observation=data/observations/$objectId.npy `
    script.object_id=$objectId `
    script.output=artifacts/exports/$($objectId)_diffusion_candidates.npy

  uv run python scripts/validate_inference_candidates.py `
    script.candidate_file=artifacts/exports/$($objectId)_diffusion_candidates.npy `
    script.observation=data/observations/$objectId.npy `
    script.output=artifacts/validated/$($objectId)_diffusion_validated.npz `
    script.object_id=$objectId `
    script.table_xml=deploy/table.xml `
    script.require_ik=true `
    script.require_lift=true `
    script.min_contacts=2
}
```

## Visualize Flow and Diffusion validated results

The example below visualizes the first object from the configured object list.
Change `$objectId` if another validated object is needed.

```powershell
foreach ($objectId in $objectIds) {
    uv run python scripts/visualize_robot.py `
    script.grasp_file=artifacts/validated/$($objectId)_flow_validated.npz `
    script.object_id=$objectId `
    script.auto_select_reachable=true `
    script.allow_ik_failure=false `
    script.close_gripper=true `
    script.lift_object=true `
    script.lift_height=0.1 `
    script.animation_duration=1.0 `
    script.table_xml=deploy/table.xml
}
```

Visualize the matching Diffusion archive:

```powershell
foreach ($objectId in $objectIds) {
    uv run python scripts/visualize_robot.py `
    script.grasp_file=artifacts/validated/$($objectId)_diffusion_validated.npz `
    script.object_id=$objectId `
    script.auto_select_reachable=true `
    script.allow_ik_failure=false `
    script.close_gripper=true `
    script.lift_object=true `
    script.lift_height=0.1 `
    script.animation_duration=1.0 `
    script.table_xml=deploy/table.xml
}
```

## Train and compare RL for every proper object

The RL environment tracks one object per training run. Train one checkpoint
per configured object, then compare pose execution with RL in the matching
object scene. Do not reuse a policy checkpoint or grasp archive across object
IDs.

```powershell
$objectIds = @(
  uv run python -c "from grasping_ai.config import FLATTENED_YAML_CONFIG; print(' '.join(FLATTENED_YAML_CONFIG.get('objects.ids', [])))"
) -split '\s+' | Where-Object { $_ }

foreach ($objectId in $objectIds) {
  uv run python scripts/train_rl.py `
    objects.ids=[$objectId] `
    script.table_xml=deploy/table.xml `
    script.grasp_file=data/processed/$($objectId).npz `
    script.grasp_index=0 `
    rl.checkpoint=artifacts/checkpoints/$($objectId)_rl_policy.pt `
    rl.num_updates=200 `
    rl.n_steps=256
}
```

Run the pose-vs-RL comparison for each matching object/checkpoint pair:

```powershell
foreach ($objectId in $objectIds) {
  uv run python scripts/run_grasp_experiments.py `
    script.grasp_file=data/processed/$($objectId).npz `
    script.object_id=$objectId `
    script.grasp_index=0 `
    script.policy_checkpoint=artifacts/checkpoints/$($objectId)_rl_policy.pt `
    script.episodes=20 `
    script.max_steps=500 `
    script.table_xml=deploy/table.xml `
    script.output=artifacts/reports/$($objectId)_grasp_experiments.jsonl
}
```

The comparison keeps the object as a free MuJoCo body and records whether
the RL rollout maintains gripper contact and lift. A missing processed grasp
record or checkpoint should be fixed for that object before running its RL
comparison.

## Evaluate Flow candidates for the full object set

The per-object inference loop produces separate `.npy` files, so evaluate
those files with the single-object evaluator:

```powershell
foreach ($objectId in $objectIds) {
  uv run python scripts/evaluate.py `
    evaluation.single_object_key=$objectId `
    script.multi_object=false `
    script.grasps=artifacts/exports/$($objectId)_flow_candidates.npy `
    script.object_point_cloud=data/observations/$objectId.npy `
    evaluation.analytical_report=artifacts/reports/$($objectId)_flow_analytical_evaluation.jsonl
}
```

Evaluate the Diffusion candidates into separate reports:

```powershell
foreach ($objectId in $objectIds) {
  uv run python scripts/evaluate.py `
    evaluation.single_object_key=$objectId `
    script.multi_object=false `
    script.grasps=artifacts/exports/$($objectId)_diffusion_candidates.npy `
    script.object_point_cloud=data/observations/$objectId.npy `
    evaluation.analytical_report=artifacts/reports/$($objectId)_diffusion_analytical_evaluation.jsonl
}
```

For `script.multi_object=true`, first create one pickled dictionary artifact
mapping each object ID to its candidate array; separate per-object `.npy`
files are not interchangeable with that multi-object input format.

For a single-object debugging workflow, use
[`test_003_cracker_box_commands.md`](test_003_cracker_box_commands.md).
