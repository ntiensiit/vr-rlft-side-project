Use the shared processed dataset directory for the `003_cracker_box` `.npz` record.
The commands use the shared object pose `[0.5, 0.0, 0.28]` for simulation and visualization.

If MuJoCo YCB assets are missing:

```powershell
uv run python scripts/prepare_ycb_mjcf.py objects.ids=[003_cracker_box]
```

Generate physically lift-validated grasps for only `003_cracker_box`:

```powershell
uv run python scripts/prepare_data.py `
  script.mode=synthetic `
  script.output_dir=data/processed `
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

Visualize a saved simulation-validated label before training:

```powershell
uv run python scripts/visualize_robot.py script.grasp_file=data/processed/003_cracker_box.npz script.object_id=003_cracker_box script.auto_select_reachable=true script.allow_ik_failure=false script.close_gripper=true script.lift_object=true script.table_xml=deploy/table.xml
```

Audit only the generated `003_cracker_box` labels:

```powershell
uv run python scripts/audit_synthetic_labels.py `
  script.output=artifacts/reports/synthetic_label_audit.json `
  paths.dataset_root=data/processed/003_cracker_box.npz
```

Generate the single-object `003_cracker_box` observation:

```powershell
uv run python scripts/prepare_observations.py `
  objects.ids=[003_cracker_box]
```

Train the Flow model on the single-object processed dataset before inference:

```powershell
uv run python scripts/train_flow.py `
  paths.dataset_root=data/processed `
  supervised.num_epochs=5000 `
  supervised.batch_size=2 `
  supervised.learning_rate=0.0005 `
  supervised.min_grasp_score=0.001 `
  training.augment=true `
  training.flow_noise_samples=4
```

Generate inference candidates for only `003_cracker_box`:

```powershell
uv run python scripts/run_grasp_inference.py `
  model=flow `
  script.observation=data/observations/003_cracker_box.npy `
  script.output=artifacts/exports/003_cracker_box_grasp_candidates.npy
```

Visualize the single-object `003_cracker_box` candidates and validate IK reachability:

```powershell
uv run python scripts/visualize_robot.py `
  script.grasp_file=artifacts/exports/003_cracker_box_grasp_candidates.npy `
  script.object_id=003_cracker_box `
  script.grasp_pose_format=object `
  script.auto_select_reachable=true `
  script.allow_ik_failure=false `
  script.table_xml=deploy/table.xml
```
