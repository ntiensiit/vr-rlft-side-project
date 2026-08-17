Use the shared processed dataset directory for the `004_sugar_box` `.npz` record.
The commands use the shared object pose `[0.5, 0.0, 0.28]` for simulation and visualization.

If MuJoCo YCB assets are missing:

```powershell
uv run python scripts/prepare_ycb_mjcf.py objects.ids=[004_sugar_box]
```

Generate physically lift-validated grasps for only `004_sugar_box`:

```powershell
uv run python scripts/prepare_data.py `
  script.mode=synthetic `
  script.output_dir=data/processed `
  "script.object_ids=[004_sugar_box]" `
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

Visualize the processed grasp data with the top-down lift:

```powershell
uv run python scripts/visualize_robot.py script.grasp_file=data/processed/004_sugar_box.npz script.object_id=004_sugar_box script.auto_select_reachable=true script.allow_ik_failure=false script.close_gripper=true script.lift_object=true script.table_xml=deploy/table.xml
```
