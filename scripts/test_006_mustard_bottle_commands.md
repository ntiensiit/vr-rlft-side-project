Use a separate dataset directory so the existing `003` and `004` `.npz` files are not included.
The commands use the shared object pose `[0.5, 0.0, 0.28]` for simulation and visualization.

If MuJoCo YCB assets are missing:

```powershell
uv run python scripts/prepare_ycb_mjcf.py objects.ids=[006_mustard_bottle]
```

Generate physically lift-validated grasps for only `006_mustard_bottle`:

```powershell
uv run python scripts/prepare_data.py prepare.mode=synthetic prepare.output_dir=outputs/validated_006 objects.ids=[006_mustard_bottle] synthetic.num_grasps=8 synthetic.candidate_multiplier=4 synthetic.search_multiplier=100 synthetic.min_quality_score=0.001 synthetic.sim_object_position=[0.5,0.0,0.28] synthetic.sim_validate=true synthetic.sim_validate_require_ik=true synthetic.sim_validate_require_lift=true synthetic.sim_validate_min_contacts=2 synthetic.sim_validate_fallback_analytical=false paths.output_index=outputs/validated_006/index.json
```

Visualize the processed grasp data with the top-down lift:

```powershell
uv run python scripts/visualize_robot.py script.grasp_file=outputs/validated_006/006_mustard_bottle.npz script.object_id=006_mustard_bottle script.auto_select_reachable=true script.allow_ik_failure=false script.close_gripper=true script.lift_object=true script.table_xml=deploy/table.xml
```
