=== End-to-End Artifact Verification ===
Timestamp: 2026-08-12T18:06:46.1468298+07:00
Reference: dev (6e86932)
Tool: uv run python scripts/run_artifacts.py

=== Outcome ===
Exit code: 0 (success)
Elapsed: 44.9s
Commands executed: 9
Retained artifacts: 13

Description: Reproducible artifact chain: supervised (YCB mesh -> synthetic dataset -> grasp checkpoint -> generated grasps -> MuJoCo simulation -> eval report) and RL (SB3 PPO -> legacy checkpoint -> policy_runner inference)

=== Retained artifacts ===
  artifacts\checkpoints\grasp_generation.pt
  artifacts\checkpoints\rl_policy.pt
  artifacts\exports\generated_grasps.npy
  artifacts\exports\grasp_poses_cracker.npy
  artifacts\reports\evaluation_report.json
  artifacts\reports\simulation_cracker.json
  data\processed\003_cracker_box.npy
  data\processed\004_sugar_box.npy
  data\processed\006_mustard_bottle.npy
  data\processed\index.json
  data\processed\ycb_mjcf\003_cracker_box\object.xml
  data\processed\ycb_mjcf\004_sugar_box\object.xml
  data\processed\ycb_mjcf\006_mustard_bottle\object.xml

=== Stage outcomes ===
* Step 0 MJCF wrappers: 3/3 objects produced (003_cracker_box, 004_sugar_box, 006_mustard_bottle)
* Step 1 synthetic dataset: 3/3 objects (--required-objects enforced; fail-fast not triggered)
* Step 2 supervised training: artifacts/checkpoints/grasp_generation.pt produced
* Step 3 grasp generation: 24 grasps across 3 objects (dict out)
* Step 4 MuJoCo simulation: 8 IK outcomes recorded (object_0 / 003_cracker_box)
* Step 5 analytical evaluation: report on the same object_0 / 003_cracker_box (end-to-end consistency)
* Step 6 RL training: artifacts/checkpoints/rl_policy.pt exported
* Step 7 policy_runner inference: smoke test passed against (21, 4) dims

=== Cross-cutting ===
* Full pytest: 219 passed (216 fast + 3 slow)
* Ruff: clean
* MyPy: clean (41 source files)
* Artifact chain CI test: tests/test_artifact_chain.py passes (slow marker)

=== Interpretation ===
The artifact chain executes end-to-end on a clean tree. The chain keeps the
same object identity (003_cracker_box / object_0) through observation,
generated grasps, extracted poses, MuJoCo simulation, and offline analytical
evaluation. The simulation outcomes are IK failures with the shipped 2-DOF
robot (8/8) because the diffusion outputs are arbitrary 4x4 poses that the
arm workspace cannot reach. This is a research-fidelity observation
(limited robot workspace) rather than a chain defect: the chain mechanics
work, and the analytical evaluation produces genuine non-zero reports when
the corresponding grasps are force-closure-eligible (here none are because
the diffusion model is untrained and outputs near-noise).

=== Phase 4 contract ===
The flow pipeline produces rtifacts/checkpoints/flow_model.pt via
scripts/train_flow.py when run; the default artifact chain trains the
diffusion path. The flow checkpoint, when produced, contains both
ncoder.* and low_field.* keys and is reloadable through
load_flow_model_checkpoint. See docs/adr/003-flow-checkpoint-joint-encoder.md.
