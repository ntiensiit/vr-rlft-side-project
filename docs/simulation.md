# Simulation

The simulation subsystem uses MuJoCo and the YCB object set.

## Entry points

| Path | Entry point                     | Description |
| ---- | ------------------------------- | ----------- |
| Single-grasp simulation | `scripts/run_simulation.py` | Loads a batch of grasps and runs them in MuJoCo against a YCB object. |
| RL training environment | `scripts/train_rl.py`        | Uses `MuJoCoGraspingEnv` (Gymnasium-compatible) via SB3 PPO. |

## Limitations

`simulate_grasp` applies the IK solution as a state initialization and then
executes the gripper-close command for `num_simulation_steps` physics
steps. It does **not** model a full approach/grasp/lift trajectory with
explicit control phases. See `docs/architecture.md` for the full caveat.

## MJCF asset generation

`scripts/prepare_ycb_mjcf.py` writes MuJoCo wrappers for the raw YCB
objects (whose shipped description is OpenRAVE KinBody, not MJCF) under
`data/processed/ycb_mjcf/`. The wrappers are regenerable from source; see
`docs/architecture.md` for the portability trade-off.