# vr-rlft-side-project

3D robotic grasping from point clouds with equivariant representation, diffusion/flow grasp generation, MuJoCo+YCB simulation, force-closure evaluation, and reinforcement learning.

## Repository Layout

```text
configs/                       Experiment configuration (YAML)
scripts/                       Thin CLI entry points
src/grasping_ai/
  data/                        Dataset loading and transforms
  perception/                  Point-cloud preprocessing, SE(3) helpers
  sensors/                     Sensor observation acquisition
  models/                      Equivariant encoder, diffusion, flow, RL policy
  inference/                   Loading trained models, generating grasps/actions
  robotics/                    Coordinate transforms, kinematics, gripper control
  simulation/                  MuJoCo environment, scene, YCB loading
  evaluation/                  Force closure, collision, stability, lift
  training/                    Supervised and RL training loops
  pipelines/                   End-to-end orchestration of the above modules
```

## End-to-End Workflows

Grasp generation (object -> candidate grasp poses):

```text
sensors -> perception -> models -> inference -> pipelines.generate_grasps
```

Simulation and evaluation (candidate grasp -> outcome):

```text
pipelines.simulate_grasp -> simulation + robotics + evaluation
```

Supervised training:

```text
data -> perception -> models -> training -> inference
```

RL training:

```text
simulation -> inference.policy_runner -> training.rl_trainer -> checkpoints
```

## Scripts

| Script | Purpose |
| --- | --- |
| `scripts/prepare_data.py` | Build the dataset index from raw records. |
| `scripts/train.py` | Train a grasp-generation model. |
| `scripts/train_rl.py` | Train an RL grasping/interaction policy. |
| `scripts/generate_grasps.py` | Generate grasp poses for a set of observations. |
| `scripts/run_simulation.py` | Execute generated grasps in MuJoCo on YCB objects. |
| `scripts/evaluate.py` | Evaluate generated grasps and write a report. |

## Status

This repository currently contains a Python source-code skeleton. Function bodies raise `NotImplementedError`; the public interfaces, dependency boundaries, and data flow already reflect the architecture described in `docs/PROJECT.md`.
