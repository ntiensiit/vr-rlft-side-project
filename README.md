# vr-rlft-side-project

3D robotic grasping from point clouds with equivariant representation, diffusion/flow grasp generation, MuJoCo+YCB simulation, force-closure evaluation, and reinforcement learning.

## Repository Layout

```text
configs/                       Reference experiment settings (documentation templates; CLI args are authoritative)
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
simulation -> pipelines.train_rl (SB3 PPO + Gymnasium) -> exported legacy checkpoint -> inference.policy_runner
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

The initial architecture has been fully implemented, bringing all core components out of the skeleton phase.

Key implemented features:
- **Phase 1**: Foundation (Geometry, point clouds, coordinate transforms)
- **Phase 2**: Robotics & Simulation (MuJoCo env, YCB scenes, inverse kinematics)
- **Phase 3**: Data Pipeline & Perception (SE(3) processing, point cloud datasets)
- **Phase 4**: Generative Grasp Model (Diffusion, Flow matching, SE(3) Equivariant Encoders — deterministic SE(3) canonicalization + invariant features with a trivial feature action, not a nontrivial equivariant net)
- **Phase 5**: Reinforcement Learning (RL Policy network, PPO trainer integration)
- **Phase 6**: Orchestration & Evaluation (Force-closure judging, stability/collision checking, end-to-end simulation pipelines)
- **Phase 7**: Synthetic Data (Analytical antipodal grasping, ground truth grasp generation)
- **Phase 8**: Experiment Tracking and Reproducibility (TensorBoard logging, deterministic seeding)
- **Phase 9**: Standardized Gymnasium RL Environment (Gymnasium wrappers, SB3 integration, policy export)
- **Phase 10**: Offline Analytical Evaluation and Metric Standardization (Analytical contacts, grasp-quality metrics, dictionary grasp loader)

All public interfaces, data flows, and module dependencies correspond to the architecture designed in `docs/PROJECT.md` and are supported by an exhaustive test suite with >80% coverage.
