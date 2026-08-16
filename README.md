# vr-rlft-side-project

3D robotic grasping from point clouds with equivariant representation, diffusion/flow grasp generation, MuJoCo+YCB simulation, force-closure evaluation, and reinforcement learning.

## Repository Layout

```text
configs/                       YAML defaults loaded by CLI scripts (--config-dir to override)
scripts/                       Thin CLI entry points
notebooks/                     Colab-ready research notebooks (see notebooks/README.md)
notebooks/archive/             Retired exploratory notebooks (see README there)
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
| `scripts/train_diffusion.py` | Train a diffusion grasp-generation model (`--resume`, `--augment`). |
| `scripts/train_flow.py` | Train a flow-matching grasp-generation model (`--resume`, `--augment`). |
| `scripts/train_rl.py` | Train an RL grasping/interaction policy. |
| `scripts/generate_grasps.py` | Multi-object diffusion inference; writes a **dict** `.npy` (artifact chain). |
| `scripts/run_grasp_inference.py` | Single-object diffusion/flow inference; writes a plain `(K, 4, 4)` array. |
| `scripts/run_simulation.py` | Execute generated grasps in MuJoCo on YCB objects (headless). |
| `scripts/visualize_robot.py` | MuJoCo robot viewer. |
| `scripts/evaluate.py` | Evaluate generated grasps and write a report. |
| `scripts/run_artifacts.py` | End-to-end CPU-friendly artifact-chain smoke test. |
| `scripts/run_workflow.py` | Runtime workflow: inference → simulation → evaluation. |

### Grasp output formats

| Format | Writer | Loader | Use case |
| --- | --- | --- | --- |
| Pickled dict `{object_id: (K,4,4)}` | `write_generated_grasps` | `load_generated_grasps(path, object_key=...)` | Multi-object artifact chain |
| Plain array `(K, 4, 4)` | `write_generated_grasps_array` | `load_generated_grasps(path)` | Simulation / runtime workflow |

Shared inference logic lives in `grasping_ai.inference.grasp_inference_runtime`.

## Status

The initial architecture has been fully implemented, bringing all core components out of the skeleton phase.

Key implemented features:
- [**Phase 1**](docs/Phase%201%20Foundation%20%26%20Math%20Primitives.md): Foundation (Geometry, point clouds, coordinate transforms)
- [**Phase 2**](docs/Phase%202%20Simulation%20%26%20Robotics%20Core.md): Robotics & Simulation (MuJoCo env, YCB scenes, inverse kinematics)
- [**Phase 3**](docs/Phase%203%20Data%20Pipeline%20%26%20Perception.md): Data Pipeline & Perception (SE(3) processing, point cloud datasets)
- [**Phase 4**](docs/Phase%204%20Generative%20Grasp%20Model.md): Generative Grasp Model (Diffusion, Flow matching, SE(3) Equivariant Encoders — deterministic SE(3) canonicalization + invariant features with a trivial feature action, not a nontrivial equivariant net)
- [**Phase 5**](docs/Phase%205%20Reinforcement%20Learning%20Policy.md): Reinforcement Learning (RL Policy network, PPO trainer integration)
- [**Phase 6**](docs/Phase%206%20End-to-End%20Orchestration%20%26%20Eval.md): Orchestration & Evaluation (Force-closure judging, stability/collision checking, end-to-end simulation pipelines)
- [**Phase 7**](docs/Phase%207%20Synthetic%20Data%20Generation%20and%20Ground%20Truth%20Grasps.md): Synthetic Data (Analytical antipodal grasping, ground truth grasp generation)
- [**Phase 8**](docs/Phase%208%20Experiment%20Tracking%20and%20Reproducibility.md): Experiment Tracking and Reproducibility (TensorBoard logging, deterministic seeding)
- [**Phase 9**](docs/Phase%209%20Standardized%20Gymnasium%20RL%20Environment.md): Standardized Gymnasium RL Environment (Gymnasium wrappers, SB3 integration, policy export)
- [**Phase 10**](docs/Phase%2010%20Offline%20Analytical%20Evaluation%20and%20Metric%20Standardization.md): Offline Analytical Evaluation and Metric Standardization (Analytical contacts, grasp-quality metrics, dictionary grasp loader)

All public interfaces, data flows, and module dependencies correspond to the architecture designed in `docs/PROJECT.md` and are supported by an exhaustive test suite with 97% coverage.
