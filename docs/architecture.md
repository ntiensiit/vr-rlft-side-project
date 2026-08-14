# Architecture

This document summarizes the current architecture of the repository, the
contract between subsystems, and the active interfaces that downstream
users and developers need to know about.

## Module dependency direction

The intended dependency direction is strictly:

```
config → data → perception → models → inference → robotics
                                              → simulation → evaluation
                                       → training
                                       → pipelines
```

Pipelines orchestrate lower-level components; lower-level modules do not
import from pipelines or training.

## Subsystem overview

| Subsystem     | Source root                              | Responsibility |
| ------------- | ---------------------------------------- | -------------- |
| Config        | `configs/`                               | YAML defaults merged by CLI scripts; flags override. |
| Data          | `src/grasping_ai/data/`                  | Dataset discovery, serialized-sample loading, point-cloud preprocessing. |
| Perception    | `src/grasping_ai/perception/`            | Point-cloud sampling, normalization, SE(3) frame construction. |
| Models        | `src/grasping_ai/models/`                | Diffusion/flow grasp generation, equivariant encoder, RL policy. |
| Inference     | `src/grasping_ai/inference/`             | Checkpoint loading, grasp generation, policy runners. |
| Robotics      | `src/grasping_ai/robotics/`              | Coordinate transforms, kinematics, IK. |
| Simulation    | `src/grasping_ai/simulation/`            | MuJoCo env, scene assembly, YCB loading. |
| Evaluation    | `src/grasping_ai/evaluation/`            | Collision, force closure, analytical lift/stability judges. |
| Training      | `src/grasping_ai/training/`              | Supervised/RL optimizers and step/loop primitives. |
| Pipelines     | `src/grasping_ai/pipelines/`             | End-to-end orchestration of the above. |
| Scripts       | `scripts/`                               | CLI entry points; thin wrappers around pipelines. |

## RL policy: two distinct interfaces

The repository intentionally exposes **two distinct policy interfaces**.
Users must understand which one they are loading.

### 1. Native Stable-Baselines3 PPO (training-time)

* Created inside `src/grasping_ai/pipelines/train_rl.py` via
  `stable_baselines3.PPO("MlpPolicy", env, …)`.
* Used during RL training and managed entirely by SB3.
* Not serialized to disk by the project's `train_rl` pipeline; SB3 checkpoints
  (when saved) remain native SB3 `.zip` artifacts.
* Stochastic sampling distribution is part of the policy.

### 2. Exported legacy MLP inference adapter (post-training)

* Produced at the end of `run_rl_training_pipeline` by copying selected
  SB3 PPO MLP weights (`mlp_extractor.policy_net[0/2]` and `action_net`)
  into a `build_policy_network(obs_dim, act_dim, hidden_dim, 2)` instance.
* Saved via `save_rl_policy_checkpoint` to the path passed as
  `--policy-checkpoint` (e.g., `artifacts/checkpoints/rl_grasp_policy.pt`).
* Loaded by `load_rl_policy_checkpoint` + `build_rl_policy_runner`
  (in `src/grasping_ai/inference/policy_runner.py`) into a deterministic
  `Sequential` MLP that returns a single mean action per observation.
* **Does not reproduce the full SB3 stochastic policy**: the stochastic
  sampling distribution, value head, log_std, and SB3 internal state are
  discarded.
* Designed for downstream deterministic action providers (robot drivers,
  policy integration tests) that expect a plain MLP state-dict and a
  callable `(observation) -> action` signature.

### Action-bound contract

The legacy MLP runner optionally accepts `action_low`/`action_high` to
clip returned actions to the Gymnasium environment's actuator bounds.
Training-time SB3 PPO enforces actuator bounds internally through the
environment's `Box.action_space`; standalone deployment code that does not
use the Gymnasium env must clip manually.

### Choosing an interface

| Use case                                                         | Interface           |
| ---------------------------------------------------------------- | ------------------- |
| Continue or resume SB3 PPO training                              | Native SB3 PPO      |
| Deploy a deterministic action provider for robot control          | Legacy MLP export   |
| Compare against other RL algorithms (e.g., SAC, A2C)             | Native SB3 PPO      |
| Reproduce the training-time stochastic behavior at inference      | Not supported        |

## SE(3) "equivariant" encoder: actual semantics

`SE3EquivariantPointNet` (in `src/grasping_ai/models/equivariant_encoder.py`)
is a deterministic SE(3) canonicalization followed by an MLP on
canonical-frame coordinates. The pooled object descriptor is invariant
under the rigid SE(3) action; the per-point features transform under the
trivial feature action.

It is **not** a conventional nontrivial equivariant neural network (e.g.,
steerable / equivariant tensor-field networks). If the intended research
objective requires genuine nontrivial equivariance, the encoder must be
replaced; see [ADR-0001](adr/001-phase4-canonicalization-vs-equivariant.md) and
`CHECKLIST.md` (encoder validation research item).

## Grasp representation

The active grasp representation is 9D = 3 translation components + the
first two columns of the rotation matrix (a 6D rotation subset), as
produced by `se3_to_vec` in `src/grasping_ai/data/grasp_vector.py`. This
replaces an older 7D convention (3 translation + 4 quaternion) that no
code path uses; `configs/model.yaml::grasp_dim` is set to 9.

## Generated MJCF portability

`scripts/prepare_ycb_mjcf.py` writes MJCF wrappers that reference the
real YCB mesh files via MuJoCo's `<mesh file="…">` attribute using the
mesh's absolute path. This is a deliberate trade-off: MuJoCo resolves
`<include>`d-file `meshdir` relative to the including scene rather than
the included file, which prevents reliable relocatable paths. The
wrappers are therefore regenerable from source (`scripts/run_artifacts.py`)
in any fresh checkout rather than being relocatable as-is.

## simulate_grasp() trajectory simplification

`simulate_grasp` (in `src/grasping_ai/pipelines/simulate_grasp.py`)
applies the IK solution as a state initialization and then executes the
gripper-close command for `num_simulation_steps` physics steps. It does
**not** model a full approach/grasp/lift trajectory with explicit
control phases; that remains a research-modeling limitation rather than
a phase-blocker.

## Phase 10 evaluation: caveat on `compute_grasp_quality`

`compute_grasp_quality` (in `src/grasping_ai/evaluation/force_closure.py`)
is a project-specific wrench-space quality metric and should be treated
as an experimental proxy, **not** assumed equivalent to a canonical
Ferrari-Canny quality score without further validation.