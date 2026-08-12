# AI Robotics Grasping Project

## 1. Project Overview

This project focuses on robotic grasping from 3D point-cloud observations, with grasp generation, simulation, and quantitative grasp evaluation as the main technical objectives.

The project context specifies the following requirements:

- Study reinforcement learning for grasping.
- Understand how grasping can be performed from point clouds.
- Understand equivariant descriptors and why they are useful for grasping.
- Understand how kinematic flow or diffusion models can learn grasping poses.
- Understand score-based diffusion.
- Generate grasp poses in MuJoCo simulation.
- Visualize grasping on objects from the YCB object set.
- Evaluate grasp quality using metrics such as force closure.

The current conceptual backbone is:

```text
YCB object or grasping data
        |
        v
Point cloud
        |
        v
3D perception and equivariant representation
        |
        v
Grasp generation using flow or diffusion
        |
        v
Candidate grasp poses
        |
        v
Robot integration and control
        |
        v
MuJoCo simulation
        |
        v
Grasp execution
        |
        v
Grasp evaluation
```

Reinforcement learning is considered an additional learning component for grasping and robot interaction. Its exact position in the final implementation is not yet fixed by the project specification. A practical architecture is to use generated grasp poses as candidates and use RL to learn grasp execution, refinement, or interaction policies in the simulator.

## 2. Problem Definition

The central problem is to generate and evaluate robot grasp poses from 3D object geometry.

Given a point cloud representing an object, the system should generate one or more candidate grasp poses. A grasp pose contains the position and orientation of the gripper and can be represented as a rigid transformation in SE(3):

```text
G = (R, t)
```

where:

- `R` represents orientation.
- `t` represents position.

The generated grasps should then be tested in a physics simulation and evaluated quantitatively.

The project therefore combines several areas:

- 3D computer vision and point-cloud processing.
- Geometric deep learning.
- Equivariant representation learning.
- Generative modeling.
- Reinforcement learning.
- Robot kinematics and control.
- Physics simulation.
- Grasp quality evaluation.

## 3. Objectives

The project objectives currently identified from the requirements are:

1. Understand point-cloud-based grasping.
2. Understand equivariant descriptors and their role in 3D grasping.
3. Understand kinematic flow and score-based diffusion for grasp generation.
4. Implement or integrate a grasp-generation pipeline.
5. Generate grasp poses for YCB objects.
6. Execute generated grasps in MuJoCo.
7. Visualize the robot, gripper, object, and grasp execution.
8. Evaluate grasp quality quantitatively.
9. Investigate reinforcement learning for grasping and interaction.
10. Establish an experimental workflow that allows different grasp-generation approaches to be compared.

## 4. Scope

### 4.1 In scope

The current scope includes:

- Point-cloud-based object representation.
- Equivariant 3D descriptors.
- Grasp pose generation.
- Kinematic flow or diffusion-based generation.
- Score-based diffusion concepts.
- MuJoCo simulation.
- YCB objects.
- Robot and gripper interaction in simulation.
- Grasp quality evaluation.
- Force-closure evaluation.
- Reinforcement learning for grasping where required by the final architecture.
- Quantitative experiments.

### 4.2 Not yet specified

The following details have not been fixed by the current project context:

- The exact robot model to be used.
- The exact gripper model.
- The exact grasp-pose training dataset other than the explicitly mentioned YCB object set.
- Whether diffusion or kinematic flow will be the final grasp-generation method.
- Whether the equivariant encoder will be implemented from scratch, modified from an existing implementation, or reused from an existing model.
- Which RL algorithm will be used.
- Whether RL will be used for grasp generation, grasp refinement, grasp execution, or another stage.
- Whether a real robot will be integrated.
- The final experimental protocol and dataset split.
- The final hyperparameters and training schedule.

These decisions should not be treated as fixed project requirements until they are explicitly determined.

## 5. System Architecture

The architecture is organized around several functional layers.

### 5.1 Data layer

The data layer provides object geometry, point clouds, grasp data where available, and simulation-generated experience.

The current explicitly identified object source is the YCB object set.

The project may also require a grasp-pose dataset for supervised or generative model training. The exact dataset has not yet been specified.

### 5.2 Perception layer

The perception layer converts available 3D observations into a representation suitable for grasp generation.

Typical responsibilities include:

- Point-cloud loading.
- Point sampling and preprocessing.
- Geometric transformations.
- Coordinate normalization.
- Point-cloud feature extraction.

The perception layer should not contain robot control, training loops, or MuJoCo-specific business logic.

### 5.3 Equivariant representation layer

An equivariant descriptor is intended to encode 3D geometry while respecting transformations of the input.

For a transformation `T`, an equivariant representation follows the general relationship:

```text
f(Tx) = T' f(x)
```

The important motivation in this project is that object geometry and grasp poses exist in 3D space. If an object is rotated, the corresponding valid grasp representation should transform consistently.

The equivariant descriptor therefore provides a geometric representation that can condition the grasp-generation model.

**Status note (2026-08-12):** the active `SE3EquivariantPointNet` is a deterministic SE(3) canonicalization followed by an MLP on canonical-frame coordinates. The pooled descriptor is invariant under rigid SE(3); the per-point features transform under the trivial feature action. This is a canonicalization-based design, **not** a conventional nontrivial equivariant neural representation (e.g., steerable / equivariant tensor-field networks). If the intended research objective requires genuine nontrivial equivariance, the encoder must be replaced.

### 5.4 Grasp generation layer

The grasp-generation layer maps an object representation to candidate grasp poses.

The conceptual target is a conditional distribution:

```text
p(G | P)
```

where:

- `P` is the point cloud.
- `G` is a grasp pose.

Two approaches have been identified:

- Kinematic flow.
- Score-based diffusion.

They should be treated as alternative grasp-generation approaches unless the final project specification explicitly requires both.

A flow formulation can learn a vector field that transforms an initial distribution toward the grasp distribution.

A score-based diffusion formulation learns a score function used during denoising and sampling.

**Status note (2026-08-12):** both flow and diffusion paths are now fully implemented through to training. `scripts/train.py` runs diffusion-based supervised training; `scripts/train_flow.py` runs flow-matching supervised training (newly added in Priority 6 of the hot-fix checklist). Both produce checkpoints loadable by the corresponding inference modules.

### 5.5 Robotics layer

The robotics layer converts generated grasp poses into robot-level actions.

Its responsibilities include:

- Coordinate transformations.
- Robot kinematics.
- Inverse kinematics where required.
- End-effector control.
- Gripper control.

The robotics layer should not contain model-training logic.

### 5.6 Simulation layer

MuJoCo provides the physics simulation environment.

The simulation layer is responsible for:

- Loading the simulation model.
- Loading YCB objects.
- Creating scenes.
- Managing simulation state.
- Stepping the physics simulation.
- Handling contacts and physical interaction.
- Providing observations and outcomes to higher-level components.

The simulation environment should not directly contain diffusion training or model implementation.

### 5.7 Reinforcement learning layer

RL introduces a closed-loop interaction process:

```text
Observation
        |
        v
Policy
        |
        v
Action
        |
        v
MuJoCo
        |
        v
Reward and next observation
        |
        v
Policy update
```

The RL policy can potentially operate on robot state, grasp candidates, point-cloud features, or other observations depending on the final design.

Possible roles for RL identified during project analysis include:

- Grasp execution.
- Grasp refinement.
- Interaction policy learning.
- Optimization using simulation feedback.

The exact role is not yet fixed by the project specification.

### 5.8 Evaluation layer

The evaluation layer measures whether generated grasps are geometrically and physically useful.

The explicitly mentioned metric is force closure.

Other evaluation signals considered in the architecture include:

- Collision-free grasp.
- Contact quality.
- Stability.
- Object lift success.
- Overall grasp success rate.

These should be implemented as evaluation components rather than embedded inside the model.

## 6. End-to-End Workflow

The main workflow is:

```text
Dataset
        |
        v
Preprocessing
        |
        v
Point cloud
        |
        v
Equivariant representation
        |
        v
Grasp generation
        |
        v
Candidate grasp poses
        |
        v
Robot integration
        |
        v
MuJoCo execution
        |
        v
Grasp evaluation
        |
        v
Experiment results
```

For model development:

```text
Training data
        |
        v
Data loader
        |
        v
Point-cloud preprocessing
        |
        v
Model training
        |
        v
Checkpoint
        |
        v
Inference
        |
        v
Generated grasps
        |
        v
Simulation
        |
        v
Evaluation
```

For RL:

```text
MuJoCo environment
        |
        v
Observation
        |
        v
RL policy
        |
        v
Action
        |
        v
Environment transition
        |
        v
Reward
        |
        v
Policy training
```

## 7. Grasp Representation

A grasp pose is represented by position and orientation:

```text
G = (R, t)
```

and belongs to the rigid-body transformation space SE(3).

This representation is important because grasp generation is not only a position prediction problem. The model must also determine the orientation of the gripper relative to the object.

The project therefore requires understanding:

- 3D coordinate frames.
- Rotation.
- Translation.
- Rigid transformations.
- SE(3).
- SO(3).
- Transformation of grasp poses between coordinate frames.

## 8. Force Closure and Grasp Evaluation

Force closure is one of the explicitly required evaluation concepts.

A rigid object in three dimensions can be affected by a six-dimensional wrench consisting of force and torque:

```text
w = [F, tau]
```

A force-closure grasp should provide contact forces capable of resisting the relevant external wrenches.

The evaluation layer should therefore separate geometric grasp generation from grasp-quality measurement.

A typical evaluation process is:

```text
Generated grasp
        |
        v
Collision check
        |
        v
Contact analysis
        |
        v
Force-closure evaluation
        |
        v
Simulation execution
        |
        v
Lift or grasp success
```

A quantitative evaluation should eventually report metrics across multiple objects and grasp attempts rather than relying only on visual inspection.

## 9. Reinforcement Learning

RL is part of the project requirements, but the exact integration point has not yet been fixed.

The conceptual RL formulation is:

```text
state -> policy -> action -> simulator -> reward
```

A policy can be represented as:

```text
pi(a | s)
```

where `s` is the state or observation and `a` is the robot action.

The reward can incorporate grasp-related objectives such as:

- Successful grasp.
- Stable grasp.
- Successful lift.
- Collision penalties.
- Other project-defined grasp-quality objectives.

RL should remain separated from the lower-level robot-control implementation. The RL module should produce actions or policy decisions, while the robotics and simulation modules execute those decisions.

## 10. Dataset and Data Sources

### 10.1 YCB Object Set

The YCB object set is explicitly identified in the project requirements.

Its role in this project is primarily to provide objects and 3D geometry for grasp generation, MuJoCo visualization, and simulation evaluation.

The project does not currently specify that YCB alone is the training dataset for the grasp-generation models.

### 10.2 Grasp-pose training data

A grasp-pose dataset is likely required if the grasp-generation models are to be trained from data. The current context does not specify the exact dataset.

The expected conceptual structure is:

```text
Point cloud
+
Grasp pose
+
Optional grasp-quality information
```

The exact source and format remain to be determined.

### 10.3 Simulation-generated data

MuJoCo can generate interaction experience for RL or additional experiments.

A simulation trajectory can conceptually contain:

```text
state
action
reward
next_state
success/failure
```

Additional contact or physical information may be recorded for evaluation.

The exact simulation data-generation protocol has not yet been fixed.

## 11. Models

The models identified during project planning are:

| Model | Purpose | Status in current architecture |
|---|---|---|
| Equivariant point-cloud encoder | Learn geometric object representation | Planned |
| Diffusion grasp generator | Generate grasp poses | Candidate approach |
| Kinematic flow grasp generator | Generate grasp poses | Alternative candidate approach |
| RL policy | Learn grasping or interaction behavior | Planned, exact role not fixed |
| Grasp-quality predictor | Rank candidate grasps | Optional; not required by current specification |
| RL critic/value model | Support actor-critic RL | Depends on selected RL algorithm |

The project does not require training every model simultaneously.

A practical development sequence is to first establish a working grasp-generation and simulation pipeline, then add more complex learning components.

## 12. Technology Stack

The project context has identified or recommended the following technologies.

### Core

- Python: primary implementation language.
- PyTorch: neural-network and deep-learning implementation.
- MuJoCo: physics simulation.

### 3D and numerical processing

- NumPy: numerical arrays and mathematical operations.
- SciPy: scientific and numerical utilities.
- Open3D: point-cloud and 3D geometry processing.

### RL

- Gymnasium: environment interface.
- Stable-Baselines3: possible implementation of standard RL algorithms.

These RL libraries are recommended components rather than fixed project requirements because the final RL algorithm has not yet been specified.

### Experiment and development tools

- Matplotlib: experiment visualization.
- TensorBoard or Weights & Biases: training and experiment tracking.
- Git and GitHub: version control and repository management.

The exact final library selection should be recorded in project configuration once implementation begins.

## 13. Repository Structure

The recommended repository structure is:

```text
grasping_ai/
├── README.md
├── PROJECT.md
├── pyproject.toml
├── .gitignore
├── .env.example
│
├── configs/
│   ├── base.yaml
│   ├── data.yaml
│   ├── model.yaml
│   ├── training.yaml
│   ├── evaluation.yaml
│   ├── simulation.yaml
│   └── robot.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── README.md
│
├── src/
│   └── grasping_ai/
│       ├── data/
│       ├── perception/
│       ├── models/
│       ├── training/
│       ├── inference/
│       ├── robotics/
│       ├── sensors/
│       ├── simulation/
│       ├── evaluation/
│       ├── pipelines/
│       └── utils/
│
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── generate_grasps.py
│   ├── run_simulation.py
│   └── train_rl.py
│
├── artifacts/
│   ├── checkpoints/
│   ├── exports/
│   └── reports/
│
├── experiments/
│   └── README.md
│
├── logs/
│   └── README.md
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── docs/
│   ├── architecture.md
│   ├── dataset.md
│   ├── grasping.md
│   ├── simulation.md
│   └── experiments.md
│
└── deploy/
    ├── Dockerfile
    ├── docker-compose.yaml
    └── run.sh
```

This structure is intended to remain simple during the initial implementation while leaving clear boundaries for later expansion.

## 14. Module Responsibilities

| Module | Responsibility | Should not contain |
|---|---|---|
| `data/` | Dataset loading, parsing, and data transforms | Robot control or training business logic |
| `perception/` | Point-cloud and 3D preprocessing | RL or robot control |
| `models/` | Neural-network architectures | Direct hardware control |
| `training/` | Training loops, losses, optimization | Robot-specific execution |
| `inference/` | Running trained models | Training loops |
| `robotics/` | Kinematics, IK, controller, gripper interfaces | Model training |
| `sensors/` | Sensor interfaces and acquisition | High-level perception algorithms |
| `simulation/` | MuJoCo environment and YCB simulation | Neural-network implementation |
| `evaluation/` | Force closure and grasp-quality metrics | Training logic |
| `pipelines/` | Orchestration of multiple modules | Low-level reusable implementations |
| `utils/` | Small shared utilities | Domain-specific business logic |

## 15. Dependency Boundaries

The main architectural rule is that low-level modules should not depend on high-level orchestration.

The intended dependency direction is:

```text
configs
   |
   v
data
   |
   v
perception
   |
   v
models
   |
   v
inference
   |
   v
robotics
   |
   v
simulation
   |
   v
evaluation
```

Training and pipeline modules orchestrate these components rather than becoming dependencies of them.

Important restrictions are:

- AI models must not directly depend on robot hardware.
- Robot controllers must not contain training or inference logic.
- Dataset loaders must not contain model or robot business logic.
- Configuration values should not be hard-coded throughout source files.
- Lower-level modules must not import pipeline entry points.
- Modules should expose data or interfaces rather than directly controlling unrelated subsystems.

For example, a grasp-generation model should return a grasp pose. It should not call a robot controller itself.

## 16. Configuration

Configuration is separated from source code.

The `configs/` directory is intended for:

- Dataset paths and parameters.
- Model parameters.
- Training parameters.
- Evaluation parameters.
- Simulation parameters.
- Robot parameters.

The source code should consume configuration values rather than hard-code experiment-specific parameters.

The configuration system should also make it possible to compare experiments while keeping the implementation unchanged.

## 17. Training Workflow

The general training workflow is:

```text
Raw data
        |
        v
Preprocessing
        |
        v
Dataset loader
        |
        v
Equivariant encoder
        |
        v
Grasp-generation model
        |
        v
Training loop
        |
        v
Checkpoint
```

For diffusion, the model learns a score-based denoising process conditioned on the object representation.

For flow-based generation, the model learns a vector field that transforms an initial distribution toward the grasp distribution.

For RL, the training data is generated through interaction with the simulation environment rather than necessarily coming from a static labeled dataset.

## 18. Inference Workflow

At inference time, training is not performed.

The intended flow is:

```text
Object observation
        |
        v
Point cloud
        |
        v
Equivariant representation
        |
        v
Trained grasp generator
        |
        v
Candidate grasp poses
        |
        v
Robot integration
```

The candidate grasps can then be evaluated or executed in MuJoCo.

## 19. Simulation Workflow

The current simulation target is MuJoCo with YCB objects.

A typical simulation workflow is:

```text
Load robot and scene
        |
        v
Load YCB object
        |
        v
Obtain object observation
        |
        v
Generate grasp pose
        |
        v
Transform grasp into the required coordinate frame
        |
        v
Move robot toward grasp
        |
        v
Close gripper
        |
        v
Lift or otherwise test the grasp
        |
        v
Record outcome
```

The exact robot model, gripper, control strategy, and scene configuration are not yet specified.

## 20. Evaluation Workflow

Evaluation should be independent of model training.

The same evaluator should be usable for different grasp-generation approaches.

For example:

```text
Model A
        |
        v
Generated grasps
        |
        v
Common evaluator

Model B
        |
        v
Generated grasps
        |
        v
Common evaluator
```

This makes quantitative comparison possible.

Relevant metrics include:

- Force closure.
- Collision-free rate.
- Grasp stability.
- Lift success.
- Overall grasp success rate.

The final metric definitions and thresholds are not yet fixed.

## 21. Experiment Management

Experiments should record:

- Configuration used.
- Model or method.
- Dataset version or source.
- Training parameters.
- Evaluation results.
- Relevant generated artifacts.

Source code remains under `src/`.

Experiment-specific configuration and results belong under `experiments/` and `artifacts/`.

Large datasets, checkpoints, and logs should not be committed directly to the Git repository.

## 22. Development Principles

The project follows several design principles.

### Keep the baseline executable

A working end-to-end pipeline should be established before introducing every advanced learning component simultaneously.

A reasonable progression is:

```text
MuJoCo + YCB
        |
        v
Robot and gripper control
        |
        v
Grasp pose representation
        |
        v
Basic grasp generation
        |
        v
Grasp evaluation
        |
        v
Equivariant representation
        |
        v
Diffusion or flow
        |
        v
RL
```

This ordering is an engineering strategy, not a statement that RL is less important scientifically.

### Separate models from execution

A model predicts. Robotics code executes. Simulation provides the environment. Evaluation measures the result.

This separation prevents AI code from becoming coupled to a particular robot or simulator.

### Prefer one clear abstraction over premature frameworks

The initial repository should not introduce microservices, APIs, complex plugin systems, or multiple abstraction layers that are not required by the current project.

If the number of models, robots, or simulation environments grows substantially, the existing directories can be subdivided later.

### Keep configuration outside implementation

Experiment parameters should be stored in configuration files rather than scattered through Python source code.

### Make evaluation reusable

Evaluation should operate on generated grasp results independently of the method that produced them.

## 23. Current Limitations and Open Decisions

The current project definition does not yet specify:

- The final robot and gripper.
- The exact grasp training dataset.
- The final grasp representation beyond the general SE(3) pose concept.
- The exact equivariant architecture.
- The final choice between kinematic flow and diffusion.
- Whether both flow and diffusion will be implemented.
- The exact score-based diffusion architecture.
- The RL algorithm.
- The exact RL observation and action spaces.
- The exact reward function.
- The final force-closure implementation.
- The final evaluation protocol.
- Whether deployment to physical hardware will occur.

These should be treated as project decisions to be resolved rather than assumptions embedded in the codebase.

## 24. Future Development Direction

Based on the current requirements, the natural development direction is:

1. Establish MuJoCo and YCB simulation.
2. Establish robot and gripper control.
3. Implement the grasp-pose representation and basic grasp execution.
4. Establish point-cloud processing.
5. Implement or integrate an equivariant descriptor.
6. Implement or integrate one grasp-generation approach using flow or diffusion.
7. Generate and evaluate multiple grasp candidates.
8. Add quantitative force-closure and simulation-based evaluation.
9. Add RL for the part of grasping or interaction that is selected for learning.
10. Compare methods through controlled experiments.
11. Consider real-robot integration only if it becomes an explicit project requirement.

## 25. Getting Started

A new contributor should understand the project in the following order:

1. Read this document and the repository `README.md`.
2. Understand the grasping problem and SE(3) grasp representation.
3. Understand point clouds and the perception pipeline.
4. Understand why equivariant descriptors are useful for 3D grasping.
5. Understand flow and score-based diffusion at the level required by the selected implementation.
6. Set up the Python and MuJoCo environment.
7. Load a YCB object in simulation.
8. Verify robot and gripper control.
9. Run the basic grasp-generation and evaluation pipeline.
10. Only then work on training and RL components.

The first implementation milestone should be an end-to-end system that can produce a grasp pose, execute it in MuJoCo on a YCB object, and report an objective evaluation result.

## 26. Repository Artifact Policy

The repository distinguishes source and generated artifacts.

Source and configuration that define the project should be version controlled:

- `src/`
- `configs/`
- `scripts/`
- `tests/`
- `docs/`
- `deploy/`
- Project documentation.

Generated or potentially large files should normally remain outside Git:

- Raw datasets.
- Processed datasets when large.
- Training logs.
- Model checkpoints.
- Exported model binaries.
- Experiment outputs.

The `.gitignore` should reflect these boundaries.

## 27. Summary

The project is a 3D robotic grasping system centered on point-cloud perception, equivariant representation learning, generative grasp-pose generation, MuJoCo simulation, and quantitative grasp evaluation.

The main conceptual pipeline is:

```text
3D object observation
        |
        v
Point cloud
        |
        v
Equivariant descriptor
        |
        v
Flow or diffusion grasp generation
        |
        v
Candidate grasp poses
        |
        v
Robot kinematics and control
        |
        v
MuJoCo with YCB objects
        |
        v
Grasp execution
        |
        v
Force-closure and simulation evaluation
```

Reinforcement learning adds a closed-loop learning component in which a policy interacts with the simulated environment and learns from rewards. Its exact role in the final architecture remains to be determined.

The repository architecture intentionally separates data, perception, models, training, inference, robotics, sensors, simulation, evaluation, configuration, and orchestration. This keeps the initial system simple while preserving clear boundaries for future expansion.

## 28. Architecture Decision Records

Significant design decisions for the project are recorded under docs/adr/
as lightweight ADR documents. Each ADR captures the context, decision,
rationale, and consequences of a single choice, following the conventions in
docs/adr/README.md.

Current ADRs:

- [ADR-0001: Phase 4 representation — canonicalization vs. nontrivial equivariance](adr/001-phase4-canonicalization-vs-equivariant.md)
- [ADR-0002: Flow-matching training — required deliverable vs. optional extension](adr/002-flow-training-optional-vs-required.md)
- [ADR-0003: Flow model checkpoint contract — jointly train encoder + flow field](adr/003-flow-checkpoint-joint-encoder.md)