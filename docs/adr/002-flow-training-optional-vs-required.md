# ADR-0002 — Flow-matching training: required deliverable vs. optional extension

## Status

Accepted (2026-08-12). Drives whether the flow training pipeline is treated
as a maintained deliverable on of the the project's research outputs.

## Context

`docs/PROJECT.md` §5.4 identifies two grasp-generation approaches — kinematic
flow and score-based diffusion — and historically describes them as
"alternative grasp-generation approaches unless the final project
specification explicitly requires both."

Until the most recent checklist update, the repository had a real flow
field, integrator, and sampler (`src/grasping_ai/models/flow.py`), but the
training pipeline for flow was only sketched in code comments and an
empty-dataloader notebook. The diffusion path (`scripts/train.py`, now
`scripts/train_diffusion.py`) was the
maintained supervised training entry point.

`CHECKLIST.md` added `src/grasping_ai/pipelines/train_flow.py` and
`scripts/train_flow.py`, plus `tests/test_phase4_flow_training.py`, which
brings flow training to parity with the diffusion training pipeline
(canonical-frame 9D targets, MSE velocity loss, multi-epoch loop, optional
TensorBoard).

The open question is: **does the project treat flow training as a required
deliverable, or as an optional research extension?**

## Decision

Flow training is an **optional research extension** on of the the
maintained artifact chain. It is fully implemented and tested, but the
default artifact chain (`scripts/run_artifacts.py`) trains and evaluates
the diffusion path; running the flow path is a deliberate user choice via
`scripts/train_flow.py`.

The diffusion path is the default deliverable because:

* The diffusion path has more mature evaluation coverage
  (`tests/test_phase4_generative_grasp.py`).
* The default artifact chain is end-to-end on a single object
  (`003_cracker_box`) which is feasible to validate in CI.
* Running both pipelines in CI would double the artifact-chain runtime
  without a corresponding research deliverable.

The flow path remains first-class:

* `scripts/train_flow.py` is the canonical entry point.
* `src/grasping_ai/pipelines/train_flow.py` is the maintained pipeline.
* `tests/test_phase4_flow_training.py` exercises the path.
* `docs/architecture.md` lists both paths as active generation routes.

## Rationale

* The diffusion path already provides a complete, reproducible baseline.
* The flow path exists as a parallel research route without requiring
  duplicate CI infrastructure.
* Treating flow training as an explicit choice — rather than a hidden
  default — lets downstream users select the path that matches their
  experimental objective without ambiguity.

## Consequences

* `scripts/run_artifacts.py` does not invoke `scripts/train_flow.py`.
  Users who want flow-trained checkpoints invoke the script separately.
* The flow training pipeline (`scripts/train_flow.py`) is supported,
  tested, and documented but not part of the default verification gate.
* If a downstream paper or external experiment requires flow-trained
  checkpoints by default, this ADR must be superseded and
  `scripts/run_artifacts.py` extended to invoke the flow path.

## Follow-up review triggers

This ADR should be revisited when:

* a downstream paper or external experiment requires flow-trained
  checkpoints as a deliverable;
* the artifact chain's research question shifts toward comparing diffusion
  vs. flow rather than validating a single baseline;
* the CI verification gate grows large enough that running both paths
  becomes cheap.