# Checklist

Open implementation and research tasks for the grasping-ai side project.

Verify gate after each code change:
`uv run pytest -q`, `uv run ruff check src tests scripts`, `uv run mypy src`.

Last verification (2026-08-12): 219 passed, 86.82% coverage, ruff clean, mypy clean on 45 source files.

**Runtime workflow integration is complete** (2026-08-12). See [ADR-0005](docs/adr/005-runtime-workflow-integration.md).

## Cross-cutting verification

- [ ] Current commit's CI status: pending first push-time CI run.

## Research validation (no source changes required)

These tasks validate learned behavior and research claims on the existing implementation.

- [ ] Verify physical-vs-analytical evaluation experimentally: correlate offline `grasp_success` against MuJoCo simulated lift success using sufficient trained checkpoints and many grasp candidates.
- [ ] Validate analytical metrics against physical simulation / physical outcomes (same correlation goal; may be tracked as one experiment).
- [ ] Validate the learned models rather than only their execution (diffusion vs. flow vs. analytical comparison on a common MuJoCo+evaluation pipeline).
- [ ] Validate the encoder claim empirically: does the pooled descriptor improve grasp generation? (ADR-0001 accepts canonicalization + invariant features; genuine nontrivial equivariance would require encoder replacement.)
- [ ] Generated-grasp reachability / robot embodiment: the shipped 2-DOF robot (`deploy/robot.xml`) cannot reach arbitrary diffusion outputs (8/8 IK failures in the recorded verification run). Expand the robot or constrain grasp generation to a reachable workspace.
- [ ] Diffusion model training quality: the artifact chain uses CPU-friendly hyperparameters (3 epochs, 3 objects); train longer / on more data for meaningful grasp distributions.
- [ ] Flow end-to-end runtime verification as a primary verification gate (structural contract verified via `tests/test_phase4_flow_training.py`; default artifact chain still trains diffusion).
- [ ] Baseline comparison: analytical vs. diffusion vs. flow vs. RL refinement.
- [ ] Physical grasp-success evaluation across multiple objects / grasps with the standard funnel (`IK success → collision-free → contact → force closure → lift → stable grasp → success rate`).
