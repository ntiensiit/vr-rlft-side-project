# ADR-001 — Phase 4 representation: canonicalization vs. nontrivial equivariance

## Status

Accepted (2026-08-12). Drives the design of `SE3EquivariantPointNet` in
`src/grasping_ai/models/equivariant_encoder.py` and downstream conditioning.

## Context

`docs/PROJECT.md` §5.3 motivates an "equivariant descriptor" that respects
SE(3) transformations of the input point cloud. The intended use is to
condition the grasp-generation model on object geometry so that an SE(3)
action on the input produces a corresponding, predictable change in the
generated grasps.

Two architectures satisfy that motivation:

1. **Canonicalization + invariant features.** Compute a deterministic
   SE(3) frame from the point cloud, express the cloud in that canonical
   frame, then apply ordinary MLP layers to the canonical coordinates.
   The pooled descriptor is invariant under the rigid SE(3) action; the
   per-point features transform under the trivial feature action.

2. **Nontrivial equivariant network.** Use a steerable / equivariant
   tensor-field network (e.g., e2cnn, escnn, equivariant point networks)
   whose features transform under a non-trivial representation of SE(3)
   for every layer, without first reducing the input to a canonical frame.

Both can satisfy the conditioning role described in §5.3. They differ in:

| Property                          | Canonicalization + MLP                | Nontrivial equivariant               |
| --------------------------------- | ------------------------------------- | ------------------------------------ |
| Implementation complexity         | Low (custom frame, small MLP)         | High (group representations, basis)  |
| Trivial feature action            | Yes                                   | No                                  |
| Robustness to extreme point clouds | Frame construction can degenerate     | Steerable kernels remain well-posed  |
| External library dependency        | None                                  | e2cnn / escnn / similar             |
| Empirical evidence on this task   | None yet                              | Established literature              |

## Decision

The active implementation is **canonicalization + invariant features**
(option 1). This is implemented in `SE3EquivariantPointNet`. The README and
`docs/architecture.md` describe it as canonicalization-based with trivial
feature action.

## Rationale

* The project is a research prototype; the lowest-risk path is the simpler
  architecture with a deterministic, easily-debugged conditioning signal.
* Nontrivial equivariant networks require an additional external dependency
  and substantially more implementation surface area; the project's
  three-target YCB simulation experiments do not yet require them.
* The architecture remains compatible with future replacement of
  `SE3EquivariantPointNet` with a steerable variant — downstream code
  consumes only `(B, F)` pooled features and `(B, N, F)` per-point features.

## Consequences

* The "SE(3)-equivariant" naming in source and documentation is
  inaccurate. The README, `docs/architecture.md`, and `docs/PROJECT.md` §5.3
  state this explicitly. Renaming `SE3EquivariantPointNet` →
  `CanonicalizationInvariantPointNet` is a possible follow-up.
* If the project's scientific objective specifically requires nontrivial
  SE(3)-equivariant feature learning, this ADR must be superseded and the
  encoder replaced with a steerable network.
* No current code path depends on nontrivial equivariance. The grasp
  generation is conditioned on the invariant pooled descriptor; whether
  that conditioning is rotation-equivariant in the strict group-theoretic
  sense depends on the choice made here.

## Follow-up review triggers

This ADR should be revisited when:

* a downstream paper or external experiment requires nontrivial equivariance;
* the empirical correlation between offline analytical `grasp_success` and
  MuJoCo simulated lift success is computed (the result informs whether the
  current representation is empirically sufficient);
* a dependency group-convolutional library becomes available and stable.