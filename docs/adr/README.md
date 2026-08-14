# Architecture Decision Records

This directory contains lightweight records of architecturally significant
decisions for the project. Each ADR captures the context, decision,
rationale, and consequences of a single choice.

| ID    | Title                                                                              | Status   | Date       |
| ----- | ---------------------------------------------------------------------------------- | -------- | ---------- |
| 0001  | [Phase 4 representation: canonicalization vs. nontrivial equivariance](./001-phase4-canonicalization-vs-equivariant.md) | Accepted | 2026-08-12 |
| 0002  | [Flow-matching training: required deliverable vs. optional extension](./002-flow-training-optional-vs-required.md) | Accepted | 2026-08-12 |
| 0003  | [Flow model checkpoint contract: jointly train encoder + flow field](./003-flow-checkpoint-joint-encoder.md) | Accepted | 2026-08-12 |
| 0004  | [Dead-helper wiring, deduplication, and dependency retention](./004-dead-helper-wiring-and-refactoring.md) | Accepted | 2026-08-12 (updated 2026-08-13) |
| 0005  | [Runtime workflow integration scripts](./005-runtime-workflow-integration.md) | Accepted | 2026-08-12 (updated 2026-08-13) |
| 0006  | [Robot viewer and keyboard-topic teleoperation split](./006-robot-viewer-keyboard-topic-split.md) | Accepted | 2026-08-14 |

## Conventions

* Filenames use the pattern `NNNN-short-kebab-case-title.md` with a
  monotonically increasing zero-padded identifier.
* Each ADR has a `Status` field. Status transitions are recorded in the
  ADR body, not by renaming the file.
* ADRs describe decisions, not implementation details; the latter belong
  in code docstrings or design docs (`docs/PROJECT.md`).
* Superseded ADRs are marked `Superseded by ADR-NNNN` rather than deleted.