---
trigger: always_on
---

ADR (Architecture Decision Record) Maintenance Rule

`vr-rlft-side-project\docs\adr\` is the canonical record of architectural
decisions, design choices, and ongoing job state for this project.

Maintain it on every action that introduces, changes, or supersedes a
project-level decision:

- Before finalizing any architectural choice (representation, framework,
  data contract, evaluation semantics, pipeline selection, etc.), record the
  choice in a new ADR file under `docs/adr/` using the convention
  `NNNN-short-kebab-case-title.md` (zero-padded sequence number, kebab-case
  title) and the structure defined in `docs/adr/README.md`.
- When a previously recorded decision is changed, do not edit or delete the
  old file. Instead, add a new ADR that supersedes it, referencing the
  superseded ADR by id, and update the superseded file's `Status` field to
  `Superseded by ADR-NNNN`.
- When the active scope, deliverable scope, or task state changes (e.g.
  "flow training is required", "encoder is canonicalization-based", "YCB
  asset is the reference dataset"), capture the change in the relevant ADR's
  decision/consequences section, or add a new ADR if the change is large
  enough to warrant its own record.
- When the user provides new information that resolves an open question
  recorded in an ADR (e.g. "flow is optional"), update the corresponding ADR
  with the resolution and append a short rationale. Do not leave the
  resolution implicit in the codebase without an ADR entry.

The ADR index in `docs/adr/README.md` must be kept current: every new ADR
gets a row, every superseded ADR keeps its row but its status changes, and
every ADR referenced from `docs/PROJECT.md` or `docs/architecture.md` must
exist on disk.

When a code, configuration, or documentation change follows from an ADR,
reference the ADR id in the change's commit message and in any related
review notes. ADRs are the source of truth for "why"; code, configs, and
docs are the source of truth for "what".