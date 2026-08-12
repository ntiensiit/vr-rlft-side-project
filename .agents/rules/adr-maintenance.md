---
trigger: always_on
---

ADR (Architecture Decision Record) + CHANGELOG Maintenance Rule

`vr-rlft-side-project\docs\adr\` is the canonical record of architectural
decisions, design choices, and ongoing job state for this project.
`vr-rlft-side-project\CHANGELOG.md` is the high-level user-facing summary
of every change. Both must be maintained together on every action that
introduces, changes, or supersedes a project-level decision or ships a
user-visible change.

## ADR (`docs/adr/`) maintenance

Maintain on every action that introduces, changes, or supersedes a
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
- Keep `CHANGELOG.md` consistent: ADRs are the source of truth for "why";
  `CHANGELOG.md` is the source of truth for the user-visible "what" of
  each release. ADRs cite each other by id; `CHANGELOG.md` points at ADR
  ids for major architectural decisions and summarizes the rest in the
  Added/Changed/Removed/Fixed buckets.

The ADR index in `docs/adr/README.md` must be kept current: every new ADR
gets a row, every superseded ADR keeps its row but its status changes, and
every ADR referenced from `docs/PROJECT.md` or `docs/architecture.md` must
exist on disk.

## CHANGELOG (`CHANGELOG.md`) maintenance

Maintain `vr-rlft-side-project\CHANGELOG.md` on every action that ships a
user-visible change (added feature, changed behavior, removed API,
fixed bug, dependency change, breaking change to a contract). Conventions:

- Keep a single top-level `[Unreleased]` section until a release is cut.
- Do not edit past release sections; cut a new `[X.Y.Z] - DATE` section
  instead. If a previous release section needs correction, add a new
  bullet under `[Unreleased]` or a new patch release section.
- Group entries under `### Added`, `### Changed`, `### Removed`,
  `### Fixed` (and optionally `### Deprecated`, `### Security`) inside
  the active section.
- For architectural or scope changes, reference the corresponding ADR id
  in `docs/adr/NNNN-*.md` rather than re-stating the rationale. Example:
  `> See docs/adr/003-flow-checkpoint-joint-encoder.md for context.`
- Keep entries concise: one or two sentences per change plus pointers to
  the files involved. Do not paste full diffs or rationale into the
  changelog; the ADR or the commit/PR description holds that detail.
- If an entry is solely a documentation-only change with no behavior
  impact, still record it under `### Changed` so the user-visible delta is
  traceable.

When a code, configuration, or documentation change follows from an ADR,
reference the ADR id in the change's commit message and in any related
review notes. ADRs are the source of truth for "why"; code, configs, and
docs are the source of truth for "what"; `CHANGELOG.md` is the source of
truth for "what changed when".