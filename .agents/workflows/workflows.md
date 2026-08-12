---
trigger: always_on
name: audit-implement-verify
description: >
  Default end-to-end workflow for vr-rlft-side-project tasks. Runs on every
  prompt unless the user explicitly requests a read-only or docs-only pass.
---

# Workflow: Audit, Checklist, Implement, Verify, ADR, Changelog

Apply this workflow to **every user prompt** in `vr-rlft-side-project` unless the
user explicitly restricts scope (e.g. "analysis only", "do not change code",
"update CHECKLIST only").

Follow existing rules in `.agents/rules/` (especially `design-verification.md`,
`adr-maintenance.md`, `source-change-scope.md`, `test-design.md`).

## Pipeline (strict order)

1. Deep-dive audit and inspect
2. Report
3. Update `CHECKLIST.md`
4. Proceed (implement or fix)
5. Run test suites
6. Update `docs/adr`
7. Update `CHANGELOG.md`

Do not skip stages. Do not reorder unless the user explicitly asks for a
read-only audit (stages 1 through 3 only) or docs-only update (stages 6 and 7
only).

## Stage 1 — Deep-dive audit and inspect

**Goal:** Evidence-based understanding before any edit.

1. **Parse the prompt** — infer goal, constraints, and whether the ask is
   implementation, refactor, research validation, or documentation.
2. **Inspect selectively but deeply** — source is authoritative over docs:
   - `src/grasping_ai/` modules relevant to the prompt
   - `scripts/` entry points and their pipeline callers
   - `tests/` covering the touched behavior
   - `docs/adr/` for existing decisions that constrain the change
   - `CHECKLIST.md` for open items the prompt may address
3. **Audit lenses** (use those relevant to the prompt):
   - **Duplicate code** — repeated logic across pipelines, scripts, or modules
   - **Dead code** — exported/tested symbols with no production callers
   - **Obsolete code** — stale docs, notebooks, compat shims, deprecated paths
   - **Contract drift** — docs vs code, checkpoint schemas, grasp I/O formats
   - **Test gaps** — missing coverage for changed paths
4. **Record evidence** — file paths, symbols, call sites, test names. Mark
   unverified claims explicitly.

**Commands (read-only exploration):**

```bash
cd vr-rlft-side-project
uv run ruff check src tests scripts
uv run mypy src
uv run pytest -q
```

Use `grep`, targeted file reads, and call-site tracing; do not invent symbols
or APIs.

## Stage 2 — Report

**Goal:** Short, actionable summary for the user before editing.

Deliver a report containing:

| Section | Content |
| --- | --- |
| Intent | What the user asked for, restated in one sentence |
| Findings | Audit results with evidence (paths, symbols) |
| Recommended actions | Ordered list of concrete changes |
| Risk / scope | What must not change; ADR constraints |
| Checklist impact | Items to add, close, or leave open |

Keep the report concise. Do not implement yet unless the user said "proceed"
in the same prompt. Pause after Stage 3 for approval when the change set is
large or ambiguous.

If the user prompt already includes "proceed", continue through Stage 4 in the
same turn after updating the checklist.

## Stage 3 — Update CHECKLIST.md

**Goal:** Single source of open work; remove noise.

File: `vr-rlft-side-project/CHECKLIST.md`

Rules:

- **Open items only** — remove completed items; do not keep historical done
  lists in the checklist.
- **Actionable wording** — each open item must be verifiable.
- **Categories:**
  - Cross-cutting verification (CI, verify gate stats)
  - Implementation / refactor (when audit finds new work)
  - Research validation (experiments; note "no source changes required")
- **Verify gate block** — always keep at top:

```bash
uv run pytest -q
uv run ruff check src tests scripts
uv run mypy src
```

- **Update Last verification** only after Stage 5 passes.
- **Point to ADRs** for completed architectural work instead of duplicating
  rationale in the checklist.

When closing items, ensure the implementation exists before removing them.

## Stage 4 — Proceed (implement)

**Goal:** Smallest correct diff that satisfies the prompt and checklist.

Rules (from `.agents/rules/`):

- Match existing naming, types, imports, and module boundaries.
- Wire helpers into production rather than deleting tested surface unless
  the user or an ADR explicitly directs removal.
- Minimize scope; no drive-by refactors.
- Update or add tests when behavior changes.
- Do not commit unless the user asks.

Implementation order:

1. Shared modules and contracts
2. Pipelines and scripts
3. Tests
4. Docs touched by behavior change

## Stage 5 — Run test suites

**Goal:** Verification gate must pass before ADR and changelog updates.

From `vr-rlft-side-project/`:

```bash
uv run ruff check src tests scripts
uv run mypy src
uv run pytest -q
```

Optional when relevant:

```bash
uv run pytest -q -m 'not slow'
uv run pytest -q tests/test_artifact_chain.py
```

**Failure policy:** Fix failures before Stage 6. Do not mark checklist items
done or update verification stats until all three gate commands pass.

Record in the final summary: test count, coverage percentage, ruff and mypy
status.

## Stage 6 — Update docs/adr

**Goal:** Record why for architectural or contract changes.

Follow `docs/adr/README.md` and `.agents/rules/adr-maintenance.md`.

| Situation | Action |
| --- | --- |
| New architectural choice | Add `NNNN-short-kebab-case-title.md`; update index in `docs/adr/README.md` |
| Extends existing decision | Append follow-up section to relevant ADR; note update date |
| Supersedes prior decision | New ADR; mark old ADR superseded by id |
| Docs-only / no decision change | Skip new ADR; no edit required |

ADRs hold rationale and caller maps. Do not paste full diffs.

## Stage 7 — Update CHANGELOG.md

**Goal:** Concise user-visible what under `[Unreleased]`.

Follow `.agents/rules/adr-maintenance.md` changelog section.

Rules:

- Keep entries short: one line per theme; link ADR ids for rationale.
- Buckets: Added, Changed, Removed, Fixed.
- Do not edit released sections.
- Do not duplicate ADR prose.

Example entry:

```markdown
- Dual inference CLIs share runtime helpers; grasp I/O formats reconciled — [ADR-0005](docs/adr/005-runtime-workflow-integration.md).
```

## Mode overrides

| User says | Stages to run |
| --- | --- |
| default | 1 through 7 |
| audit only / analysis only / do not change code | 1 through 3 |
| proceed (after audit) | 3 through 7 |
| update checklist only | 3 |
| update docs only | 6 and 7 |

## Deliverables (end of run)

Before finishing, confirm:

- Audit report delivered (Stage 2)
- `CHECKLIST.md` reflects current open work (Stage 3)
- Code changes implemented if in scope (Stage 4)
- ruff, mypy, pytest pass (Stage 5)
- `docs/adr/` updated when decisions changed (Stage 6)
- `CHANGELOG.md` `[Unreleased]` updated when user-visible behavior changed (Stage 7)

## Key paths

| Artifact | Path |
| --- | --- |
| Checklist | `CHECKLIST.md` |
| Changelog | `CHANGELOG.md` |
| ADR index | `docs/adr/README.md` |
| Agent rules | `.agents/rules/` |
| Verify gate | `uv run pytest -q && uv run ruff check src tests scripts && uv run mypy src` |
