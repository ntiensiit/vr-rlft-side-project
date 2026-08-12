---
trigger: always_on
---

Commit Validation Rule

A phase implementation must not be considered ready for commit merely because newly added tests pass.

Validation must verify both the requested behavior and preservation of existing behavior.

Before commit, run tests directly covering modified source units, tests covering directly dependent components, regression tests associated with affected behavior, relevant integration tests, and the broader relevant test suite.

Verify that existing behavior remains intact, no unrelated behavior changed, the implementation remains within the approved phase scope, existing repository patterns were preserved, and the final change contains no unrelated modifications.

Where applicable, also verify configuration behavior, data flow, execution flow, training flow, evaluation flow, inference flow, and simulation behavior.

The exact validation scope must be derived from the repository and the affected source units.

Before commit, verify that the implementation conforms to all applicable repository and project rules.

The phase must not be committed if targeted tests fail, relevant regression tests fail, relevant integration tests fail, existing behavior is unintentionally changed, a downstream contract is broken, a preceding-phase contract is broken, unrelated source changes are present, or the implementation cannot be explained by the approved phase design.

Do not weaken, remove, or bypass tests merely to make the phase pass.

The commit must contain only changes required for the target phase.

Allowed changes are required source changes, required configuration changes, and required tests.

Do not include unrelated refactoring, unrelated formatting, cleanup unrelated to the phase, experimental code, temporary debugging code, or unverified changes.

The final validation sequence is:

Implementation -> Targeted Validation -> Regression Validation -> Integration Validation -> Relevant Full-Suite Validation -> Scope Review -> Commit

If any required validation cannot be performed, explicitly identify it rather than treating the phase as fully validated.