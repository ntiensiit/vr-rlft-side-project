---
trigger: always_on
---

Test Design Rule

Every implementation design must include a concrete test strategy for the target phase.

Inspect the existing test suite before proposing tests.

Follow the repository's established test framework, test naming, test organization, fixture conventions, mocking strategy, assertion style, parameterization strategy, and integration-test conventions.

Do not introduce a new testing framework or testing pattern when the existing test infrastructure is sufficient.

The test design must cover behavior affected by the phase.

Where applicable, include normal behavior, boundary conditions, invalid input, failure behavior, state transitions, configuration behavior, direct dependencies, integration behavior, and regression behavior.

Do not create tests merely to increase test count.

Each test must protect a meaningful behavior or contract.

For every new or modified test, specify the test file, test function, target source unit, scenario, setup, input, expected behavior, assertions, relevant edge cases, and failure condition.

Identify existing behavior that could be affected by the phase.

For each regression risk, define the existing behavior, potential failure mechanism, test protecting the behavior, and expected result.

Distinguish between unit-level regression, integration-level regression, and pipeline-level regression.

Prefer modifying existing tests when an existing test already represents the behavior being changed.

Create a new test only when existing tests cannot adequately represent the new behavior.

Do not redesign the test architecture as part of the phase.

The test suite must validate the implementation against the actual source behavior rather than against assumptions from documentation.