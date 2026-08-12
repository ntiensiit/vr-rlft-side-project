---
trigger: always_on
---

Design Verification Rule

Before producing any implementation design, inspect and verify the repository against the requested phase.

The source code is the primary source of truth. Do not derive implementation requirements solely from README files, documentation, directory names, comments, or the phase description.

Selectively inspect only the repository areas relevant to the target phase, while following dependencies far enough to understand their actual behavior.

Verify the repository structure, relevant source modules, entry points, imports, dependencies, call sites, existing interfaces, configuration, runtime flow, data flow, existing tests, existing test conventions, relevant preceding phases, and relevant downstream dependencies.

If the repository is related to robotics, simulation, machine learning, or reinforcement learning, inspect the relevant environment, observation, action, reward, policy or model, training loop, evaluation, inference, simulation, and data pipeline when they are actually present.

Before producing the design, determine which behavior is actually implemented, which source units are relevant to the phase, which dependencies are real, which existing behavior must be preserved, which assumptions are supported by repository evidence, and which information cannot be verified.

Do not invent files, symbols, dependencies, APIs, configuration entries, commands, tests, or execution flows.

When documentation conflicts with source code, treat the source code as authoritative and record the discrepancy.

Every major design decision must be traceable to repository evidence such as a file path, symbol, call site, import, configuration entry, existing test, or dependency relationship.

If a required fact cannot be verified from the repository, explicitly mark it as Unverified.

Do not expand the implementation scope merely because additional changes appear desirable.

The final design must represent the smallest verified implementation scope that satisfies the target phase.