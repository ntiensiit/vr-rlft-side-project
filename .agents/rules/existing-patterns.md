---
trigger: always_on
---

Existing Pattern Preservation Rule

All implementation designs must preserve the established patterns of the repository.

Before proposing a change, identify the existing repository pattern that performs the same or a closely related responsibility.

The design must explain which existing pattern is being followed, where that pattern exists, why it is applicable to the target change, and how the proposed change remains consistent with it.

Prefer existing repository conventions for module organization, function structure, naming, data flow, configuration, error handling, logging, dependency management, test organization, fixture usage, assertions, parameterization, integration boundaries, and execution flow.

Do not introduce a new architectural pattern when an established pattern already satisfies the requirement.

Do not replace an existing pattern merely because another approach appears cleaner or more modern.

Do not reorganize unrelated modules.

Do not rename existing components unless the target phase explicitly requires the rename.

Do not introduce a new dependency when an existing repository dependency already provides the required capability.

Do not perform architectural cleanup unrelated to the target phase.

When an existing pattern must be extended rather than replaced, describe the extension explicitly.

When no existing pattern is available, state this clearly and justify the minimum new design required by the phase.

The goal is behavioral and architectural consistency with the existing repository, not architectural redesign.