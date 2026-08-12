---
trigger: always_on
---

Cross-Phase Impact Rule

Every phase design must explicitly analyze its relationship with other implementation phases.

The analysis must distinguish preceding phases, the target phase, and downstream phases.

For every relevant preceding phase, determine its existing contract, the behavior it supplies, how the target phase consumes it, whether the target phase changes the contract, whether existing behavior remains compatible, which preceding-phase behavior must remain protected, and which existing tests should be considered during validation.

If no preceding phase is affected, explicitly state: No verified impact on preceding phases.

For every relevant downstream phase, determine its dependency on the target phase, required interface, required behavior, required data, required configuration, new capability provided by the target phase, and constraints that must remain stable.

Do not design the target phase in a way that unnecessarily constrains downstream implementation.

Define the stable contract produced by the target phase.

The contract should specify inputs, outputs, behavior, invariants, error behavior, configuration assumptions, data assumptions, and runtime assumptions where applicable.

The contract must contain only behavior supported by repository evidence or explicitly required by the approved phase roadmap.

Verify that no preceding-phase behavior is silently invalidated, no downstream phase depends on an undocumented behavior, no component is required before its dependency exists, no unnecessary dependency is introduced between phases, and phase boundaries remain meaningful.

If a phase dependency cannot be verified, mark it as Unverified.

Do not convert an assumption into a cross-phase requirement.