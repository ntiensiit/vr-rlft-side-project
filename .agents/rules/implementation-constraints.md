---
trigger: always_on
---

Implementation Constraints Rule

All implementation proposed for the target phase must comply with the following constraints.

Do not create helper functions, utility functions, helper classes, utility classes, generic abstraction layers, wrapper functions whose only purpose is delegation, or generic convenience APIs.

Reuse existing functionality when it already exists.

Implement phase-specific behavior directly within the appropriate existing source unit.

Do not introduce an abstraction solely to reduce local code repetition.

Do not create new classes.

Do not convert existing procedural functionality into classes.

Do not introduce object-oriented abstractions.

Existing classes may only be modified when the target phase explicitly requires modification of an existing class.

Do not redesign existing classes as part of unrelated cleanup.

Do not introduce global variables, global constants, module-level mutable state, or module-level configuration state.

Keep values within the smallest appropriate execution scope.

Use the repository's existing configuration mechanism when configuration is required.

Do not create module-level constants merely to avoid repeating a local literal.

Do not add file-level descriptions, module banners, generated-file headers, or header comments describing the purpose of the source file.

Do not add explanatory comments solely to describe what a file contains.

Only implementation comments required to explain genuinely non-obvious behavior may be introduced, and they must follow the existing repository convention.

Every function introduced or modified by the implementation must have a Google-style docstring.

The docstring must describe the function according to its actual behavior.

Where applicable, use the Google-style sections Args, Returns, and Raises.

Do not add unnecessary documentation to untouched code.

Do not use docstrings to compensate for unclear implementation structure.

Do not include unrelated refactoring, unrelated formatting changes, file reorganization, naming cleanup, dependency replacement, architectural redesign, experimental code, or temporary debugging code unless explicitly required by the target phase.

The implementation must remain the smallest change that satisfies the verified phase requirements.