---
trigger: always_on
---

Source Change Scope Rule

The implementation design must explicitly identify the exact source units affected by the target phase.

Do not describe the scope using only directories, packages, or vague references such as relevant files.

Identify source units precisely whenever possible, including file paths, functions, existing class methods, existing module-level execution blocks, configuration entries, test functions, test fixtures, and test configuration.

For every affected source unit, document its exact file path, exact symbol or execution unit, current responsibility, reason it must change, required behavior after the change, existing behavior that must remain unchanged, direct dependencies affected, and tests associated with the change.

Only include source units that have been verified as affected.

The design must distinguish between files that must change, files that must be inspected but remain unchanged, files that are dependencies but do not require modification, and files that are explicitly out of scope.

Do not include speculative files.

Do not propose unrelated refactoring.

Do not expand the change scope merely to improve code organization, naming, readability, or abstraction.

The design must provide a source-unit change matrix containing the file, symbol, change type, current responsibility, required change, and affected dependencies.

Change types must describe the actual operation, such as Modify, Extend, Remove, Configuration Change, or Test Change.

The scope must remain limited to the target phase.