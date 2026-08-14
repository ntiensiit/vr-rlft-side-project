# ADR-0006 — Robot viewer and keyboard-topic teleoperation split

## Status

Accepted (2026-08-14).

## Context

Interactive robot inspection was initially mixed into grasp simulation
(`run_simulation.py --render`) and a single teleoperation pipeline
(`pipelines/control_robot.py`). Users need:

1. **Headless batch simulation** for E2E workflow artifacts (no popup).
2. **A MuJoCo viewer** that loads robot/table/object scenes.
3. **A terminal TUI** that publishes keyboard commands without opening its
   own window.

The prior table-pick automation (`run_grasp_from_table`, position-only IK
waypoints) had no production script caller after the TUI split and duplicated
simulation concerns already covered by `simulate_grasp`.

## Decision

### Entry points

| Script | Module | Role |
| --- | --- | --- |
| `scripts/run_simulation.py` | `pipelines/simulate_grasp.py` | Headless MuJoCo grasp batch; writes JSON outcomes only. |
| `scripts/visualize_robot.py` | `pipelines/visualize_robot.py` | MuJoCo passive viewer; listens on UDP topic `robot/keyboard`. |
| `python -m grasping_ai.pipelines.visualize_robot --keyboard-tui` | `pipelines/visualize_robot.py` | Terminal TUI; publishes keycodes to `robot/keyboard`. |

### Keyboard topic contract

- Topic name: `robot/keyboard`
- Transport: UDP datagram on `127.0.0.1:5511`
- Payload: JSON `{"topic":"robot/keyboard","keycode":<int>}` where keycodes
  match GLFW codes used by `RobotController.handle_key`.

### Module boundaries

- **`pipelines/visualize_robot.py`**: scene load, home keyframe, viewer loop,
  `RobotController`, UDP keyboard topic bind/poll, and `run_keyboard_tui`.
- **Removed**: `pipelines/control_robot.py`, `grasping_ai.control` package,
  `GraspCommandPlayer`, and dead table-pick helpers with no script callers.

### Simulation render flag

Remove `--render` from `run_simulation.py`. Visualization is exclusively via
`visualize_robot.py`; simulation stays headless for workflow/CI compatibility.

## Rationale

- Separates **artifact generation** (headless) from **human inspection**
  (viewer + TUI).
- UDP topic avoids requiring MuJoCo window focus for teleoperation.
- Drops untested-in-production table-pick automation that overlapped with
  `simulate_grasp` IK/physics path.

## Consequences

- E2E demos that need a popup: run `visualize_robot.py` and
  `python -m grasping_ai.pipelines.visualize_robot --keyboard-tui` in two
  terminals; replay grasps visually only via a future dedicated replay script
  if needed.
- Do not re-add `--render` to `run_simulation.py` without a new ADR that
  addresses CI headless constraints.

## Follow-up (2026-08-14)

Consolidated keyboard teleoperation into `pipelines/visualize_robot.py` as
procedural functions (no new classes, no separate `grasping_ai.control`
package) to satisfy `.agents/rules/implementation-constraints.md`.

## Caller map

| Symbol | Callers |
| --- | --- |
| `run_keyboard_tui` | `python -m grasping_ai.pipelines.visualize_robot --keyboard-tui` |
| `run_robot_viewer` | `scripts/visualize_robot.py` |
| `run_simulation_sweep` | `scripts/run_simulation.py`, `run_workflow.py` |
