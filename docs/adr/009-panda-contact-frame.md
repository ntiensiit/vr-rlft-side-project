# ADR-0009 — Panda contact-frame grasps and sim fidelity

## Status

Accepted (2026-08-14)

## Context

Synthetic grasps from ``generate_analytical_grasps`` place the grasp origin at the
antipodal **contact midpoint** (contact frame). MuJoCo inverse kinematics and the
Panda MJCF operate on the **hand** body frame. The MuJoCo Grasping Simulator
(`mj-grasp-sim`) defines an explicit ``base_to_contact`` offset for the Franka
Panda gripper and uses high fingertip friction on pad collision geoms.

Without the offset, ``simulate_grasp`` targeted the hand frame using contact-frame
poses, introducing a systematic placement error. Fingertip pads in
``deploy/robot.xml`` also omitted the high-friction coefficients used in
reference Panda models.

## Decision

1. Document that dataset and analytical grasps remain **contact-frame** SE(3)
   poses; simulation converts to **hand-frame** before IK.
2. Store Panda numeric constants in ``configs/gripper/default.yaml`` (width
   limits, joint ranges, ``base_to_contact`` translation and wxyz quaternion).
3. Implement ``panda_hand_to_contact_transform`` and
   ``panda_width_to_finger_joints`` in ``grasping_ai.robotics.gripper`` using
   the same values as the reference simulator.
4. Apply ``hand_pose = contact_pose @ invert(T_hand_contact)`` in
   ``simulate_grasp`` before IK and FK error checks.
5. Set fingertip pad ``friction="2.4 0.3 0.1"`` on Panda MJCF defaults in
   ``deploy/robot.xml`` and ``deploy/franka_emika_panda/panda.xml``.
6. Optional ``grasp_width`` on ``simulate_grasp`` maps to finger joint targets
   when two finger actuators are present; default ``None`` preserves existing
   tendon-based close commands on the full-arm model.

## Consequences

- Existing ``.npy`` grasp records require no schema change.
- Callers of ``simulate_grasp`` continue passing contact-frame poses; conversion
  is internal.
- Analytical ``metrics.friction_coefficient`` (force closure) remains separate
  from MuJoCo contact friction on pad geoms.

## Follow-up review triggers

Revisit when the robot MJCF or end-effector body used for IK diverges from the
Panda ``hand`` body assumed by ``base_to_contact``.
