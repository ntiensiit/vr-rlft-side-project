"""Shared numerical tolerances for geometry, grasp search, and evaluation."""

from __future__ import annotations

# Vector norm thresholds used when comparing lengths or clamping divisors.
NORM_EPS = 1e-8
TORCH_NORM_CLAMP_MIN = 1e-8

# Degenerate-geometry checks in SE(3) frame construction.
DEGENERATE_SPAN_EPS = 1e-12
DEGENERATE_COMPONENT_EPS = 1e-9
TORCH_DEGENERATE_CLAMP_MIN = 1e-12

# Analytical grasp search and rotation validity.
GRASP_DISTANCE_EPS = 1e-4
ROTATION_DET_EPS = 1e-4

# Force-closure and convex-hull feasibility checks.
LP_FEASIBILITY_EPS = 1e-5
HULL_HALFSPACE_EPS = 1e-9

# Inverse-kinematics convergence tolerance in simulation.
IK_POSE_TOLERANCE = 1e-3
