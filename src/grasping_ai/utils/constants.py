"""Shared shape, geometry, and numerical constants for grasping pipelines."""

from __future__ import annotations

POINT_CLOUD_NDIM = 2
SPATIAL_DIM = 3
GRASP_VECTOR_DIM = 9
GRASP_POSES_NDIM = 3
SE3_MATRIX_SHAPE = (4, 4)
QUATERNION_DIM = 4
WRENCH_DIM = 6
WRENCH_LP_EQUALITY_ROWS = 7
MIN_WRENCH_COLUMNS = 6
MIN_ANTIPODAL_CONTACTS = 2
DUAL_GRIPPER_COUNT = 2
ALIGNMENT_DOT_THRESHOLD = 0.9
OBSERVATION_NDIM = 2
GRASP_OBJECT_BATCH_NDIM = 4
MIN_NORMAL_NEIGHBORHOOD = 3

# GLFW key codes used by the robot teleoperation viewer.
KEY_DIGIT_1 = 49
KEY_DIGIT_9 = 57
KEY_GRIPPER_TOGGLE = 71
KEY_HOME = 72
KEY_SPACE = 32

# Teleoperation step sizes and UDP port bounds.
ACTUATOR_SPAN_LARGE_THRESHOLD = 10.0
ACTUATOR_STEP_LARGE = 15.0
ACTUATOR_STEP_SMALL = 0.05
UDP_PORT_MIN = 0
UDP_PORT_MAX = 65535

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
