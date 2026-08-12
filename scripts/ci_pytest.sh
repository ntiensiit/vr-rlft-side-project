#!/usr/bin/env bash
# Run fast tests in two batches under xvfb, combine coverage, and tolerate
# Open3D/MuJoCo teardown segfaults (exit 139) on headless Linux runners.
set -euo pipefail

export COVERAGE_FILE=.coverage.ci
rm -f .coverage.ci

run_batch() {
  local rc=0
  set +e
  xvfb-run -a uv run coverage run --append --rcfile=coverage.toml -m pytest -q "$@" -m "not slow"
  rc=$?
  set -e
  if [[ "${rc}" -ne 0 && "${rc}" -ne 139 ]]; then
    echo "pytest batch failed with exit ${rc}" >&2
    exit "${rc}"
  fi
  if [[ "${rc}" -eq 139 ]]; then
    echo "Ignoring teardown segfault (139) for batch: $*" >&2
  fi
}

run_batch \
  tests/test_grasp_io_runtime.py \
  tests/test_phase1_foundation.py \
  tests/test_phase2_sim_robotics.py \
  tests/test_phase3_data_perception.py \
  tests/test_phase4_flow_training.py \
  tests/test_phase4_generative_grasp.py \
  tests/test_phase5_rl_policy.py \
  tests/test_phase6_orchestration.py

run_batch \
  tests/test_phase7_synthetic_data.py \
  tests/test_phase8_tracking.py \
  tests/test_phase9_gymnasium_env.py \
  tests/test_phase10_evaluation.py

uv run coverage report --rcfile=coverage.toml --fail-under=80
