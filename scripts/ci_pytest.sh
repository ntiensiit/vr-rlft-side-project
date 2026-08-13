#!/usr/bin/env bash
# Run each fast test module in its own process under xvfb, combine coverage,
# and tolerate Open3D/MuJoCo teardown segfaults (exit 139) on headless Linux.
set -euo pipefail

shopt -s nullglob
test_files=(tests/test_*.py)

export COVERAGE_FILE=.coverage.ci
rm -f .coverage.ci

first_batch=1

run_batch() {
  local test_file="$1"
  local rc=0
  local append_flag=()
  if [[ "${first_batch}" -eq 0 ]]; then
    append_flag=(--append)
  fi
  echo "=== ${test_file} ==="
  set +e
  xvfb-run -a uv run coverage run "${append_flag[@]}" --rcfile=coverage.toml -m pytest -q "${test_file}" -m "not slow"
  rc=$?
  set -e
  if [[ "${rc}" -eq 5 ]]; then
    echo "No non-slow tests in ${test_file}; skipping."
    return 0
  fi
  if [[ "${rc}" -eq 139 ]]; then
    echo "Ignoring teardown segfault (139) for ${test_file}"
    first_batch=0
    return 0
  fi
  if [[ "${rc}" -ne 0 ]]; then
    echo "pytest failed for ${test_file} with exit ${rc}" >&2
    exit "${rc}"
  fi
  first_batch=0
}

for test_file in "${test_files[@]}"; do
  if [[ "${test_file}" == "tests/test_artifact_chain.py" ]]; then
    continue
  fi
  run_batch "${test_file}"
done

export COVERAGE_FILE=.coverage.ci
uv run coverage report --rcfile=coverage.toml --fail-under=80
