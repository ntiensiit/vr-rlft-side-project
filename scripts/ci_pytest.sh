#!/usr/bin/env bash
# Run fast tests one module at a time under xvfb to avoid native-library
# teardown segfaults on headless Linux CI runners (Open3D / MuJoCo / torch).
set -euo pipefail

shopt -s nullglob
test_files=(tests/test_*.py)

if ((${#test_files[@]} == 0)); then
  echo "No test files matched tests/test_*.py" >&2
  exit 1
fi

rm -f .coverage.ci
export COVERAGE_FILE=.coverage.ci

for test_file in "${test_files[@]}"; do
  echo "=== ${test_file} ==="
  set +e
  xvfb-run -a uv run coverage run --append --rcfile=coverage.toml -m pytest -q "${test_file}" -m "not slow"
  rc=$?
  set -e
  if [[ ${rc} -eq 5 ]]; then
    echo "No non-slow tests in ${test_file}; skipping."
    continue
  fi
  if [[ ${rc} -ne 0 ]]; then
    exit "${rc}"
  fi
done

uv run coverage report --rcfile=coverage.toml --fail-under=80
