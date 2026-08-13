#!/usr/bin/env bash
# Run each fast test module in its own process under xvfb, combine coverage,
# and tolerate Open3D/MuJoCo teardown segfaults (exit 139) on headless Linux.
set -euo pipefail

shopt -s nullglob
test_files=(tests/test_*.py)
coverage_files=()

run_batch() {
  local test_file="$1"
  local module_id
  module_id="$(basename "${test_file}" .py)"
  local data_file=".coverage.ci.${module_id}"
  local rc=0

  echo "=== ${test_file} ==="
  set +e
  COVERAGE_FILE="${data_file}" xvfb-run -a uv run coverage run --rcfile=coverage.toml -m pytest -q "${test_file}" -m "not slow"
  rc=$?
  set -e

  if [[ "${rc}" -eq 5 ]]; then
    echo "No non-slow tests in ${test_file}; skipping."
    return 0
  fi
  if [[ "${rc}" -eq 139 ]]; then
    echo "Ignoring teardown segfault (139) for ${test_file}"
    if [[ -f "${data_file}" ]]; then
      coverage_files+=("${data_file}")
    fi
    return 0
  fi
  if [[ "${rc}" -ne 0 ]]; then
    echo "pytest failed for ${test_file} with exit ${rc}" >&2
    exit "${rc}"
  fi
  if [[ -f "${data_file}" ]]; then
    coverage_files+=("${data_file}")
  fi
}

for test_file in "${test_files[@]}"; do
  if [[ "${test_file}" == "tests/test_artifact_chain.py" ]]; then
    continue
  fi
  run_batch "${test_file}"
done

if ((${#coverage_files[@]} == 0)); then
  echo "No coverage data files were produced" >&2
  exit 1
fi

export COVERAGE_FILE=.coverage
uv run coverage combine "${coverage_files[@]}"
uv run coverage report --rcfile=coverage.toml --fail-under=80
