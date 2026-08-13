#!/usr/bin/env bash
# Run each fast test module in its own process under xvfb, then combine coverage.
set -euo pipefail

shopt -s nullglob
test_files=(tests/test_*.py)
coverage_files=()

run_batch() {
  local test_file="$1"
  local module_id
  module_id="$(basename "${test_file}" .py)"
  local data_file="cov-ci-${module_id}"
  local rc=0

  echo "=== ${test_file} ==="
  set +e
  (
    export COVERAGE_FILE="${data_file}"
    xvfb-run -a uv run coverage run --rcfile=coverage.toml -m pytest -q "${test_file}" -m "not slow"
  )
  rc=$?
  set -e

  if [[ "${rc}" -eq 5 ]]; then
    echo "No non-slow tests in ${test_file}; skipping."
    return 0
  fi
  if [[ "${rc}" -ne 0 ]]; then
    echo "pytest failed for ${test_file} with exit ${rc}" >&2
    exit "${rc}"
  fi
  if [[ -f "${data_file}" ]]; then
    coverage_files+=("${data_file}")
  else
    echo "Expected coverage data file missing: ${data_file}" >&2
    exit 1
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
rm -f .coverage .coverage.*

if ((${#coverage_files[@]} == 1)); then
  cp "${coverage_files[0]}" .coverage
else
  cp "${coverage_files[0]}" .coverage
  for fragment in "${coverage_files[@]:1}"; do
    uv run coverage combine --rcfile=coverage.toml "${fragment}"
  done
fi
uv run coverage report --rcfile=coverage.toml --fail-under=80
