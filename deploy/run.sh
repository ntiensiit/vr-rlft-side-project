#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    pip install --upgrade pip
    pip install uv
fi

uv sync --all-extras
export PYTHONPATH="$(pwd)/src"

python scripts/run_artifacts.py