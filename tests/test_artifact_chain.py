"""Tests for artifact generation and chaining."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from grasping_ai.pipelines.evaluate import read_jsonl_records

ROOT = Path(__file__).resolve().parents[1]
YCB_ROOT = ROOT / "data" / "raw" / "ycb"
RUNNER = ROOT / "scripts" / "run_artifacts.py"
ARTIFACTS = ROOT / "artifacts"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_OBSERVATIONS = ROOT / "data" / "observations"
MIN_COMMAND_RECORDS = 8
MIN_RETAINED_ARTIFACTS = 10
MIN_OBJECT_RECORDS = 3


@pytest.fixture(scope="module")
def chain_run() -> subprocess.CompletedProcess[str]:
    """Fixture to run the full artifact pipeline and return the subprocess result.

    Skips the tests if YCB assets are missing, and fails if the script runner is absent
    or if the execution returns a non-zero exit code.
    """
    if not YCB_ROOT.is_dir():
        pytest.skip(f"YCB root not found: {YCB_ROOT}")
    if not RUNNER.is_file():
        pytest.fail(f"Artifact runner missing: {RUNNER}")

    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONPYCACHEPREFIX": str(ROOT / ".pycache"),
    }
    completed = subprocess.run(  # noqa: S603  # fixed internal runner script, no untrusted input
        [sys.executable, str(RUNNER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(f"Artifact chain failed:\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    return completed


@pytest.mark.slow
def test_manifest_records_retained_artifacts(chain_run: subprocess.CompletedProcess[str]) -> None:
    """Verify that the manifest file correctly logs retained artifacts and execution commands.

    Ensures command logs use relative paths and that all referenced artifacts exist on disk.
    """
    _ = chain_run
    manifest_path = ARTIFACTS / "manifest.jsonl"
    if not (manifest_path.is_file()):
        msg = "manifest.jsonl was not produced"
        raise AssertionError(msg)
    records = read_jsonl_records(manifest_path)
    manifest_headers = [r for r in records if r.get("record_type") == "manifest"]
    commands = [r for r in records if r.get("record_type") == "command"]
    retained = [r for r in records if r.get("record_type") == "retained_artifact"]
    if not (manifest_headers):
        raise AssertionError
    if not (len(commands) >= MIN_COMMAND_RECORDS):
        raise AssertionError
    if not (len(retained) >= MIN_RETAINED_ARTIFACTS):
        raise AssertionError
    root_posix = ROOT.as_posix()
    for record in commands:
        if not (record["cwd"] == "."):
            raise AssertionError
        if not (root_posix not in str(record["command"]).replace("\\", "/")):
            raise AssertionError
    for record in retained:
        rel = str(record["path"])
        if not ((ROOT / rel).is_file()):
            msg = f"manifest references missing artifact: {rel}"
            raise AssertionError(msg)


@pytest.mark.slow
def test_artifact_chain_produces_key_files(chain_run: subprocess.CompletedProcess[str]) -> None:
    """Verify that executing the full artifact pipeline produces all expected outputs.

    Checks for the existence of checkpoints, dataset exports, reports, processed
    objects, and observed state files.
    """
    _ = chain_run
    expected = [
        ARTIFACTS / "manifest.jsonl",
        ARTIFACTS / "checkpoints" / "diffusion_grasp_generator.pt",
        ARTIFACTS / "checkpoints" / "rl_grasp_policy.pt",
        ARTIFACTS / "exports" / "diffusion_grasp_candidates_by_object.npy",
        ARTIFACTS / "reports" / "diffusion_analytical_evaluation_report.jsonl",
        ARTIFACTS / "reports" / "diffusion_simulation_outcomes_003_cracker_box.jsonl",
        DATA_PROCESSED / "index.json",
        DATA_PROCESSED / "ycb_mjcf" / "003_cracker_box" / "object.xml",
        DATA_OBSERVATIONS / "003_cracker_box.npy",
        DATA_OBSERVATIONS / "gripper.npy",
    ]
    missing = [str(p) for p in expected if not p.is_file()]
    if missing:
        msg = f"missing retained artifacts: {missing}"
        raise AssertionError(msg)


@pytest.mark.slow
def test_evaluation_report_uses_grasp_success_key(chain_run: subprocess.CompletedProcess[str]) -> None:
    """Verify that the generated evaluation summary includes expected success metrics.

    Checks that success_rate, collision_free_rate, and force_closure_rate are logged.
    """
    _ = chain_run
    records = read_jsonl_records(ARTIFACTS / "reports" / "diffusion_analytical_evaluation_report.jsonl")
    object_records = [r for r in records if r.get("record_type") == "object"]
    summary_records = [r for r in records if r.get("record_type") == "summary"]
    if not (len(object_records) >= MIN_OBJECT_RECORDS):
        raise AssertionError
    report = summary_records[0]
    if "success_rate" not in report:
        raise AssertionError
    if "collision_free_rate" not in report:
        raise AssertionError
    if "force_closure_rate" not in report:
        raise AssertionError
