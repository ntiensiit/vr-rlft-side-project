"""Artifact-chain verification test.

Runs ``scripts/run_artifacts.py`` from a clean working tree (except for the
shipped raw YCB assets) and asserts that the documented retained artifacts
are produced and contain valid content.

Marked ``slow`` so it can be skipped during fast iteration with
``pytest -m 'not slow'``.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
YCB_ROOT = ROOT / "data" / "raw" / "ycb"
RUNNER = ROOT / "scripts" / "run_artifacts.py"
ARTIFACTS = ROOT / "artifacts"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_OBSERVATIONS = ROOT / "data" / "observations"


@pytest.fixture(scope="module")
def chain_run():
    if not YCB_ROOT.is_dir():
        pytest.skip(f"YCB root not found: {YCB_ROOT}")
    if not RUNNER.is_file():
        pytest.fail(f"Artifact runner missing: {RUNNER}")

    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    completed = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            "Artifact chain failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


@pytest.mark.slow
def test_manifest_records_retained_artifacts(chain_run):
    manifest_path = ARTIFACTS / "manifest.json"
    assert manifest_path.is_file(), "manifest.json was not produced"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "generated" in manifest
    assert isinstance(manifest["generated"], list)
    assert "retained_artifacts" in manifest
    assert len(manifest["retained_artifacts"]) >= 10
    for rel in manifest["retained_artifacts"]:
        assert (ROOT / rel).is_file(), f"manifest references missing artifact: {rel}"


@pytest.mark.slow
def test_artifact_chain_produces_key_files(chain_run):
    expected = [
        ARTIFACTS / "manifest.json",
        ARTIFACTS / "checkpoints" / "grasp_generation.pt",
        ARTIFACTS / "checkpoints" / "rl_policy.pt",
        ARTIFACTS / "exports" / "generated_grasps.npy",
        ARTIFACTS / "reports" / "evaluation_report.json",
        ARTIFACTS / "reports" / "simulation_cracker.json",
        DATA_PROCESSED / "index.json",
        DATA_PROCESSED / "ycb_mjcf" / "003_cracker_box" / "object.xml",
        DATA_OBSERVATIONS / "003_cracker_box.npy",
        DATA_OBSERVATIONS / "gripper.npy",
    ]
    missing = [str(p) for p in expected if not p.is_file()]
    assert not missing, f"missing retained artifacts: {missing}"


@pytest.mark.slow
def test_evaluation_report_uses_grasp_success_key(chain_run):
    report = json.loads(
        (ARTIFACTS / "reports" / "evaluation_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert "success_rate" in report
    assert "collision_free_rate" in report
    assert "force_closure_rate" in report
