# Checklist

Open implementation and research tasks for the grasping-ai side project.

## Verification evidence (keep separate)

| Evidence | Meaning | Current state |
| --- | --- | --- |
| Local gate pass | Engineering verification on this machine | 226 passed, 85.74% coverage, ruff clean, mypy clean on 48 source files (2026-08-13) |
| Record in this file | Repository-recorded verification | Same as local gate above |
| GitHub Actions on current `dev` HEAD | GitHub CI verified | Run #37 (`935a012`) pending confirmation; matrix run #30 proved all 12 fast modules pass on Linux |

Local verify gate (run after each code change):

```bash
cd vr-rlft-side-project
uv run ruff check src tests scripts
uv run mypy src
uv run pytest -q
```

Completed wiring and refactor work: [ADR-0004](docs/adr/004-dead-helper-wiring-and-refactoring.md), [ADR-0005](docs/adr/005-runtime-workflow-integration.md).

## Cross-cutting verification

- [ ] **Current GitHub CI status confirmed** on the current `dev` HEAD commit.

### Execution plan: GitHub CI verification

**Pass/fail:** Mark the item above `[x]` only when the current `dev` HEAD has a completed GitHub Actions run and every required job passed (not skipped, not `continue-on-error`).

Required jobs in workflow `CI`: `Lint and typecheck`, `Fast tests and coverage`.

**Step 1 — Push current `dev` commit**

Commands:

```bash
git status
git log -1 --oneline
git push origin dev
git rev-parse HEAD
```

Expected: working tree clean (or only intentionally uncommitted files); push succeeds.

Pass/fail: record the SHA from `git rev-parse HEAD`. This SHA is the commit that must appear on the GitHub run.

**Step 2 — Open Actions for this repository**

URL: https://github.com/ntiensiit/vr-rlft-side-project/actions

Find the workflow run for workflow `CI` on branch `dev` that matches the SHA from Step 1.

Pass/fail: if no run exists for that SHA, status is **UNKNOWN** (not complete).

**Step 3 — Confirm workflow commit**

In the run details, confirm:

- Branch: `dev`
- Commit: exact match to `git rev-parse HEAD`

Pass/fail: mismatch or run on an older SHA means **UNKNOWN**.

**Step 4 — Confirm job executed (not skipped)**

Workflow file: `.github/workflows/ci.yml`

Single job: `Lint, typecheck, and fast tests` (`jobs.fast`).

Inspect the run: the job must have executed. A green workflow with a skipped required job does not count.

**Step 5 — Ruff step**

Step name in log: `Ruff`

Command in CI: `uv run ruff check src tests scripts`

Pass/fail: step exits 0; log shows no ruff errors.

**Step 6 — Mypy step**

Step name in log: `Mypy`

Command in CI: `uv run mypy src`

Pass/fail: step exits 0; log reports success on source files.

**Step 7 — Pytest and coverage step**

Step name in log: `Pytest (excluding slow / artifact-chain tests)`

Command in CI: `bash scripts/ci_pytest.sh` (two `coverage run` batches under `xvfb-run`, then `coverage report --fail-under=80`)

Pass/fail: step exits 0; tests ran (not skipped); coverage gate at or above 80%.

Note: CI excludes `@pytest.mark.slow` tests (artifact chain). Full local suite is 226 passed; CI fast suite is smaller by three slow tests. Both are valid; do not treat a green CI run as proof the slow artifact-chain tests ran on GitHub.

**Step 8 — Record evidence**

When all steps pass, add under this section (replace placeholders):

```text
GitHub CI verification
Branch: dev
Commit: <SHA>
Workflow: CI
Job: Lint, typecheck, and fast tests
Ruff: PASS
Mypy: PASS
Pytest (not slow): PASS
Coverage (>= 80%): PASS
Overall GitHub CI: PASS
Verified: <YYYY-MM-DD>
```

Then change the checklist item to `[x]`.

**Latest run (2026-08-13):** Matrix run https://github.com/ntiensiit/vr-rlft-side-project/actions/runs/31640843518 — all 12 `Pytest tests/*` jobs PASS on Linux; coverage aggregation fixed in `935a012` (`scripts/ci_pytest.sh`).

---

## Research validation (no source changes required)

These tasks validate learned behavior and research claims. Each item is independent unless noted.

### 1. Physical vs analytical evaluation correlation

- [ ] Correlate offline `grasp_success` against MuJoCo simulated lift success.

Commands (illustrative; adjust paths to trained checkpoints):

```bash
uv run python scripts/evaluate.py --help
uv run python scripts/run_simulation.py --help
```

Pass/fail: report correlation over many grasp candidates and sufficient checkpoints; document sample size and correlation metric.

### 2. Analytical metrics vs physical outcomes

- [ ] Validate analytical metrics against simulation or physical outcomes (may combine with item 1).

Pass/fail: same correlation goal as item 1 with explicit mapping from analytical rates to simulated/physical success.

### 3. Learned model comparison (diffusion vs flow vs analytical)

- [ ] Compare diffusion, flow, and analytical baselines on a common MuJoCo plus evaluation pipeline.

Pass/fail: side-by-side metrics on the same objects, grasps, and evaluation script paths.

### 4. Encoder claim (pooled descriptor)

- [ ] Test whether the pooled descriptor improves grasp generation vs ablation.

Pass/fail: controlled experiment documented; see [ADR-0001](docs/adr/001-phase4-canonicalization-vs-equivariant.md) for scope (canonicalization plus invariant features; nontrivial equivariance would need encoder replacement).

### 5. Reachability and robot embodiment

- [ ] Address 2-DOF robot IK failures on arbitrary diffusion outputs (`deploy/robot.xml`; 8/8 IK failures in recorded run).

Options: expand robot DOF or constrain grasp generation to reachable workspace.

Pass/fail: measurable reduction in IK failure rate or documented workspace constraint enforced in generation.

### 6. Diffusion training quality

- [ ] Train longer and/or on more data than artifact-chain defaults (3 epochs, 3 objects).

Pass/fail: checkpoint produces non-degenerate grasp distributions on held-out objects.

### 7. Flow end-to-end as primary gate

- [ ] Run flow through the same runtime verification path as diffusion (structural contract exists in `tests/test_phase4_flow_training.py`; default artifact chain still trains diffusion).

Pass/fail: flow checkpoint runs through inference, evaluation, and simulation workflow with recorded outcomes.

### 8. Baseline comparison

- [ ] Compare analytical vs diffusion vs flow vs RL refinement on shared metrics.

Pass/fail: single table or report with comparable success rates and funnel stages.

### 9. Physical grasp-success funnel

- [ ] Evaluate across multiple objects and grasps using the standard funnel: IK success, collision-free, contact, force closure, lift, stable grasp, success rate.

Pass/fail: funnel metrics reported per object and aggregated; raw counts included.
