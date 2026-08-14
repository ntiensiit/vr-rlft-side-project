# ADR-0007 — W&B artifact versioning for the artifact chain

## Status

Accepted (2026-08-14)

## Context

The project maintains a reproducible **artifact chain** driven by
`scripts/run_artifacts.py`. Intermediate and final outputs are stored as local
files:

* Dataset records and indices: `data/processed/*.npy`, `data/processed/index.json`
* Model weights: `artifacts/checkpoints/*.pt`
* Generated grasps and reports: `artifacts/exports/*.npy`, `artifacts/reports/*.jsonl`

These paths are gitignored and regenerated on demand. Lineage is recorded in
`artifacts/manifest.jsonl`, which lists executed commands and retained relative
paths but does not version blobs, attach content hashes, or link downstream
steps to upstream dataset/checkpoint versions.

Phase 8 added **TensorBoard** for scalar metrics (loss, evaluation aggregates).
TensorBoard is well suited to time-series metrics but is **not** an artifact
registry: it does not version `.pt` checkpoints, `.npy` dataset shards, or
`index.json`, and it offers no cross-run lineage for the artifact chain.

`wandb` is already declared in `pyproject.toml` but was unused. **MLflow** was
considered; it provides a model registry and artifact store but would add a new
dependency and duplicate much of the W&B surface area already available in the
project.

## Decision

1. **Keep TensorBoard** for local scalar metrics during training and evaluation
   (no change to Phase 8 behavior when tracking is disabled).

2. **Add optional Weights & Biases artifact publishing** for the artifact chain
   directly in `scripts/run_artifacts.py`, enabled when
   `tracking.backend: wandb` in config (default `none`).

3. **Publish one W&B run per artifact-chain execution** with:
   * A versioned `artifact-chain` artifact containing all retained outputs
     (checkpoints, `.npy` exports, dataset index, JSONL reports).
   * Run metadata: config directory, manifest path, and file count.
   * Manifest augmentation: when publishing succeeds, append a
     `wandb_tracking` record with `run_id` and artifact version.

4. **Default to offline W&B mode** (`tracking.mode: offline`) so CI and local
   runs work without API credentials; operators opt into online sync via config
   or `WANDB_MODE=online`.

5. **Do not replace** `manifest.jsonl` or local file paths. W&B is an optional
   overlay for versioning and lineage, not the primary storage contract.

## Rationale

| Concern | TensorBoard alone | W&B artifacts (this ADR) |
| --- | --- | --- |
| Checkpoint / `.npy` versioning | No | Yes (content-addressed artifacts) |
| Dataset index lineage | Manual via manifest paths | Linked to run + artifact version |
| CI / offline use | Local event files | Offline W&B dir, no API key |
| Extra dependency | Already present | Already in `pyproject.toml` |
| MLflow alternative | N/A | Deferred; would add dep without removing W&B gap |

TensorBoard and W&B are complementary: TensorBoard for metrics dashboards,
W&B for blob versioning and experiment comparison across artifact-chain runs.

## Consequences

* Artifact-chain tests and CI remain unchanged when `tracking.backend` is `none`
  (default).
* Enabling W&B writes to `wandb/` (gitignored). Operators sync with
  `wandb sync` when using offline mode.
* Future per-stage runs (individual `train_diffusion.py` invocations) may
  inline the same W&B artifact logging pattern without changing this ADR.
* MLflow integration remains out of scope unless W&B proves insufficient for
  model-registry workflows.

## Follow-up review triggers

Revisit when:

* the artifact chain grows large enough that monolithic W&B artifacts are
  unwieldy (split per-stage artifacts);
* a hosted model registry with promotion stages is required (re-evaluate MLflow);
* flow training becomes the default chain path (ADR-0002).
