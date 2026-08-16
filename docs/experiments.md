# Experiments

End-to-end reproducible artifact generation is driven by:

```bash
uv run python scripts/run_artifacts.py
```

The script writes `artifacts/manifest.jsonl` documenting the executed commands
and the retained artifacts. The artifacts themselves (checkpoints, processed
dataset, reports, MJCF wrappers, observations) are excluded from version
control by `.gitignore` and are produced from source on demand by this script.

## Artifact versioning (optional)

TensorBoard (Phase 8) records training and evaluation **metrics** locally.
For **model weights**, dataset `.npz` shards, and `index.json` lineage, enable
W&B artifact tracking in `configs/base.yaml`:

```yaml
tracking:
  backend: wandb
  project: vr-rlft-side-project
  mode: offline   # or online when WANDB_API_KEY is set
```

Re-run `scripts/run_artifacts.py` with `backend: wandb`. The manifest gains a
`wandb_tracking` record with `run_id` and `artifact_version`. Offline runs
store data under `wandb/`; sync with `wandb sync` when ready.

See [ADR-0007](./adr/007-wandb-artifact-versioning.md) for rationale (W&B vs
MLflow, complementary use with TensorBoard).

## Grasp frames (Panda)

Synthetic and generated grasps store **contact-frame** poses (origin at the
antipodal midpoint). ``simulate_grasp`` converts to the Panda **hand frame**
before IK (ADR-0009). Constants live in ``configs/gripper/franka_emika_panda.yaml``.

For per-phase design and verification status, see
`docs/PROJECT.md` and `CHECKLIST.md`.