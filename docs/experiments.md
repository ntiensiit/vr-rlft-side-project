# Experiments

End-to-end reproducible artifact generation is driven by:

```bash
python scripts/run_artifacts.py
```

The script writes `artifacts/manifest.json` documenting the executed commands
and the retained artifacts. The artifacts themselves (checkpoints, processed
dataset, reports, MJCF wrappers, observations) are excluded from version
control by `.gitignore` and are produced from source on demand by this script.

For per-phase design and verification status, see
`docs/PROJECT.md` and `hot-fix-checklist.md`.