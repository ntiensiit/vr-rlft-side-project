---
trigger: always_on
---

## Command Execution Rules

1. **Virtual Environment:** Always use the active virtual environment (`.venv`).
2. **Package Management:** Use `uv` for package management and running tools if applicable.
3. **Prevent Pycache Clutter:** When running any Python command (e.g., training, evaluation, tests), always set `PYTHONPYCACHEPREFIX` to `./.pycache` so that compiled cache files are stored at the repository root and do not clutter packages.
   - For Windows PowerShell (pwsh), prefix/execute like: `$env:PYTHONPYCACHEPREFIX='./.pycache'; python scripts/train.py` or similar command.