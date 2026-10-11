# Windows / PowerShell quickstart

Use Python 3.10 or later. Extract the package to a new, writable project directory. Do not overwrite your interrupted v0.6.1 workspace or its private receipts.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe tools/verify_release.py
.\.venv\Scripts\python.exe tools/run_tests.py
```

`tools/run_tests.py` creates a unique writable directory inside `build/` for pytest and its temporary-file environment. It does not rely on the inaccessible Windows default temp directory encountered in earlier runs. It also fixes thread counts for consistent local resource use. The code has been tested on the build environment, not on your Windows machine; these are cross-platform precautions, not an assertion of Windows execution here.

## One integrated offline run

```powershell
.\.venv\Scripts\python.exe -m orgsim_refiner plan
.\.venv\Scripts\python.exe -m orgsim_refiner study `
  --seed 910273 --tenants 24 `
  --train 32 --validation 128 --test 256 --stress 256 `
  --query-budget 8192 --audit-samples 128 `
  --out build/active_study

.\.venv\Scripts\python.exe -m orgsim_refiner verify --root build/active_study
```

The run needs no API key and makes no Dolibarr requests. It includes learning, promotion, matched-data baselines, heldout scoring, stress and failure controls. Query budgets apply to **synthetic observations**, not dollars or ERP operations. Runtime depends on CPU, library versions and learned partition size; no live-run duration guarantee is made.

The supplied completed run can be checked without rerunning training:

```powershell
.\.venv\Scripts\python.exe -m orgsim_refiner verify --root results/active_confirmatory
```

That run is now disclosed, not a future blind test. `REPORT.json` includes all tenants, not only promoted ones. `FINAL_FREEZE.json` binds result files. Keep per-tenant sealed data and evidence journals with your local audit materials.

## Optional model comparison

See [the provider guide](MODEL_REFINER_INTEGRATION.md). Decide whether to include the provider before starting the study. Do not add a model arm after viewing that study's holdout outcomes and call it the same blind comparison.

## Evidence to return

For the synthetic study, `REPORT.json` and `FINAL_FREEZE.json` suffice for reviewing aggregate performance. Keep `SEALED_EVALUATOR.json` and provider/evidence journals locally unless you explicitly choose to disclose them. The included synthetic reports contain no live tenant records.
