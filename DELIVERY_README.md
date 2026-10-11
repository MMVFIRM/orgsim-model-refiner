# v0.9.0 delivery and recovery

This is a newly reconstructed archive from the verified v0.8 package. The
previous v0.9 download link referenced a nonexistent artifact; this one includes
actual source files and was checked before being linked.

## Integrity

Run `python tools/verify_delivery.py`; it recomputes SHA-256 for each
manifest-listed file. Run `python tools/test_delivery.py` from an installed
project environment. See `release_validation/BUILD_STATUS.json` for actual
check results, including failures or timeouts.

## Installation

```
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe tools/verify_delivery.py
.\.venv\Scripts\python.exe tools/test_delivery.py
```

**Production: read-only observation only.** See docs/PRODUCTION_DEPLOYMENT.md.
**Active learning: disposable clone only.** See docs/CLONE_LEARNING.md.

This is not independently security-audited and does not authorize production
transaction processing. The inherited experimental results are historical.
