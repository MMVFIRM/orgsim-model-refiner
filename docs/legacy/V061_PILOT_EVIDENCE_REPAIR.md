# v0.6.1 — Resume the already-executed Dolibarr factorial pilot

**No need to repeat the successful/failed real-order validations.** This is a targeted evidence parser fix, not a new or modified factorial study. Pre-registration, labels, split allocation and superiority thresholds are unchanged.

## Exact reason for the fix

The 10 (on-sale/stock-insufficient) pilot returned Dolibarr HTTP 500 with an application message stating **insufficient stock and negative stock disallowed**. Its local diagnostic showed an attempted stock movement and a transaction rollback, but did not duplicate the application-level error text. v0.6 incorrectly required the *diagnostic alone* to match the stock reason. The corrected validator checks two separate evidence paths:

1. An HTTP 500 with the Dolibarr `Error when validating Order:` prefix and a reason-specific local diagnostic; OR
2. The same HTTP validation prefix with an explicitly matching **business reason in the response** plus a separately recorded **stock-movement attempt and transaction rollback** for an insufficient-stock refusal. The sale-refusal variant instead requires an order-validation attempt and transaction rollback, if the reason appears only in the HTTP body.

No file is considered a cryptographic server signature. The operator still attests that the two files represent the same request, and the run still requires GET-confirmed unchanged state. The response and diagnostic are both hashed and their evidence locations are reported. Generic HTTP 500s, generic rollback without a specific response reason, and unrelated logs do not qualify.

## Fix an existing v0.6 project on Windows

Copy the patched `src/simlearn/dolibarr_factorial.py` over that same relative file in your **existing v0.6 project**. If it was installed with `pip install -e`, the new source is used immediately. Do not copy your private pilot data into a freshly extracted project or reset the already-intervened records.

In PowerShell, from the root of the existing project, with the same read-only watcher key configured, run:

```powershell
py -m pytest -q tests/test_dolibarr_factorial.py

py -m simlearn.dolibarr_factorial check-evidence `
  --receipt private_receipts/pilot-10-response.txt `
  --diagnostic private_receipts/pilot-10-diagnostic.txt `
  --reason insufficient_stock

py -m simlearn.dolibarr_factorial check-evidence `
  --receipt private_receipts/pilot-01-response.txt `
  --diagnostic private_receipts/pilot-01-diagnostic.txt `
  --reason product_not_on_sale
```

The private file names above are illustrative: use the actual paths on your Windows machine. The checker reads local evidence only; it makes **zero** Dolibarr or model-provider calls and does not change the pilot.

When both classifications succeed, rerun exactly your previously failed `pilot-finish` command with `--root` pointing to the **original pilot directory**. Use your actual recorded 200/500/500 statuses and existing receipt/diagnostic files. `pilot-finish` re-reads the three observed states with GET requests and creates `public/PILOT_CONFIRM.json` only if the successful order changed while both refused orders remained unchanged.

Do not repeat any writer POSTs. Do not touch the withheld 00 combination. Do not fit/score based on pilot evidence alone.

If either local log does not contain even an attempted operation/rollback, do **not** manufacture such a log. Provide a redacted excerpt for further technical diagnosis, keep that case unqualified and retain the experiment's original scientific stopping rule.

## Next step

After `PILOT_CONFIRM.json` exists, run the **unchanged** original v0.6 study scaffold command. You still need fresh, separate study orders/products and a restorable database + document-store snapshot before operator-run interventions. Keep API credentials and raw receipts local. The evidence is operator-attested; no real-ERP action authorization follows from this research workflow.
