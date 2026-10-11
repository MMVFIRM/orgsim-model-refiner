# Gate 2C Dolibarr read-only discovery (operator-run)

This gate deliberately **does not test business-rule transitions**. It gathers new development observations to improve the next model-generated reconstruction. Never treat discovery observations as blind holdout results.

**Prerequisites:** an owned disposable Docker Dolibarr 24.0.1 instance at `http://127.0.0.1:8088`; the `orgsim_gate` account must retain its previously confirmed read-only permissions. No write permissions are needed. Never export the API key in a JSON or ZIP file.

## Windows PowerShell commands

From the extracted engine directory:

```powershell
py -m pip install -e ".[dev]"
py -m pytest -q

# Default: dry run, does not contact Dolibarr
py -m orgsim.gate2c_dolibarr_readonly `
  --base-url http://127.0.0.1:8088 `
  --budget 16 `
  --out build/gate2c_dryrun
```

The default plan contains **15 fixed GET requests** (four list baselines, one-variable pagination/limit/property variants). The `--budget 16` is a maximum, not a requirement to issue 16 calls.

Optionally specify only synthetic test fixture IDs in `fixtures.json` (no values/credentials):

```json
{"thirdparty_id":1,"product_id":1,"product_ref":"ORG-GATE-TEST-PROD"}
```

The example IDs are **placeholders**. Replace them with actual synthetic fixture identifiers from your local tenant. Including fixtures may add extra detail/alternate-key GET calls, up to an absolute maximum of 20.

Only after previewing the plan, store the API key in your **local process environment** and execute:

```powershell
py -m orgsim.gate2c_dolibarr_readonly `
  --base-url http://127.0.0.1:8088 `
  --budget 20 `
  --fixtures fixtures.json `
  --execute-readonly `
  --out build/gate2c_discovery

# Clear the key from this PowerShell session afterward
Remove-Item Env:DOLIBARR_API_KEY -ErrorAction SilentlyContinue
```

`DOLIBARR_API_KEY` must already be configured securely before running the executed command; it is not passed as a CLI argument. The program never runs PUT, POST, PATCH, or DELETE, and refuses non-loopback hosts, arbitrary URL routes, auth endpoints, and redirects. It does not store API response values or credentials. Inspect the generated field-key metadata before sharing any reports.

## Files to return

- `build/gate2c_discovery/discovery_report.json`
- `build/gate2c_discovery/FREEZE.json`

You do not need to upload the raw Swagger again unless it changes; the previous sanitized copy is sufficient. The next proper *blind* scoring gate needs a separately frozen candidate and new heldouts with controlled initial states. The read-only report does not establish payment, order-validation or stock side-effect fidelity.
