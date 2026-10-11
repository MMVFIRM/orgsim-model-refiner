# Gate 2B — safe local Dolibarr discovery (preparation)

The Gate 2A test is **synthetic**. It does not prove that any inferred rule matches Dolibarr. Use Gate 2B as a limited, operator-controlled source of new **development** observations. The runner never performs writes and never reads /login, /setup, /users, document endpoints, or arbitrary URL paths.

## Prerequisites

1. Your **own** disposable Dolibarr 24.0.1 Docker tenant is running at `http://127.0.0.1:8088`.
2. `orgsim_gate` retains read-only third-party, product, order and invoice permissions. No need to enable write rights for this gate.
3. Export/store the sandbox API key in a local process environment variable (`DOLIBARR_API_KEY`). Never place it in a ZIP, YAML file, prompt, shell history, screenshots, or chat transcript.
4. **All fixture IDs must be synthetic**. The runner does not seed records or reset the tenant.

## Dry run (default)

```powershell
py -m orgsim.gate2b_readonly --base-url http://127.0.0.1:8088 --out build/gate2b_dry_run
```

Optional `fixtures.json`, consisting only of identifiers (no credentials), can include:

```json
{"thirdparty_id": 1, "product_id": 1, "product_ref": "ORG-GATE-TEST-PROD"}
```

Then dry run with `--fixtures fixtures.json`. IDs shown above are placeholders, not assumed to be your actual IDs.

## Execute reviewed read-only probes

```powershell
py -m orgsim.gate2b_readonly `
  --base-url http://127.0.0.1:8088 `
  --fixtures fixtures.json `
  --execute-readonly `
  --out build/gate2b_discovery
```

If not supplying fixture IDs, omit `--fixtures` and the runner will issue only **four GET list calls** (`thirdparties`, `products`, `orders`, and `invoices`) with `limit=3`. With all supported fixture selectors, maximum is **nine GET calls**. The code explicitly blocks non-loopback hosts, credentials embedded in URLs, arbitrary endpoint paths and all non-GET operations. HTTP redirects are disabled.

`build/gate2b_discovery/discovery_report.json` contains status codes, structural shapes, field names and types but **no field values or credentials**; inspect it before uploading. It is explicitly marked `phase: discovery` and cannot be counted as an unseen holdout.

## Next blind phase: pending

Do **not** score v0.2.1 on old Gate 1B/1C read probes as if they are new holdouts. The initial read-only Gate 2B work cannot test workflow transitions, inventory reservations, payment side effects or approval policies. A later *fresh, blinded differential workflow gate* will require a separate clone/restore of the local tenant and narrowly scoped, explicitly approved temporary test-only write rights. No such write runner is included here.
