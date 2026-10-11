# Gate 2D: Adaptive, read-only Dolibarr environment discovery

**Status:** runnable offline-tested investigator; live Dolibarr Gate 2D outcomes **pending**. The earlier Gate 2C report established that read-only GET operations work, but it did not establish behavioral reconstruction.

## What changed from Gate 2C

Gate 2C planned 15 mostly fixed GETs. Gate 2D reads four resource baselines and **then** selects further GETs from a bounded allowlist based on observations. It uses a limited observation budget (default 20, max 32), skips pagination for empty/singleton collections when low-information, extracts only safe numeric IDs and a limited product ref *in memory* for detail/relationship probes, and can exit early. No ID or reference values are saved to files. Optional fixture IDs can be provided in a local JSON file. No endpoint can be invented by the language model.

The deterministic scout predicts response shapes before seeing the server reply. The prediction event is appended and fsynced to a SHA-256 linked event log **before** the GET. The observation is recorded separately, with status, response kind, list length, first-record key names and field *types*; raw values are discarded. A signed-outcomes audit is not possible solely from a local hash chain; the chain provides local integrity evidence, not a trusted external timestamp.

There is an **optional model advisor**, disabled by default. If explicitly enabled, it receives only sanitized metadata (probe IDs, names, path templates with `{id}` placeholders, previously observed response field names/kinds/counts). It may choose from the already-allowed probe IDs and submit a shape prediction. Model suggestions are hypotheses, not policy certification; model output cannot inject an API path, HTTP method, fixture ID, credential, or state mutation. **Enabling the model sends metadata to the external OpenAI API** and can incur API charges. It requires both `OPENAI_API_KEY` and `DOLIBARR_API_KEY`. Do not enable it when organizational metadata may not leave the machine.

## Safety prerequisites

- Use **your owned, disposable, loopback Docker Dolibarr tenant** at `http://127.0.0.1:8088`. Do not point this tool at production or a shared demo.
- Confirm that the test API user **only has read permissions** for the relevant modules. The runner hardcodes GET at the HTTP call boundary and permits a finite list of known read routes.
- Keep API credentials out of the ZIP and reports. Supply `DOLIBARR_API_KEY` through a local environment variable. Do not paste your key into ChatGPT or a command in a shared console transcript.
- If possible, populate **synthetic** orders and invoices at distinct lifecycle states through the *Dolibarr UI*, then give the read-only investigator access to see them. With empty orders/invoices, the tool cannot infer their response structure or downstream relationships. The tool itself does not seed or mutate Dolibarr.
- It deliberately does **not** test order validation, invoice creation, payments, permissions changes, or stock mutations.

## PowerShell commands

From the extracted `enterprise-sim-engine-v0.2.3` folder:

```powershell
py -m pip install -e ".[dev]"
py -m pytest -q

# Offline synthetic comparison; no Dolibarr or model calls
py -m orgsim.gate2d_lab --budget 15 --out build/gate2d_offline_local

# Dry run; no API or model calls
py -m orgsim.gate2d_adaptive `
  --base-url http://127.0.0.1:8088 `
  --budget 20 `
  --out build/gate2d_preview
```

The dry run shows the four baseline probes. Subsequent probes cannot be known before observing those responses.

**Optional fixture file** `fixtures.json` (synthetic IDs only, each actual ID must match your tenant):

```json
{
  "thirdparty_id": "123",
  "product_id": "456",
  "product_ref": "SYNTH-PRODUCT-001",
  "order_id": "789",
  "invoice_id": "790"
}
```

The numbers above are **placeholders**. Omit any missing fixture. The investigator automatically extracts safe IDs from nonempty list responses by default, so the fixture file is usually unnecessary. Add `--no-auto-ids` if you wish to disable this feature.

Set `DOLIBARR_API_KEY` in the current local PowerShell session using your usual secure method (avoid leaving it in history), then run:

```powershell
py -m orgsim.gate2d_adaptive `
  --base-url http://127.0.0.1:8088 `
  --budget 20 `
  --execute-readonly `
  --out build/gate2d_local_first

py -m orgsim.gate2d_verify build/gate2d_local_first
Remove-Item Env:DOLIBARR_API_KEY -ErrorAction SilentlyContinue
```

Use a **fresh output directory on each run**; the investigator refuses to overwrite the append-only evidence log. The run will stop before budget exhaustion if no informative allowed probes remain. No writes, account/password endpoints, remote hosts, redirects, or arbitrary tool-selected URLs are allowed.

**Optional true LLM-in-the-loop** (additional opt-in and potential API charges; run only if metadata-sharing is permitted):

```powershell
# Both DOLIBARR_API_KEY and OPENAI_API_KEY must already be set securely in this shell.
py -m orgsim.gate2d_adaptive `
  --base-url http://127.0.0.1:8088 `
  --budget 16 `
  --execute-readonly `
  --execute-model `
  --model gpt-6 `
  --model-calls 8 `
  --out build/gate2d_llm_first

py -m orgsim.gate2d_verify build/gate2d_llm_first
Remove-Item Env:DOLIBARR_API_KEY, Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
```

The model API mode has **not** been exercised against a real provider in this release. Its narrow interface has been tested with mock advisors; provider compatibility and actual model exploration quality are not yet verified. A provider failure stops the run and leaves its local event log for debugging, rather than pretending the run passed.

## Return these three files

Upload these, after reviewing field names for sensitive organizational metadata:

1. `build/gate2d_local_first/discovery_report.json`
2. `build/gate2d_local_first/candidate_read_model_unreviewed.json`
3. `build/gate2d_local_first/FREEZE.json`

Keep `events.jsonl` locally unless we need to audit a suspected ordering or integrity problem. It contains metadata only, but can be lengthy. You can also share `gate2d_verify`'s PASS output.

`candidate_read_model_unreviewed.json` is a **discovery profile**, not an executable OrgSim v0.1 policy manifest or a verified behavioral twin. A fresh, independently withheld evaluation must follow a frozen candidate. This gate does not establish accepted/rejected workflow transitions or business-rule fidelity.
