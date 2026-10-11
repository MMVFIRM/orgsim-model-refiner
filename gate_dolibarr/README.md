# Gate 1 — Dolibarr reconstruction benchmark

**What this package actually does:** Imports the *live tenant's* Swagger 2.0 REST contract; compiles an explicitly unreviewed OrgSim resource simulator; reports API operations that were/weren't represented; and compares selected real/simulated requests without logging business records. This is a baseline **contract-driven compiler**, not yet an LLM autonomously inferring rules or learning to perform workflows.

**What it cannot do yet:** Prove behavioral fidelity from an OpenAPI file; reconstruct proprietary business rules, validation/approval transitions, permissions, workflow operations, or relationships just by parsing Swagger. The v0.1.3 numeric create-ID projection addresses one previously observed mismatch, but actual read/list response fields and business semantics remain unverified.

## A. Start your own disposable Dolibarr tenant

Requires Docker Desktop/Compose on your machine. This analysis environment does *not* have Docker, so we have not spun up a real Dolibarr server or claimed to test it.

1. `cd gate_dolibarr/compose`
2. Copy `.env.example` to `.env`, change all password placeholders to long unique values.
3. `docker compose --env-file .env up -d`
4. Open `http://127.0.0.1:8088`, finish initial setup if necessary and enable **API REST**, **Third parties**, **Products**, **Orders**, **Invoices** in Setup → Modules. Module auto-enablement is not always reliable; verify in UI.
5. Create a dedicated *sandbox API user*, enable only the permissions needed for the tests, and generate its **Key for API**.
6. On your own machine, set `DOLIBARR_API_KEY` in the shell (do **not** put it in arguments, chat, files or screenshots). The client sends `DOLAPIKEY` as an HTTP header and never saves the token.

The official Docker repo/docs: https://github.com/Dolibarr/dolibarr-docker . The documented Swagger location is `/api/index.php/explorer/swagger.json` and authorized actions use `DOLAPIKEY` header.

**Windows PowerShell example:**

```powershell
$env:DOLIBARR_API_KEY = Read-Host "Disposable Dolibarr API key"
py -m orgsim.dolibarr_gate capture --base-url http://127.0.0.1:8088 --out build/dolibarr/live_swagger.json
py -m orgsim.dolibarr_gate reconstruct --swagger build/dolibarr/live_swagger.json --out-dir build/dolibarr/reconstructed
```

**Bash example:**

```bash
read -rs DOLIBARR_API_KEY; export DOLIBARR_API_KEY
python -m orgsim.dolibarr_gate capture --base-url http://127.0.0.1:8088 --out build/dolibarr/live_swagger.json
python -m orgsim.dolibarr_gate reconstruct --swagger build/dolibarr/live_swagger.json --out-dir build/dolibarr/reconstructed
```

First install OrgSim from the repo root: `python -m pip install -e '.[dev]'`. For Windows use `py -m pip install -e ".[dev]"`.

If your Dolibarr version uses a different Swagger format or custom endpoints, the compiler will report what it couldn't interpret. Do not set `reviewed: true` simply to bypass validation.

## B. Run independently scored probes

The suite is in `gate_dolibarr/fixtures/example_probes.yaml`. This is a template, not a guarantee that the exact payloads will work on every Dolibarr version. First run **discovery** (a few harmless GETs). Keep **holdout** cases out of model-facing context.

```bash
python -m orgsim.dolibarr_gate probe \
  --base-url http://127.0.0.1:8088 \
  --manifest build/dolibarr/reconstructed/draft_manifest.yaml \
  --suite gate_dolibarr/fixtures/example_probes.yaml \
  --phase discovery --out build/dolibarr/discovery_results.json

python -m orgsim.dolibarr_gate probe \
  --base-url http://127.0.0.1:8088 \
  --manifest build/dolibarr/reconstructed/draft_manifest.yaml \
  --suite gate_dolibarr/fixtures/example_probes.yaml \
  --phase holdout --out build/dolibarr/holdout_results.json
```

By default, write cases are recorded as **write_blocked**. To run write probes, the operator must explicitly request `--allow-writes --confirm I-OWN-THIS-DISPOSABLE-DOLIBARR-SANDBOX`, and the target must be loopback. WARNING: those probes modify Dolibarr's database. Always use a resettable test-only instance. **Never** run them on a production ERP or shared public demo.

The report stores only HTTP status, response types, array lengths and object field **names**, not real record values. Swagger and any customized probe suites should still be handled as internal organization artifacts. Review for sensitive information before sharing.

## C. Proper experimental evaluation

| Stage | Model/engine gets | Scorer gets | Measured |
|---|---|---|---|
| Discovery | Swagger + discovery probe outcomes | Same | Contract coverage and training failures |
| Blind reconstruction | Only allowed discovery artifacts | Hidden workflows/cases | Quality of generated rule/behavior spec |
| Holdout | Frozen OrgSim manifest | Fresh disposable Dolibarr + tests | Status, shape, state transitions and workflow parity |

**Strict rule:** Dolibarr source code or hard-coded business policies must not enter the blind reconstruction context. The code and documentation can be used by an independent evaluator to explain residual errors.

An adequate later gate should include at least **100 held-out operations across valid, invalid and stateful sequences**, with fresh resettable states and an explicit field/state-level oracle. A simple GET returning an empty list doesn't demonstrate policy fidelity. Do not report a 'pass' for the engine based only on contract compilation, or a few matching 404 responses.

## D. Offline developer smoke test

No Docker or API key is required:

```bash
python -m orgsim.dolibarr_gate reconstruct \
  --swagger gate_dolibarr/fixtures/illustrative_swagger2.json \
  --out-dir build/illustrative_fixture
pytest -q
```

The fixture is **hand-authored for testing import logic, not downloaded from Dolibarr**. A successful fixture smoke test verifies the translation/compiler, not Dolibarr behavior.

## Security and operational notes

- Do not run on production data, user PII, or a public shared demo. No default write access.
- Bind Docker tenant to loopback; HTTPS required for non-loopback capture/read-only requests.
- Store API keys in environment only, not in URLs, logs or JSON files.
- No redirects across domains; the data never leaves the operator's computer by this code except intentional GET/POST requests against the configured tenant.
- Captured Swagger may expose internal API paths and hostname; keep it private.
- Reset test-tenant volumes between independent benchmark runs; do not equate two independently populated states without seed alignment.

## E. Feed only permitted evidence to the reconstruction model

After running discovery probes, build a **blind model bundle**:

```bash
python -m orgsim.dolibarr_gate bundle \
  --manifest build/dolibarr/reconstructed/draft_manifest.yaml \
  --coverage build/dolibarr/reconstructed/coverage.json \
  --discovery build/dolibarr/discovery_results.json \
  --out build/dolibarr/model_input.json
```

The model receives the imported manifest, contract surface summary and **discovery-phase** results only. It must output a revised candidate manifest (not executable Python). The evaluator freezes it, runs held-out cases against the real Dolibarr sandbox, and reports comparison failures. This keeps the deterministic compiler baseline separate from any genuine model-inferred behavioral improvements. There is no model API call in this package; the operator selects the model and provides the bundle.

## F. v0.1.3 — preserve semantic checks and expose discovery gaps

**Install from this v0.1.3 folder and reconstruct the live Swagger anew.** Do not overwrite original v0.1.1/v0.1.2 reports, and keep previous revealed probe outcomes in the development set.

The reported v0.1.2 live regression matched HTTP 200 on create and readback, but differed on POST return kind (number/string) and had only ~1.8% full readback field recall. On the reported populated list probes, field recall was about 1.9% (third parties) and 5.7% (products). The following offline changes are motivated by those observations; they have **not** been verified against your localhost tenant.

**1. Improve ID representation without hard-coding Dolibarr IDs.** The runtime now emits decimal-string identity keys for a resource where stored ID is typed `string` and POST returns a `number`. The declared response projector turns `"13"` into JSON `13`, and subsequent GET calls still locate the same logical record. This is not a proof of exact ID allocation or permission semantics.

**2. Score actual business fields, not just field presence.** Add `compare_fields` to operator-authored cases and optional `match_items_by` for multi-row lists. Example discovery-only suite: `gate_dolibarr/fixtures/semantic_discovery_template.yaml`. The comparator aligns list records using a unique synthetic field such as `name` or `ref`. Only numeric counts of field comparisons/matches are written to the report, not values. If the rows cannot be aligned, the field comparison is marked unavailable rather than scored as a match. Do not use an unstable field or compare auto-generated IDs across independent databases.

**3. Summarize discovery response gaps (no write permission required):**

```powershell
py -m orgsim.dolibarr_gate shape-gaps --discovery build/dolibarr/v013_discovery_results.json --out build/dolibarr/v013_shape_gaps.json
```

This command refuses reports marked `phase: holdout`. It summarizes names/types of missing fields but never invents missing values or policies. A model may use it as discovery evidence; it must not use hidden outcomes.

**4. Generate a useful model reconstruction bundle with all operation signatures:**

```powershell
py -m orgsim.dolibarr_gate bundle `
  --manifest build/dolibarr/reconstructed/draft_manifest.yaml `
  --coverage build/dolibarr/reconstructed/coverage.json `
  --discovery build/dolibarr/v013_discovery_results.json `
  --swagger build/dolibarr/live_swagger.json `
  --out build/dolibarr/v013_model_input.json
```

The optional `--swagger` catalog contains endpoint paths, operation descriptions, parameter names and declared field types, not values or secret headers. **Review the exported bundle before sharing**; schemas, paths, and human-authored descriptions can themselves be sensitive. The 204 unimplemented operations remain listed, not represented by fictitious working tools.

**5. Keep model reconstruction and scoring separate.** The bundle is an input for a future model run, not proof that a model has inferred Dolibarr behavior. A candidate manifest must be frozen before running *fresh, previously undisclosed* holdout cases. Existing v0.1.1/v0.1.2 probes are regression checks only. For a strong gate, compare successful and rejected operations, genuine transitions, and operator-selected values on **aligned synthetic state**, not just 404s or matching list lengths.

## F. v0.1.2 — controlled comparison and diagnostics

The initial local Dolibarr test reported **249 discovered operations and 45 mapped**, but showed only comparisons with an empty simulator. Those results must not be reported as successful or failed business-rule predictions because state alignment was not established. Creation rejection was a genuine request-acceptance discrepancy and requires investigation.

The v0.1.2 importer records `create_input_schema` (if declared by Swagger) independently of the stored/read resource schema. The runtime checks the create input against that request contract *before* adding its generated ID. The gate can compare an API's Swagger-declared numeric create return value rather than accidentally comparing `123` to a simulator record object. This does **not** synthesize response-only fields or hidden organizational policies.

The gate reports `sim_error.code` and `sim_error.validation` when a local call is rejected; these diagnostics contain only structural metadata, not input values. If it reports `REQUEST_SCHEMA_VIOLATION`, inspect the live Swagger's create-body definition versus the genuinely accepted request. If it reports `SCHEMA_VIOLATION`, the draft *storage* model remains overconstrained. Do not use a hidden holdout response as model training evidence.

To compare an intentionally populated Dolibarr tenant with an intentionally populated simulator, create a **new operator-authored discovery probe suite**, optionally with a top-level `sim_seed_calls` list:

```yaml
# ALL values here are fake and should match records the operator already
# seeded in a disposable Dolibarr tenant. Never use private organizational data.
sim_seed_calls:
  - tool: dolibarr.thirdparties.create
    args: {name: "Fake QA Customer", client: 1}
  - tool: dolibarr.products.create
    args: {ref: "FAKE-QA-PROD", label: "Fake QA Product"}
cases:
  - id: seeded-customers
    phase: discovery
    method: GET
    path: /thirdparties?limit=100
    tool: dolibarr.thirdparties.list
    args: {limit: 100}
```

A seed operation may still fail because the imported request contract or state schema is incomplete. That failure is reported as a structural error, **not** silently patched. Also, matching list lengths after seeding does not establish response-field fidelity or correct business rules. Identities created in the two environments may not share the same numeric IDs.

After installing `v0.1.2` or later, reconstruct again from the previously captured local Swagger so the new compiler fields appear, then run a *discovery* suite. Preserve the original results unchanged for comparison. Treat all previously revealed probe outcomes as development evidence, not blind holdout scores. Reserve a newly generated unseen test set for the model-assisted experiment.
