# Gate 2F — operator guide (Dolibarr 24.0.1, STRICT GET-only)

**Status:** Offline code and fake-ERP paths verified. Real Dolibarr Gate 2F-A **not yet run**. Gate 2F-B is *synthetic only* and its hidden-interaction coverage gate **fails**. This package does **not** authorize real ERP mutations.

## What is in this release

- **2F-A:** Freeze and compare two local prediction arms (`contract` = documented Swagger 200 response type; `symbolic` = previous Gate 2D observations) plus optional API-backed `model` arm. All arms predict the **same** newly selected GET queries before any query executes. The package's prepared `gate2f_seed_frozen/` has the two local arms only.
- **2F-B:** Hidden synthetic rule/effect oracle. Compare uniform and counterexample-guided discovery with the same induction algorithm, 20 or 32 observations per operation and common held-out cases. Run entirely on the operator computer, not against Dolibarr.
- **Safety:** The live runner can only use `GET` on a loopback host. It does not possess a route for a live write. The model adapter cannot choose or execute an API route. Selectors and discovered IDs stay in memory; output reports contain schema metadata and booleans, not customer values.

## Initial checks (Windows PowerShell)

Open PowerShell in the extracted folder. Run:

```powershell
py -m pip install -e ".[dev]"
py -m pytest -q

# Run Gate 2F-B OFFLINE. The hidden-interaction FAIL is currently EXPECTED.
py -m orgsim.gate2f_causal `
  --seeds 12 --budgets 20 32 --holdout 200 `
  --out build/gate2f_causal_local

# Gate 2F-A preview: ZERO Dolibarr requests
py -m orgsim.gate2f_readonly evaluate `
  --frozen gate_dolibarr/gate2f_seed_frozen `
  --out build/gate2f_preview_local
```

The supplied frozen candidate was built from the existing Gate 2D discovery report and the **sanitized** 249-operation Swagger file. Their hashes appear in `gate_dolibarr/gate2f_seed_frozen/PREPARE_FREEZE.json`. This freezes two preexisting predictor implementations **without exposing any Gate 2F live responses**.

For a new or materially changed Dolibarr tenant, use `prepare` with its *own permitted development discovery*; never use evaluation results to modify an already frozen candidate:

```powershell
py -m orgsim.gate2f_readonly prepare `
  --discovery gate_dolibarr/gate2f_inputs/gate2d_discovery_report.json `
  --swagger gate_dolibarr/gate2f_inputs/sanitized_live_swagger.json `
  --out build/gate2f_new_frozen
```

**Only use a new freeze when no Gate 2F held-out response has yet been observed with it.** The supplied first freeze avoids the accidental reuse of the just-revealed Gate 2E cases: its 18 scored templates are different, and its read-only selector queries are unscored.

## Execute on your local Dolibarr sandbox

Use your existing **read-only, non-admin** `orgsim_gate` user. Store the key in `DOLIBARR_API_KEY` in your local process environment; never put it in a script, command argument, Git repository or a shared report.

```powershell
py -m orgsim.gate2f_readonly evaluate `
  --frozen gate_dolibarr/gate2f_seed_frozen `
  --base-url http://127.0.0.1:8088 `
  --budget 32 --execute-readonly `
  --out build/gate2f_live

py -m orgsim.gate2f_readonly verify `
  --frozen gate_dolibarr/gate2f_seed_frozen `
  --evaluation build/gate2f_live

Remove-Item Env:DOLIBARR_API_KEY -ErrorAction SilentlyContinue
```

The evaluation makes **four unscored selector GETs**, plus up to **18 scored GETs**, normally **22 total** if fixture data is present. It may skip requests for unavailable identifiers or unsafe-looking references. **A short or aborted run is not a pass.** The hard upper bound is 40 GETs; default budget 32. The runner cannot follow redirects.

### Optional genuine GPT-6 arm (before live reads)

The included candidate has **zero model-provider calls**. To test an LLM, create a **new frozen candidate before evaluating any of its held-out cases**, with your `OPENAI_API_KEY` set locally. The preparation step makes a model API request containing **sanitized schema metadata and query templates only**; it does not contact Dolibarr and does not transmit ERP keys, record IDs, or record values.

```powershell
py -m orgsim.gate2f_readonly prepare `
  --discovery gate_dolibarr/gate2f_inputs/gate2d_discovery_report.json `
  --swagger gate_dolibarr/gate2f_inputs/sanitized_live_swagger.json `
  --execute-model --model gpt-6 `
  --out build/gate2f_model_frozen

# Then use --frozen build/gate2f_model_frozen when evaluating.
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
```

The real provider transport was not exercised in this build. The local adapter was tested using a stub. Model API access/cost/formatting may vary. **Do not compare a model arm newly prepared after you have inspected Gate 2F results with the already-used holdout.** That would be a development comparison rather than blind evidence; use a fresh test plan/tenant for a genuine next holdout.

## Outputs to send back

For the live symbolic comparison:

- `build/gate2f_live/holdout_report.json`
- `build/gate2f_live/EVAL_FREEZE.json`
- `gate_dolibarr/gate2f_seed_frozen/PREPARE_FREEZE.json` (or the newly prepared one)

The optional `events.jsonl` permits a local audit of prediction-before-request order, but the verifier is **not an independent third-party timestamp**. The freeze is a local tamper-evidence mechanism, not cryptographic proof of when the user executed requests. No active credential is saved. Review metadata before sharing.

**Do not send `DOLIBARR_API_KEY`, `OPENAI_API_KEY`, the full customer records, or production data.**

## Expected interpretation

A 403 for an endpoint means **this role on this tenant** lacks access, not that the operation violates a business rule. `contract` deliberately does not exploit discovery evidence; `symbolic` and `model` have access to it. In all cases, success is scored on *attempted* predictions, with separate coverage and accuracy. Perfect HTTP status alone is not behavioral reconstruction. The exercise has no live writes, no measured real causal transitions, and no independent-tenant evaluation.

The causal benchmark's known hidden-interaction FAIL (20% coverage, zero false accepts) means **no live transactional or financial mutation gate** is authorized by this release.
