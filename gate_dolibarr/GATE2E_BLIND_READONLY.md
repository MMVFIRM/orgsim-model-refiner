# Gate 2E — frozen blind read-behavior benchmark (v0.2.4)

**Status: offline harness verified, local Dolibarr testing pending.** This is a *read-only* diagnostic against a disposable local Dolibarr 24.0.1 tenant. The same API key/user/permissions and compatible test state should be used as in Gate 2D. The test does not reconstruct or invoke order validation, invoice creation, payments, stock mutation, or other write workflows.

## The critical separation

1. **Prepare OFFLINE.** Load the prior Gate 2D `discovery_report.json`. The program creates a candidate of same-role access predictions, response types and unverified relationship hypotheses, plus a **held-out query plan**. Both are frozen by hashes. No Dolibarr connection is established and no new observations are obtained.
2. **Evaluate.** The benchmark first performs unscored GETs to select fixture IDs *in memory*, then runs fresh query variants and (when available) reads **second** customers/orders that Gate 2D did not inspect individually. Predictions from the frozen candidate are durably logged before each GET. Both symbolic and optional model arms score the **same observed response**.
3. **Verify.** A separate local audit checks candidate/plan/report SHA-256 digests, prediction-first event ordering, and GET-only evidence. This is local tamper detection, **not external proof of timing or a remotely authenticated audit log**.

A query holdout in the same tenant is not an independent-tenant holdout and it is **not** a test of business-rule or state-transition fidelity. Exact requests against the lone previously observed product/invoice have been removed from headline scores rather than counted as new evidence.

## Quickest path using your supplied Gate 2D result

The included folder `gate_dolibarr/gate2e_seed_frozen/` contains a **symbolic-only** candidate frozen from the earlier Gate 2D discovery results (three synthetic third parties, one product, two orders, one invoice). The input JSON was reserialized and its SHA-256 confirmed to match the `report_sha256` value in your prior Gate 2D `FREEZE.json` before preparing this package. These are still **unreviewed hypotheses**. It is preferable to run a new `prepare` when permissions or records have changed.

From PowerShell in the extracted v0.2.4 project:

```powershell
py -m pip install -e ".[dev]"
py -m pytest -q

# Dry run: zero ERP requests. Uses the previously frozen symbolic candidate.
py -m orgsim.gate2e_blind evaluate `
  --frozen gate_dolibarr/gate2e_seed_frozen `
  --out build/gate2e_preview
```

Then, with an API key for the **same read-only non-admin Dolibarr user** stored only in `DOLIBARR_API_KEY`, perform live reads:

```powershell
py -m orgsim.gate2e_blind evaluate `
  --frozen gate_dolibarr/gate2e_seed_frozen `
  --base-url http://127.0.0.1:8088 `
  --budget 32 `
  --execute-readonly `
  --out build/gate2e_live

py -m orgsim.gate2e_blind verify `
  --frozen gate_dolibarr/gate2e_seed_frozen `
  --evaluation build/gate2e_live

Remove-Item Env:DOLIBARR_API_KEY
```

Only GET operations to a loopback Dolibarr server are permitted. The evaluator never makes a POST, PUT, PATCH, or DELETE request, including in model mode. For read-only discovery, key permissions must include the relevant read rights, not write rights. No automated live test was performed while building the release.

## Preferred path: recreate the freeze on your own original evidence

If you retain your **original** Gate 2D report, re-freeze from that report before evaluating. This is especially important if the tenant's data or user privileges have changed:

```powershell
py -m orgsim.gate2e_blind prepare `
  --discovery C:\path\to\gate2d_live\discovery_report.json `
  --out build/gate2e_frozen

py -m orgsim.gate2e_blind evaluate `
  --frozen build/gate2e_frozen `
  --budget 32 `
  --execute-readonly `
  --base-url http://127.0.0.1:8088 `
  --out build/gate2e_live

py -m orgsim.gate2e_blind verify `
  --frozen build/gate2e_frozen `
  --evaluation build/gate2e_live `
  --discovery C:\path\to\gate2d_live\discovery_report.json
```

**Never edit a frozen prediction after the live comparison.** If you refine your hypotheses, choose a new output directory and reserve fresh read cases as a new evaluation. If testing on a tenant with fewer than two orders or customers, those distinct-identity cases are skipped rather than silently replaced by seen records; this will reduce sample size and coverage.

## Optional genuine model-versus-symbolic comparison

To compare a real model with the symbolic baseline, the real model must predict **before the first new Dolibarr GET**. This step uses an independent, explicitly supplied `OPENAI_API_KEY` locally; it sends only whitelisted response metadata and route templates, not ERP record values, fixture identifiers, ERP API keys, or the local address.

```powershell
# Requires OPENAI_API_KEY; may incur model usage charges.
py -m orgsim.gate2e_blind prepare `
  --discovery C:\path\to\gate2d_live\discovery_report.json `
  --execute-model `
  --model gpt-6 `
  --out build/gate2e_model_frozen

# THEN use --frozen build/gate2e_model_frozen for evaluate/verify.
```

The actual model service is **not accessible from this build environment**. The provider path was exercised only with a stubbed transport. If your model entitlement or chosen model's API capabilities differ, provider setup or response formatting may require adjustment. A model call is never triggered by the default commands.

## Output and interpretation

Files from `prepare`:

- `candidate_frozen.json`: symbolic predictions and optional real-model predictions, hypothesis-level relationships and role-scoped access outcomes; `reviewed:false`.
- `holdout_plan.json`: query templates and evidence-novelty labels; never stores IDs.
- `PREPARE_FREEZE.json`: exact candidate, plan, and source hashes.

Files from `evaluate`:

- `holdout_report.json`: per-query HTTP status, response kind, list count, selected business field **types**, relationship validity booleans, plus separate metrics for permission checks, relationships and response types. Field values are discarded rather than written.
- `events.jsonl`: hash-chain event log with **prediction precommit before each GET** and metadata-only responses.
- `EVAL_FREEZE.json`: report digest and log tip hash.

The 13 planned cases from the supplied populated discovery contain: four query projections, four distinct-record reads (including an order-line read), three role/permission queries, and three relational checks (some groups overlap, depending on the classification). The real evaluator also makes up to five unscored selector GETs. The actual case count depends on the data and first discovery evidence. It is valid for a live run to have fewer scored cases than configured.

Do **not** interpret HTTP/array-shape-only matches as full behavioral reconstruction. Use `field_coverage`, `selected_type_accuracy`, `permission_status_accuracy`, `relationship_accuracy`, `nontrivial_full_matches`, and counts separately. In particular, a 403 indicates this user lacks access under this installation and configuration; it does not prove the underlying business operation is invalid. A model may abstain on predictions it cannot justify; abstentions reduce coverage.

**What to send back:** `build/gate2e_live/holdout_report.json`, `build/gate2e_live/EVAL_FREEZE.json`, and `build/gate2e_frozen/PREPARE_FREEZE.json` (or the included frozen candidate's `PREPARE_FREEZE.json`). Include `events.jsonl` only if you want an independent hash-chain audit. Review field names and types before sharing. **Do not share either API key.**
