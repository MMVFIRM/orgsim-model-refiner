# OrgSim × Model Refiner v0.4 — Paired Dolibarr causal transfer

**Operator manual.** This experiment targets a **disposable local Dolibarr Docker tenant** only. The Python runner and its API client issue **GET requests only**. It never executes an ERP mutation. Any order validation, invoice validation, payment, or other action is performed *separately by the authorized operator* using Dolibarr's UI or an explicitly authorized, independent API client after making a restorable snapshot. Do not run against production data. No actual Dolibarr transaction has been conducted by this release.

## Why this replaces v0.3

v0.3's labeled holdout had **17/17 accepts**. Always-accept reached 100%; a tree couldn't learn a rejection boundary. Six invoices returned 500 and one request returned 403; these were appropriately unlabelable. In v0.4, train/validation must contain matched pairs with **one verified scoped state change and one receipt-supported no-change outcome** *before* the model can be frozen or any holdout is opened. Evaluation is reported against always-accept and always-reject controls (50% on fully qualified opposite-outcome pairs).

**Two types of negative evidence are kept separate**:

- **HTTP 304 Not Modified** plus no GET-observable change is an eligible *no-transition* label. Swagger for Dolibarr 24's `POST /orders/{id}/validate` lists 304 as a possible response. It is **not proof of business-rule rejection**.
- **Documented business refusal**, observed using a separate authorized writer API identity with HTTP **400, 409, or 422**, a real machine-readable error code and a locally retained redacted response file; plus no GET-observable state change. The operator classifies the failure. This still isn't independent server attestation.

**HTTP 401/403**, **404**, **5xx**, HTTP 200 without an observed transition, and successful requests without corroborating GET differences **do not become reject labels**. In particular, 500 cannot be interpreted as a business-rule denial without additional independent investigation. The local tool can triage these categories before you seed 38 records.

## Recommended scope and effort

Start with **`order_validate` only** to avoid repeating the v0.3 invoice-validation 500 errors. This is a *single integrated experiment*, not several gates. Defaults are **six training pairs, three validation pairs, ten holdout pairs**: 19 pairs, **38 independently disposable orders**, 38 manual action attempts (one per order), and roughly 152 watcher GET requests at four GETs per paired start plus two per finish, excluding retries. Exact request count may be higher if related records are watched. Separate writer and read-only identities are mandatory; never use the watcher key for a write. Optionally include `--operation invoice_validate` only once you have reliable independent receipts for *both* outcomes.

The minimum to freeze is **four qualifying training pairs** and **two qualifying validation pairs** per operation. Remaining pairs may be inconclusive, but every train/validation case must be finished and classified. At least **eight qualifying held-out pairs** per operation are required to claim a scoped transition-classification pass, with 95% accuracy, 95% decision coverage, zero observed false accepts and superiority over always-accept. To claim *actual business-refusal discrimination*, at least eight **documented business refusals** are required among the held-out qualifying pairs—not merely 304 responses. These targets are not evidence of real-world safety.

## 0. Preflight and environment

Extract ZIP, open PowerShell at package root, create a restorable Docker/database snapshot, and verify the user has **legitimate permission to create disposable orders** and to perform the intended writes from a *separate writer account*. Configure the read-only watcher environment variable without including a credential in any log or uploaded artifact:

```powershell
py -m pip install -e ".[test]"
py -m pytest -q tests/test_dolibarr_counterfactual.py
$env:DOLIBARR_API_KEY = "<private local read-only token>"

# No network calls: classify a sample operator-observed HTTP result.
py -m simlearn.dolibarr_counterfactual triage --receipt-status 304 --writer-authorized
py -m simlearn.dolibarr_counterfactual triage --receipt-status 500 --writer-authorized
```

Before investing in fixtures, use a **separate disposable pilot order outside the benchmark** to establish what your Dolibarr instance actually does when an authorized writer attempts validation on a suitable draft order and on a non-transitioning state. Do **not** assume any specific response status. If no trustworthy negative evidence can be obtained, **stop**: another all-accept evaluation will not establish discrimination. The pilot is development evidence; do not insert it into holdout training.

In your independently operated HTTP client, the Swagger operation is `POST /orders/{id}/validate` (not a GET); only execute it with an authorized writer in the disposable clone. Its documented outcomes include 200, 304, 401, 404 and 500. A 304 is not guaranteed in every module/version/configuration. For invoices, `POST /invoices/{id}/validate` is documented, but the preceding run encountered 500 failures; inspect PHP/Dolibarr logs locally instead of interpreting those as legitimate business refusals.

## 1. Generate and bind matched fixtures (offline)

```powershell
py -m simlearn.dolibarr_counterfactual scaffold `
  --operation order_validate `
  --train-pairs 6 --validation-pairs 3 --holdout-pairs 10 `
  --out build/paired_template.json
```

The command creates `build/paired_template.json` and `build/paired_template_IDS.csv`. Open the CSV **locally** and fill `record_id` for all 38 rows with distinct numeric IDs of separately created disposable orders. Leave the `case_id`, `pair_id`, `operation`, and `split` columns unchanged. For non-payment operations, leave `payment_amount` blank. Keep the CSV and plan private.

Each `*-a`/`*-b` pair must have **identical observed safe fields except the declared contrast**. The default contrast is the order's `main.status`/`main.statut`; copy product lines, totals and other relevant configuration between the two fixtures and vary the intended lifecycle status. This does not guarantee a true causal match: timestamps, plugins, unexposed state, privileges and external side effects may differ. The runner compares only its narrowly allowlisted GET-observable fields. If either snapshot contains unexpected differences, `start-pair` refuses to proceed **without writing a pair commitment**, and you must reseed/fix the fixtures before continuing.

**Do not assume one pair will be accepted and the other rejected.** The actual labels come exclusively from later observed operator actions. Idempotent success reported as HTTP 200 without observable change is unknown; only a genuine 304 with no change can qualify as a no-op label. If both outcomes are the same, the pair is *non-discriminating* and cannot help satisfy the freeze minimum.

```powershell
py -m simlearn.dolibarr_counterfactual bind `
  --template build/paired_template.json `
  --csv build/paired_template_IDS.csv `
  --out build/paired_fixtures_PRIVATE.json

py -m simlearn.dolibarr_counterfactual init `
  --plan build/paired_fixtures_PRIVATE.json `
  --root build/dolibarr_paired_001

py -m simlearn.dolibarr_counterfactual status --root build/dolibarr_paired_001
```

The binder enforces that no target ID is reused by another case. The initial plan and tenant label are immutable after `init`. A fresh rerun uses a new workspace and new disposable records, not previously revealed holdouts.

## 2. Process train and validation paired interventions

Start both members of a pair **before either intervention**. This performs four bounded GET reads in total: two eligibility prechecks and two independently committed predictor reads. For holdouts it freezes the model's predictions before either manual action.

```powershell
py -m simlearn.dolibarr_counterfactual start-pair `
  --root build/dolibarr_paired_001 --pair train-order_validate-001
```

The output displays the two corresponding **private** case/record identifiers for the human operator. Make the action attempt yourself, with your appropriately authorized **writer** identity. Use `finish-case` immediately after each action to record a watcher GET after-state. Pick *exactly the case's true observed outcome*; never submit a desired label.

For an independently witnessed UI success **with observable scoped state change**:

```powershell
py -m simlearn.dolibarr_counterfactual finish-case `
  --root build/dolibarr_paired_001 --case train-order_validate-001-a `
  --outcome SUCCESS --source UI --operator-role authorized_writer
```

For an actual **304 Not Modified** returned to the separate authorized writer, and **no observed state change**, preserve a *locally redacted* text/JSON receipt and record:

```powershell
py -m simlearn.dolibarr_counterfactual finish-case `
  --root build/dolibarr_paired_001 --case train-order_validate-001-b `
  --outcome NO_OP --source MANUAL_API --operator-role authorized_writer `
  --receipt-status 304 --receipt-file .\private_receipts\pair001_noop_redacted.txt
```

For an actual, **documented application-level business-rule refusal**, **only** if returned status is 400, 409 or 422, with a real nonsecret machine error code and a redacted local receipt:

```powershell
py -m simlearn.dolibarr_counterfactual finish-case `
  --root build/dolibarr_paired_001 --case train-order_validate-001-b `
  --outcome BUSINESS_REJECTED --source MANUAL_API --operator-role authorized_writer `
  --receipt-status 409 --business-reason state_precondition `
  --business-code ACTUAL_SERVER_CODE --receipt-file .\private_receipts\pair001_reject_redacted.txt
```

**`409` and `ACTUAL_SERVER_CODE` are placeholders here, not claimed Dolibarr behavior.** Never report one unless your independently executed action produced that exact result. The helper hashes the local receipt file; it doesn't upload or copy its contents into the experiment.

For the two main inconclusive outcomes seen in v0.3:

```powershell
# Real HTTP 500 from an authorized writer: investigate PHP logs, do not call this a reject.
py -m simlearn.dolibarr_counterfactual finish-case `
  --root build/dolibarr_paired_001 --case train-order_validate-001-b `
  --outcome OTHER_ERROR --source MANUAL_API --operator-role authorized_writer --receipt-status 500

# Real HTTP 403 from a read-only or unauthorized identity: access control, NOT workflow denial.
py -m simlearn.dolibarr_counterfactual finish-case `
  --root build/dolibarr_paired_001 --case train-order_validate-001-b `
  --outcome ACCESS_DENIED --source MANUAL_API --operator-role read_only --receipt-status 403
```

These are **mutually exclusive example commands for the same case**. Run only the one matching what happened, once. You can check counts at any time with `status`; this is an offline local operation.

The runner will not accept `BUSINESS_REJECTED` for HTTP 403/500, nor an unobserved HTTP 304, nor a rejection with GET-observed changes. If the backend returns 200 but the target does not change, record `UNKNOWN` (for UI) rather than manufacturing a negative label.

## 3. Mandatory qualified-evidence freeze

Complete all train and validation pairs first. Then:

```powershell
py -m simlearn.dolibarr_counterfactual fit --root build/dolibarr_paired_001
```

If there are fewer than four qualifying train pairs or two qualifying validation pairs **per operation**, the command exits with an explicit reason and **does not create a frozen policy**. The separate `holdout` stage is inaccessible until it succeeds. A pair only qualifies when there is one eligible observed `SUCCESS` and one eligible observed `NO_OP` (304) **or** `BUSINESS_REJECTED` with the required receipt. No uncertain or role-blocked result is used as supervised negative evidence. `fit` then uses only the qualified train and validation case IDs to learn a tenant-bound CART policy; the same qualified evidence feeds the comparison bank.

All plans, receipts, watch snapshots and policy details remain local. The private tree is **not an authorization system**.

## 4. Fresh held-out pairs and optional GPT-6

Start only *untouched* `holdout-*` pairs after fit:

```powershell
py -m simlearn.dolibarr_counterfactual start-pair `
  --root build/dolibarr_paired_001 --pair holdout-order_validate-001
```

This commits both predictions before human action. Execute each action separately and finish it with the genuine witnessed outcome as in step 2. Do not change the tree, receipt taxonomy, fixtures, or plan after observing holdout outcomes.

The optional four-condition Model Refiner comparison is **explicitly opt-in**, transmits allowlisted but potentially sensitive numerical information, and requires a supported model key:

```powershell
py -m pip install -e ".[openai]"
# Configure OPENAI_API_KEY *locally*, without saving or uploading it.
py -m simlearn.dolibarr_counterfactual start-pair `
  --root build/dolibarr_paired_001 --pair holdout-order_validate-001 `
  --execute-model --share-sanitized-financials --model gpt-6-luna
```

It performs **up to eight logical model predictions per pair** (four adaptation conditions × two members); exact provider billing depends on availability/usage. No model call runs by default. The comparison is advisory, does not retrain provider weights, and cannot execute writes.

## 5. Verify, report, share only public files

```powershell
py -m simlearn.dolibarr_counterfactual verify --root build/dolibarr_paired_001
py -m simlearn.dolibarr_counterfactual report --root build/dolibarr_paired_001
```

`verify` checks the inherited event hash chain plus pair comparability, two-member prediction precommit, precommit-before-receipt ordering, qualified-only training use and freeze-before-holdout ordering. Integrity is locally computed, **not externally timestamped**. `report` is immutable and requires both members of every held-out pair to be finished, even if some are unknown. It generates:

- `build/dolibarr_paired_001/public/PAIRED_REPORT.json`
- `build/dolibarr_paired_001/public/PAIRED_AUDIT_FREEZE.json`

**Upload only those two files for review.** Keep your CSV, plan, receipts, private snapshots, raw logs, credentials and Docker backups locally.

The report includes paired class balance, always-accept and always-reject baselines, predictor accuracy/coverage/false accepts, error-type dropouts, and *separate* booleans for **nontransition classification** and **true business-rule refusal evidence**. State effects are only scored when an effect formula was actually learned and predicts exact observed changes. Unknowns are counted against the planned evaluation coverage, never converted into successes.

## Known limits

This is a genuine same-tenant, operator-attested intervention experiment, **not independent verification of business rules**. A pair is only matched over the small set of allowlisted GET-observable fields; hidden state may differ. Many Dolibarr endpoints have version-, rights-, or module-dependent semantics; a 304 indicates a no-op, not a 4xx business refusal. A real business-rule error can be encoded as HTTP 500 on some systems, but that cannot be reliably distinguished from an application crash by status alone—do not relabel it without a separate research protocol and new held-out cases. A near-perfect result on only 304 no-ops establishes lifecycle nontransition classification, **not** general payment, credit, invoice, or inventory policy reconstruction. No live writes are automated or implied.
