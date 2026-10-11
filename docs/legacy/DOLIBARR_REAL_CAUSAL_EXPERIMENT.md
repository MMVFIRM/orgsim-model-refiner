# One integrated real-Dolibarr experiment: executable learning from observed workflows

**Version:** OrgSim × Model Refiner v0.3 experimental integration. **Default risk:** strictly GET-only; the operator, not the harness, performs any intervention. This is a **real-world measurement protocol**, not a claim that our existing synthetic simulator already reproduces Dolibarr business logic.

## Question and experimental arms

Can externally learned, tenant-specific *executable decision rules* improve the performance of a frozen LLM on previously unseen, actual Dolibarr operations, beyond a simple baseline, without trusting guessed OrgSim policies as if they were ground truth?

Our existing 100% synthetic results came from a known monotone-upper-interaction grammar, public feature axes and invented OrgSim workflow semantics. They **cannot be mapped directly** to Dolibarr. This experiment instead learns an advisory policy from **real, user-attested, read-corroborated Dolibarr interventions**. The comparison arms are (a) an operation-specific majority-label baseline; (b) a train-only structured decision tree; and **four optional GPT-6 Luna conditions**: plain prompting, retrieval of real labeled training examples, retrieval of counterfactual rehearsal examples *pseudo-labeled by the fitted tree*, and prompting with the frozen executable-tree decision paths. All model calls use the same before-state and input features. This separates ordinary retrieval, policy-derived simulated practice, and direct program transfer. It tests *knowledge transfer into inference*, not provider fine-tuning or native `suite.fit` optimization.

The test additionally measures scoped after-state field effects when train-derived deterministic formulas are available, not complete ERP database state.

## What Dolibarr API 24 Swagger actually documents

The user-provided Swagger (Dolibarr 24.0.1) lists these write operations, **for operator reference only**: `POST /orders/{id}/validate`, `POST /orders/{id}/close`, `POST /invoices/{id}/validate`, and `POST /invoices/{id}/payments`. The last requires multiple payment/account fields and is documented as paying the remaining amount; it is **not** interchangeable with an arbitrary partial-payment payload. We therefore recommend using Dolibarr's UI for interventions. The research runner **never calls those endpoints** and has **no live write mode**.

Only normal GET routes are built by the runner: `/orders/{id}`, `/invoices/{id}`, optional related `/products/{id}` or `/thirdparties/{id}`, and `/invoices/{id}/payments` for payment trials. IDs must be decimal and all targets must be on `http://127.0.0.1:<port>`. HTTP redirects are refused by the underlying client. API keys are read from `DOLIBARR_API_KEY`, not saved in the artifacts or placed on command lines. A fully sandbox-only UI/admin identity is separate from the read-only `orgsim_gate` API key.

## Prerequisites

1. A **fully disposable cloned** Dolibarr 24.x tenant bound to `127.0.0.1:8088`. Confirm it contains no live business data, side-effecting integrations, email/SMS jobs, payment processor hooks, webhooks, or production CRM synchronization. Have a reversible database **and document-store** snapshot plus rollback procedure before trying any write. The harness cannot verify the presence/quality of a Docker snapshot; a string `snapshot_label` is an operator assertion.
2. Read-only GET API access via `orgsim_gate`, configured locally as `DOLIBARR_API_KEY`. Real actions are performed manually with a separate authorized sandbox-only UI account or separately executed API client. The runner cannot be used to perform them.
3. Distinct disposable record IDs for every planned case, across training, validation and holdout. This prevents running later cases on an already-mutated target or leaking record identities. `order_validate` and `invoice_validate` are the recommended first two operations. `order_close` is optional; `invoice_payment` requires careful manual payment/account setup and numeric `payment_amount` in the plan. Do not invent state transitions merely to complete a quota.
4. Both successful and deliberately invalid attempted operations per operation type. UI buttons being absent is **not** proof that Dolibarr rejected a submitted operation; record such cases as `UNKNOWN` unless a real attempt and outcome occurred. With a manual API, record the true observed status as a *user-reported* receipt. A status of 403/404 is permission/not-found evidence, not necessarily a business-rule rejection.

## Install and prepare an immutable case plan

Use PowerShell from the extracted release root:

```powershell
py -m pip install -e ".[test]"
py -m pytest -q tests

# No network activity. Creates an intentionally incomplete template.
py -m simlearn.dolibarr_transfer scaffold `
  --operation order_validate --operation invoice_validate `
  --train 8 --validation 4 --holdout 12 `
  --out build/dolibarr_cases_EDIT.json
```

Edit `build/dolibarr_cases_EDIT.json` **locally**. Assign a *different* decimal `record_id` in each entry and replace the `tenant_label` and `snapshot_label` with labels for your disposable clone. The `FILL_ID` placeholders deliberately prevent initialization or network access. For a payment case also replace `FILL_AMOUNT`. Extra GET watchers can be configured as objects such as `{"family":"products","record_id":"45"}`; at most four per case. The watcher API is also GET-only.

```powershell
# Do not modify the private plan once initialized.
py -m simlearn.dolibarr_transfer init `
  --plan build/dolibarr_cases_EDIT.json `
  --root build/dolibarr_real_run_001
```

The default 2-operation configuration entails 16 training, 8 validation and 24 holdout actions (48 total) performed **manually**. This is an initial feasibility-scale sample, not a production-strength evidence base. More independent fixtures and multiple cloned-tenant policies are needed for high-confidence performance estimates.

## Step 1: capture and execute train/validation cases

First capture a case, **before** performing the intended operation:

```powershell
$env:DOLIBARR_API_KEY = "<local read-only sandbox API key>"
py -m simlearn.dolibarr_transfer start `
  --root build/dolibarr_real_run_001 --case train-order_validate-001
```

Now, using only your separate authorized **disposable-clone UI**, attempt the `order_validate` action against the designated order. Record whether Dolibarr actually accepted or rejected it. **Do not** use the read-only API identity for a write. Immediately capture the after-state:

```powershell
# Use the case's actual outcome. This example is ONLY for a verified success.
py -m simlearn.dolibarr_transfer finish `
  --root build/dolibarr_real_run_001 --case train-order_validate-001 `
  --outcome SUCCESS --source UI
```

For a real attempted business-rule failure with an observed API 4xx status, the command would instead be:

```powershell
py -m simlearn.dolibarr_transfer finish `
  --root build/dolibarr_real_run_001 --case train-order_validate-002 `
  --outcome BUSINESS_REJECTED --source MANUAL_API --receipt-status 400
```

Only supply that result if your independent API operator really attempted the action and observed 400; never use the command to manufacture ground truth. `--outcome ACCESS_DENIED`, `OTHER_ERROR`, and `UNKNOWN` remain distinguishable and are *not* mislabeled as business-rule failures. A claimed success without any scoped GET-observable change, or a claimed rejection with a change, is automatically marked `UNKNOWN` and excluded from decision training. UI rejection without a recorded HTTP response is lower-quality human attestation even if no change was observed.

Complete the other `train-*` and `validation-*` cases first. **Don't inspect or attempt the holdout actions yet.** Each start and finish is immutable and appended to a locally hash-chained event log.

## Step 2: freeze tenant-specific rules

```powershell
py -m simlearn.dolibarr_transfer fit `
  --root build/dolibarr_real_run_001 --min-train 6
```

This creates `private/FROZEN_POLICY.json` and `private/MEMORY_BANKS.json`. The rehearsal bank is **not** an OrgSim execution oracle: it perturbs real training features, assigns provisional labels with the fitted decision tree, and marks every generated case as `SIMULATOR_PSEUDO_LABEL_UNVERIFIED`. These samples are used **only** as lower-trust examples for the optional retrieval experiment, never to fit or validate the tree itself. The effect of this synthetic practice versus retrieval of actual labeled examples is an independent comparison in the same held-out evaluation.

The policy freeze trains a small, auditable CART rule tree from training trajectories only, selects depth/decision thresholds using validation only, and attempts to infer **only observed deterministic scoped state-effect formulas**. The underlying rule class is deliberately broader than the old synthetic monotone-veto grammar (learned numeric/categorical splits), but may still be insufficient for genuine Dolibarr semantics.

If an operation lacks at least six labeled train outcomes and two of each class, it is marked insufficient and its predictions abstain. Do not change the original plan or retroactively relabel revealed holdout cases just to achieve a higher score; create a fresh, genuinely withheld cohort if the design proves underpowered.

The learner **does not import** any synthetic tenant's fitted program as a Dolibarr authority. `FROZEN_POLICY.json` is bound to the local tenant/snapshot identity and includes hashes of the permitted training and validation cases. Inference cannot trigger a business transaction.

## Step 3: frozen held-out trials

```powershell
py -m simlearn.dolibarr_transfer start `
  --root build/dolibarr_real_run_001 --case holdout-order_validate-001
```

The command makes a bounded GET capture and **commits the majority and structured-model predictions before your separate intervention**. Then attempt that action in the UI and record its real observed outcome with `finish` as above. All holdout scores are computed only from those already committed predictions, including abstentions and false accepts. Never call `fit` a second time on this workspace.

### Operator progress helper

At any stage, run this **offline, local-only** command to see how many cases remain and which fixture ID to process next:

```powershell
py -m simlearn.dolibarr_transfer status --root build/dolibarr_real_run_001
```

It does not contact Dolibarr or a model provider. Its console output can contain private fixture IDs; do not upload it.

### Optional actual Model Refiner / GPT-6 comparison

This is **opt-in**. It uses Model Refiner's native `Predictor` and `Recipe` inference across four prompt conditions (plain, real retrieval, policy-generated rehearsal retrieval and compiled policy). No `suite.fit` is performed and the frozen provider's weights are not changed. To explicitly allow uploading **sanitized but still potentially sensitive financial/state values** to the model provider, install `.[openai]`, configure `OPENAI_API_KEY` locally, and pass the two flags *only when starting an untouched holdout case*:

```powershell
py -m pip install -e ".[openai]"
py -m simlearn.dolibarr_transfer start `
  --root build/dolibarr_real_run_001 --case holdout-order_validate-001 `
  --execute-model --share-sanitized-financials --model gpt-6-luna
```

This requests **four provider inferences per case** (plain, real-example retrieval, pseudo-rehearsal retrieval, compiled policy), with repairs disabled and no implicit retries; the displayed accuracy remains subject to the model's response limits and provider pricing. No raw customer names, email addresses, references, keys, API IDs, or whole API bodies are intentionally transmitted. Rehearsal and real-example retrieval can expose numeric features from earlier training trials as well as the current case. The additional rehearsal inputs are **derived from the fitted policy**, not observations of real Dolibarr responses, and do not certify sim-to-real fidelity. **Do not enable** this option if your sandbox contains private or regulated organizational information, because even selected numeric balances are potentially sensitive. The model prompt is advisory and never automatically authorizes or executes writes.

## Step 4: local verification and portable report

```powershell
py -m simlearn.dolibarr_transfer verify --root build/dolibarr_real_run_001
py -m simlearn.dolibarr_transfer report --root build/dolibarr_real_run_001
```

`verify` checks event hashes, immutable record contents and freeze-before-holdout event ordering. It is not externally timestamped. `report` generates a **sanitized** `public/REPORT.json` plus `public/AUDIT_FREEZE.json`. It can be run only once per workspace so the frozen report cannot be silently replaced. You can upload **only those two public files** for analysis. Keep `private/` (including the plan, local record IDs, selected monetary fields and evidence journal), DB backups, `.env`, API keys and raw HTTP bodies on your own computer.

## Pass criteria and negative controls

This is **one combined experiment**, not another series of gates. Report accuracy on *all* **labeled** held-out cases (abstentions counted as errors) **and the correct/planned rate including missing labels as failures of evidence completeness**, coverage, false accepts and rejects, and optionally accuracy in the more strongly corroborated subset. Require at least 95% decision accuracy **and** at least 95% coverage **and** zero observed false accepts on a sufficiently diverse real-tenant holdout before treating the policy as a candidate. Report comparisons against the matched majority baseline and optional frozen-model baseline, not just a top-line score. These thresholds are research targets; even reaching them does not authorize production writes or establish 100% safety.

Evaluate side effects only where the exact changed-field/value prediction was learned from training and passed available validation. A missing formula is **unscored**; changes outside the allowlisted GET fields are unobserved rather than assumed absent. Reuse of a policy from a different clone/tenant is prohibited; one tenant's rules can be wrong for another. Include role and module configuration when interpreting 403/404/503; transient outages are not necessarily stable policies.

### Fundamental limits

- **Ground truth is operator-attested**, with some success outcomes corroborated by GET differences. The harness cannot independently prove which action caused a state change, nor that a stated failed attempt actually reached the server. Manual API receipts and independent transaction/database logs would strengthen evidence.
- Original OrgSim synthetic `reserve_stock`/payment/credit semantics are not automatically equivalent to actual Dolibarr 24 REST behavior; the synthetic 100% must not be transferred as a real-world accuracy claim.
- The runner is passive GET-only. Permission to execute a transaction remains outside its control. The absence of automatic writes is intentional while validating real business logic.
- With only a few existing invoices/orders (as in the early Dolibarr test tenant), it is **not possible** to credibly claim high generalization accuracy on business workflows. More independently seeded disposable fixtures and varied outcomes are prerequisites.
