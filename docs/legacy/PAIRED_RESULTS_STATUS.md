# Real Dolibarr v0.4 paired transfer — result status

**No v0.4 real Dolibarr test has been run in the assistant environment.** The experiment is packaged and offline-verified. The earlier v0.3 external Dolibarr run is historical evidence and **cannot** be reused as a fresh holdout.

## Why this release exists

The v0.3 run completed 24 held-out intervention attempts, but only 17 were labeled and all 17 were accepts. Its retrieval model's 17/17 could not beat an always-accept baseline. Six invoice attempts returned HTTP 500 without observed changes, and one attempt reached an identity returning 403. None established a business-rule rejection. An unfitted tree appropriately abstained throughout. The v0.4 protocol blocks an all-accept training set **before** revealing any holdout.

The new paired runner uses two conceptually separate negative classes:

1. `NO_OP` — an observed HTTP 304 with unchanged read-observable state: a **nontransition**, not proof of rejection by a business rule.
2. `BUSINESS_REJECTED` — an explicit server business-error response classified by a separately authorized operator, with HTTP 400/409/422 and a locally hashed redacted receipt. This remains operator-attested, not independently authenticated.

`ACCESS_DENIED`, `OTHER_ERROR` (including HTTP 500), 404/missing fixtures and unconfirmed outcomes remain unknown for training. Only pairs with one scoped success and one eligible no-transition/refusal may be used for the paired benchmark. A strict freeze check requires both classes in training and validation per operation. An internally hash-verified audit preserves pre-intervention prediction ordering. The public report has **two separate quality-gate booleans**, one for transition classification and one for genuine business refusal evidence.

**Do not report live accuracy, workflow fidelity, or deployment readiness until the user conducts a fresh real-tenant trial.** Model-provider calls and ERP write operations are zero by default. The runner never performs writes, even when model inference is enabled.

See [operator instructions](DOLIBARR_PAIRED_OPERATOR_GUIDE.md).
