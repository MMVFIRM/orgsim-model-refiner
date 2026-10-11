# v0.9.0 — hardened consolidation

- Preserves v0.8 active executable policies, oblique learner, effect learner,
  simulator, Model Refiner source, benchmarks, and legacy fixtures.
- Adds allow-listed GET-only production observer with HTTPS enforcement for
  remote targets, no redirects, no forwarded credentials, response-size caps.
- Requires an explicit short-lived clone permit and a verified unique sentinel
  product before every automated mutation pathway can issue POST.
- Refuses non-loopback mutations, parallel mutation sessions, and write retries
  after ambiguous network failures; direct old `--oracle live` invocation without
  `--clone-permit` is blocked.
- Converts generic HTTP 500 and incomplete write readbacks into unknown/error,
  not verified business rejection; known stock/sale refusals must have unchanged
  observed order and product stock.
- Uses actually observed ERP post-state for successful AND refused replication
  outcomes rather than assuming unchanged state on rejected cases.
- Adds tests for network controls, permit expiry, target mismatch, ambiguous
  write handling, and presence of legacy entry-point guards.
- Historical v0.8 measurements retained as historical material, not new
  efficacy certification. No live Dolibarr or provider calls were made here.

# v0.8.0

Real-ERP learning and action replication. Details and evidence: `docs/ERP_LEARNING.md`.

- Added `oblique.py` (exact small-integer oblique decision trees, Occam selection over several build modes) and `expr_learner.py` (counterexample-guided loop with exact boundary probes). v0.7 `learner.py` is unchanged.
- `PolicyArtifact.predict` evaluates the AST directly for expression policies (empty region table).
- `EffectModel` learns `before ± sum(action fields)` and categorical single-field conditionals.
- `TrainingOracle.query_many` (budgeted, reserved-input-safe, optionally concurrent).
- `erp/` package: Dolibarr order-validation task, documented-rule twin, and a loopback-only live oracle that labels from receipt AND read-back state, retries transient server faults, and serialises transactions (concurrent validation was measured to corrupt labels).
- New provenance class `REAL_API_VERIFIED`.
- Tools: `erp_experiment`, `erp_replicate`, `erp_drift_check`, `erp_rule_suite`; `lab/` disposable-Dolibarr setup.
- Tests: `tests/test_v08_expr.py` (learning, noise never promoted, reserved inputs never queried, live-oracle safety rules with a fake ERP, effects).
- Known limits: integer-rounding arithmetic rules may not converge (they fail closed); evidence is one tenant/operation/configuration.

# v0.7.0

Integrated active executable-policy framework added in `src/orgsim_refiner`.

- Replaced unverified English summaries as the primary learning object with typed ASTs and empirically audited region policies.
- Added hard tenant/operation/configuration isolation, full-evidence replay, contradiction quarantine, bounded active queries and promotion-only calibration.
- Added finite non-monotone boundary discovery, conservative uncertainty, canonical cross-platform artifacts and an append-only ledger.
- Added separate empirical effect learning, pure predicted simulator, actual Model Refiner inference bridge and optional AST proposer.
- Added one-study frozen evaluation, matched-data comparators, shared-partition reference, boundary-focused stress, resource controls, and explicit logical-signature coverage.
- Added observed-outcome inbox, local Dolibarr evidence/workspace checks, and historical audit regression.
- Preserved v0.6.1 application source and frozen Dolibarr protocol. Historical documents moved to `docs/legacy`.
- Corrected two inherited privacy tests that mistook a coincidental four-digit substring of a SHA-256 digest for a disclosed record ID. Tests now check standalone identifier tokens. No live evidence, classifier, protocol, or scoring rule was changed for this correction.
- Added a project-local temporary-directory test launcher for Windows permission issues.

Development failures and fixes: a midpoint-based AST compiler initially misplaced learned cuts; the final compiler uses region-index splits translated to exact cutpoints and replays every learning label. This was fixed on development data before the supplied confirmation. One-dimensional reserved-input conflicts remain fail-closed rather than querying heldout labels. The delivered result reports include all non-promotions and controls.

Actual provider and Dolibarr efficacy for v0.7 remain unmeasured. Previous synthetic/live results included in legacy folders are not new release tests.
