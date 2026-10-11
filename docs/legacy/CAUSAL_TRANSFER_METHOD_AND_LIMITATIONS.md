# v0.5 experiment method and limitations

## Hypothesis

On a real, disposable Dolibarr 24 tenant, a Model Refiner system trained on verified before/action/after observations should distinguish two **draft** orders that differ in a functional business precondition, without exploiting order lifecycle status. Training examples are *real operator-attested transactions*, not OrgSim pseudo-labels. A secondary experiment asks whether compiled policy/retrieval/synthetic rehearsal adds predictive value beyond unadapted GPT-6.

## Source-derived candidate rule

`Commande::valid` in Dolibarr 24 calls `checkActiveProductInLines()` unless the configured `ORDER_NOCHECK_ONSALE_PRODUCTS_ONVALID` bypass is active. A validation failure can become HTTP 500 in its REST layer. All specifics must be confirmed against the user's own Dolibarr container. Reference: [Dolibarr 24 order class source](https://doxygen.dolibarr.org/dolibarr_24.0/dev/build/html/dc/df5/commande_8class_8php_source.html), [REST handler](https://doxygen.dolibarr.org/dolibarr_24.0/dev/build/html/d9/d7b/api__orders_8class_8php_source.html).

## Design

- **Pilot gate**: 2 disposable draft orders, each with one product; same status, same totals and public safe fields, sale-status mismatch. The first expected success requires a GET-observed transition; the negative must be accompanied by an HTTP 500 Dolibarr validation prefix and a separate local product-validation diagnostic (not just a generic 500). The pilot and receipt content remain private. If this is not reproducible, do not generate a study.
- **Study**: 5 train + 3 validation + 10 holdout pairs by default, each using independent one-time order IDs. Within each pair both statuses are `draft`, both raw order lines are linked to their watched product IDs, and only sale-enabled product status differs among the strictly allowlisted snapshot fields. Pair-side assignment alternates.
- **Evidence**: `SUCCESS` needs operator authorization and observed state change. `BUSINESS_REJECTED` needs separately authorized writer, original 400/409/422 receipt or correlated application 500 validation refusal, and no observed change. HTTP 304, 401/403, 404, uncorroborated 500, 2xx without a detectable change and unknown are *not* negative training labels.
- **Fit**: min 4 qualified train and 2 qualified validation pairs; class-balanced. Both the real-example memory and the structured decision tree are derived solely from qualified training evidence. Holdout cannot be opened before fit and signatures are written before manual actions.
- **Score**: all scheduled holdout cases count in denominator; qualifying balanced opposite-outcome pairs are scored, missing/unqualified pairs decrease planned coverage. Report accuracy, false accepts, false rejects, coverage, effect correctness and an adaptation gain flag only if both the baseline and policy-model arms were run. Baselines: always accept, always reject, majority, status-only, structured tree, optional unadapted GPT-6, retrieval, simulator rehearsal, compiled-rule GPT-6.

## Leakage/safety limitations

The intended product status is given privately to the *fixture operator*, and the underlying source code makes it a plausible known rule; the model may infer it from input alone. No transfer benefit is claimed if unadapted GPT-6 also predicts perfectly. Counterfactual matching covers only allowlisted GET-observable fields; hidden triggers, configuration, product history and user rights are uncontrolled except by the operator's provisioning. The negative application error requires manual, locally correlated diagnostic evidence, not signed server receipts. The evidence chain is internally verifiable but not externally timestamped. The protocol cannot certify permission or safety for any real ERP writes. It is deliberately limited to one well-defined same-tenant business condition, rather than claiming general ERP reconstruction.
