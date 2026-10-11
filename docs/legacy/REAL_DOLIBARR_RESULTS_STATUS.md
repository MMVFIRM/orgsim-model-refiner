# Real Dolibarr experiment — pending external run

**Status:** The study harness has been implemented and tested offline, but **no Dolibarr business operation has been evaluated by this release in the assistant environment**. The user has access to the actual disposable local Dolibarr sandbox and will conduct any transaction attempts there. No live-ERP accuracy figure is currently available.

## Evidence available before the new experiment

- The earlier Gate 2G GET-only experiment predicted 86 of 89 selected fields correctly (96.6% claim accuracy, 74.1% fixed-denominator target recall), but **did not** reconstruct state transitions.
- OrgSim's synthetic policy learner attained 100% accuracy within its public-feature, monotone-upper-interaction rule class. The synthetic out-of-grammar audit failed and abstained on many interventions. That result is not transferable by assumption to Dolibarr.
- An earlier real GPT-6 Luna inference experiment scored 64/64 on a synthetic tenant **given the compiled policy**, versus 38/64 without it. It did not change neural weights. It did not use real Dolibarr ground truth.

## What's being tested now

One human-controlled, snapshot-backed real-world holdout. The executable bridge captures whitelisted Dolibarr numeric/status fields via GET before and after user-executed UI/API workflow attempts. All manual outcomes are explicitly marked as human-attested. It learns a real-tenant local decision tree from training and validation examples and freezes its predictions before separate holdout actions.

Optional inference arms using Model Refiner's native Recipe/Predictor classes: plain model, retrieval from actual training traces, retrieval from program-generated counterfactual rehearsal examples (pseudo-labeled), and structured program prompt. This partitions claims about *retrieval versus rehearsal versus program transfer* instead of assuming all adaptations count as simulator fidelity.

**No real ERP writes or model-provider calls are made automatically by the release.** Model calls require explicit flag and consent to send selected numerical organizational data. ERP writes are performed externally by the operator, never by the bridge.

## Test and evaluation caveats

The sample script defaults to 8 training, 4 validation and 12 holdout examples for each of two operation families. Each case needs a distinct local fixture ID. A 24-case held-out sample is an initial feasibility test, not strong evidence of production reliability. Errors, abstentions, unknown outcomes and missing labels must be reported, not dropped from operational coverage claims. The score distinguishes all human-attested labels from a narrower read-corroborated/manual-API-receipt subset.

A model learned from one organization's successful transactions might fail badly for another organization's permissions, module configuration, custom triggers and workflow rules. Tenant identity and the snapshot provenance are operator assertions rather than cryptographic attestations. Immutable local files and hash chains detect ordinary modification but do not independently authenticate real-world transactions.

See [full operator guide](DOLIBARR_REAL_CAUSAL_EXPERIMENT.md).
