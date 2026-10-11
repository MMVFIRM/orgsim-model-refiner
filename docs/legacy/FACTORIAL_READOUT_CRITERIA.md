# v0.6 pre-registered research contract

**Primary hypothesis:** For held-out real Dolibarr order validation in one configured throwaway tenant, Model Refiner's GPT-6 recipe supplied with an *oracle-observation-derived, tenant-bound executable policy* improves decision accuracy over matched-evidence GPT-6 retrieval.

**One factorial test only:** 16 groups of four outcomes (11, 01, 10, 00); 64 held-out cases. No double-fail combination 00 appears in pilot/train/validation. Each block includes fresh products/orders and same draft status, quantity, and amount. A=yes on-sale; B=sufficient physical stock. Both mechanisms must first independently produce application-level business-rule refusals with correlated local evidence. One mutation per case by a separate authorized operator; reader is GET only.

**Baseline comparators:** always accept/reject; sale-only; stock-only; transparent AND; acquired program; optional 4 real Model Refiner inference conditions: no learning, retrieval of real training and validation cases available before the holdout, pseudo-labeled simulated rehearsal, and acquired program recipe. The only native Model Refiner integration is Recipe/Predictor external adaptation, not updating base weights or a `suite.fit()` run.

**Decision gate:** >=95% accuracy over qualified cases, >=95% planned coverage, zero false accepts, full or nearly full factorial representation (the default report presently requires all 4 cells completely qualified), precommitted holdout, no prospectively altered labels. Any unknown/unclassified case fails the current strict gate.

**Incremental advantage gate:** decision gate plus completed GPT-6 comparisons on every scored case, learned-policy recipe >=95% accuracy and coverage and zero false accepts, at least +10pp versus retrieval, and lower endpoint of 2500-resample **block** bootstrap difference >0. This is descriptive uncertainty, not a familywise-controlled multi-tenant trial. If baseline or retrieval ties, no superiority claim.

**Effect gate:** Separately frozen formula for stock level after success (`new stock = old stock − line_qty`) must be supported on ALL training and validation successes; only then score that field on qualifying successful holdout cases. Status effects are scored separately. Never count LLM classification as an LLM state-effect prediction.

**Stop rule:** Pilot fails for any configuration/stock/receipt reason => infeasible, do not fabricate a second rule. Training/validation fails => no holdout. If any heldout report fails, record a failure or inconclusive result rather than creating post-hoc gates. No real environment is connected from this package; all real writes remain manual and independently authorized, with locally stored source receipts.

**Scientific constraints:** Two factor identities and stock comparison were specified before the experiment; this is evidence of applying one known bounded grammar in a tenant, **not discovery of arbitrary unknown rules**. Likewise, a directly-coded transparent conjunction may tie at 100%, so positive program accuracy alone cannot establish unique value. Repeated cases within a tenant are correlated and receipt authenticity is not externally attested.
