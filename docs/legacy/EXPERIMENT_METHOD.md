# Methods: one consolidated simulation-to-adaptation experiment

## Research question

Can a *completely frozen* model acquire organization-specific workflow competence from a learned simulation via Model Refiner's external adaptation layer? Before incurring genuine model API cost, establish an offline ceiling and a group-safe, reproducible comparison using conventionally trained adapters and the exact Model Refiner dataset/runtime interfaces.

## Real implementation boundary

- **OrgSim 0.2.6** `SimulationEngine` runs precondition guards and transactional effect steps against its SQLite state. Failed preconditions yield `POLICY_DENIED` without partial state changes. This interpreter generates every labeled trajectory.
- Four workflows modify up to four entity types: order validation adjusts order status, product stock and customer credit; invoice payments adjust balance, payment count and customer receipts; reservations adjust product stock/reserved and order allocation; order closure changes order status and customer counters.
- Per-environment hidden policies alter eleven threshold settings, including *upper* caps and joint vetoes. Model prompts and supervised features contain state/action values but no threshold values, oracle source code or compiled manifest. The code is present in the distributed research package for reproducibility and **is not an independent external black-box holdout**.
- The allowed hypothesis class is known/compact: arithmetic prerequisites, upper bounds, at most one two-variable conjunctive veto per operation, and numeric addition/subtraction side effects. This is a **scoped** benchmark; no claim about arbitrary Dolibarr rules.
- Three seeds, four operations, and the same schema but distinct business thresholds. Training and test for a given organization share that organization's policy; each organization receives its own training data. There is no test of *zero-shot* transfer of a policy to a never-seen organization.

## Data generation and splits

Each environment produces per-operation 400 train, 100 router, 100 validation, 200 test, and 200 shifted-stress records. Every per-operation partition is **50% ACCEPT / 50% REJECT**, obtained by querying the simulator, with extra draws/denials counted. Scenarios are de-duplicated by a SHA-256 state-action fingerprint; group identifiers are disjoint across all partitions.

**Training records** are available to retrieval, numeric-effect induction, and fitted conventional classifiers. **Router records** are reserved for Model Refiner's contextual Artificial Weights. **Validation records** may be used to select conventional classifiers, decision thresholds, retrieval `k`, and native static recipes; they are not valid training retrieval demonstrations. **Test/stress records** are not supplied to fitting or hyperparameter selection. Scoring is done after model selection. The test labels exist in local fixture files for reproducibility, so this is **not an externally controlled sealed benchmark**. Do not retroactively tune on these disclosed results and call the next replay a fresh holdout.

The event strings are canonical JSON. Model Refiner `Record.from_dict`, `Dataset.prepare`, `TaskSpec`, `Memory`, `sdk.fit`, and immutable `SuiteArtifact` are used directly from v0.5.0rc1. The bridge does not substitute a similar API.

## Baselines and learning

1. Always `REJECT` (50% by balanced design). Demonstrates that avoidance of risky operations alone is insufficient.
2. Public-precondition heuristic. Uses plausible public status, stock and credit rules, *not* random organization-specific thresholds. It predicts all rows without abstention.
3. **Native Model Refiner TF-IDF retrieval** (`model_refiner.suite.runtime.Memory`) with k-neighbor majority labels. The k value is selected on validation.
4. Conventional simulated-data adapter. Scikit-learn decision tree, ExtraTrees, random forest and histogram gradient boosting candidates trained separately per operation; candidate and threshold are selected **only** on validation. Derived feature engineering includes stock/credit/shipping/payment margins, exposed to all model inputs as state facts, but no private policy thresholds.
5. Effect induction. Finds a compact state-change expression from *executed accepted training trajectories*: small constants, plus or minus existing public input/state fields, or a categorical constant. It must reproduce every successful validation effect to be marked `audited` and remains `reviewed:false` for any real ERP.

The default offline benchmark runs **no LLM and does not exercise native Artificial Weights inference**. A distinct, opt-in native Model Refiner run uses actual `sdk.fit`, budget reservations, baseline, retrieval-only, supervised static and Artificial Weights recipes. `--mock` exercises the exact pipeline with a fixed public-rule backend but is not model evidence. `--execute-model` sends synthetic inputs to a genuine model provider. For a fair result, the native run additionally evaluates conventional baselines on the **identical** selected train and test identities; do not compare it directly with the full 400-example offline adapter.

## Metrics and decisions

Each benchmark partition reports accuracy and balanced accuracy, false-accepts, false-rejects, false-accept rate, positive recall and precision, and exact match of *every changed field* on correctly accepted operations. End-to-end exact means either (a) a correct reject and no state changes or (b) a correct accept and exact reconstruction of the state-change map; wrong accepts always fail. In the benchmark every arm issues one decision on every case (100% decision coverage). Genuine Model Refiner output can also be invalid or incomplete, reported via its native runtime.

**Offline performance goals** (illustrative development thresholds): >=95% decision accuracy, >=95% end-to-end accuracy and <=2% false-accept rate on the *balanced synthetic test*. This is not sufficient to authorize live financial or inventory writes. For live autonomous action the experimental standard must be far stricter, particularly for false accepts and state-effect verification.

## Limitations and risks

- Programmatically synthesized organizations share rule families and schemas. A model may learn that grammar rather than general enterprise reasoning.
- Simulator data is correct only relative to the provided synthetic manifests. Dolibarr 24.x workflow ground truth was not supplied for these examples.
- Effect grammar specifically fits additive side effects. Arbitrary tax, invoice, inventory reservation, role, time or integration logic may not be representable.
- Bounded numerical values and a balanced evaluation set do not represent enterprise operating distributions.
- Each labeled rejected/successful simulator interaction is counted in oracle-query totals; balanced training is more data-hungry than the final example count suggests.
- Native Model Refiner provider runs and the Artificial Weights route must earn a measured improvement over both baseline and retrieval under matching train/test evidence. The offline model accuracy does not substitute for that result.
- Full state/action values in research files are **synthetic**. Do not repurpose the exporter for live customer records without a privacy and authorization review.
- Local SHA-256 freezes establish repeatable content integrity, not external attestation or experimenter blindness to available code.
