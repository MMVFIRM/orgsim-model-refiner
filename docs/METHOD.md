# One-study method

## Objective and claim boundary

Test whether additional **observed simulator experience**, represented as executable scoped policies, can improve decision reliability compared with passive examples and ordinary raw-feature learners. The objective is accuracy with explicit evidence cost, not a claim that Model Refiner is a uniquely superior learning algorithm.

This is a new synthetic benchmark. It is not a repaired blind rerun of `hard_transfer_003`: that archive lacks executable oracle definitions and is already disclosed. The historical archive is used only for regression auditing.

## Fixed procedure

The configuration and source hashes are written before the study. Each tenant's oracle specification and train/calibration/holdout/stress rows are sealed before learning starts; the learner receives only public schema and scope, seed training observations, a bounded observation callback, and prohibited evaluation-input hashes. It does not receive hidden thresholds, feature flags, formula names, test inputs or labels.

There are 24 independently parameterized tenants from eight *declared synthetic formula families*: conjunction, disjunction, nesting, XOR, conditional branches, counting, non-monotone interval exceptions, and signed mixtures. These are eight structures with randomized parameters, not 24 entirely novel structures. Raw inputs include four bounded integer fields and two four-valued categorical fields.

Defaults per tenant: 32 balanced initial labels, 128 independent calibration cases, 256 held-out cases and 256 boundary-focused stress cases; at most 8,192 training oracle attempts. Dataset-generator calls used to construct balanced sets are counted separately from learner queries. Artificial label balance does not estimate an organization's operational base rate.

Stress changes sampling: one numeric coordinate is placed within one integer of a true evaluator-only threshold or interval boundary, and other features are independently sampled. These exact boundaries remain hidden from the learner. Both heldout sets are raw-input-disjoint from admitted learning evidence.

## Comparators

- **Passive raw-feature CART:** same initial training labels, no active labels.
- **Matched-active raw-feature CART:** all initial plus active/audit labels supplied to the executable learner.
- **Matched-active kNN:** same evidence, normalized distance, hard scope filtering before retrieval.
- **Shared-partition tree reference:** the same partition and compiler used by the executable policy, without the promotion wrapper. This is a representation/implementation control, not an independent discovery baseline. It can and should tie a correctly promoted program.
- **Always reject:** exposes class-balance effects; its zero false accepts does not constitute useful competence.

The CART candidates are conventional sampled tree fits replayed on the available training evidence; they are not claimed to be exhaustively hyperparameter-optimized baselines. All comparator limitations and evidence budgets remain part of the result.

Optional actual provider conditions are zero-shot, tenant-only example retrieval, and frozen executable-policy context. All use the same provider model and heldout samples. Retrieval selects up to 16 examples from the same training pool; the policy is a compressed product of the full pool. Equal evidence availability does not imply equal prompt length or total compute. No cost-matched superiority is claimed. A fourth reported result is the deterministic policy-verifier output, not a fourth independent model condition.

## Metrics and ordering

Accuracy on all cases counts `UNKNOWN` as uncorrect. Coverage is the fraction answered. False accepts are actual REJECT cases predicted APPROVE. State-effect accuracy is separate and conditional on correctly accepted actions, with its denominator reported. Raw model effects are not scored by borrowing a symbolic effect learner.

Policy, effects, training evidence and predictions are frozen before result scoring. The evaluator alone opens the labels. Final file hashes and evidence-chain ordering are checked by `verify`.

Paired differences resample tenant blocks, not individual rows. These intervals describe this generated task population and one algorithm run. They do not turn eight synthetic structures into a broad real-ERP evaluation.

## Crucial novelty distinction

The high-fidelity learner queries representatives of every discovered logical region. A novel raw input can have an already observed latent condition signature. The report exposes both counts. A high score with zero unseen signatures is **not** a success on an unqueried-combination challenge.

## Failure controls

The same study includes two diagonal-boundary environments with deliberately tight 64-query/64-region limits and foreign-scope predictions. Their results include abstention and any errors without being mixed into the supported-family accuracy denominator. They test resource/binding behavior; they do not prove complete detection of out-of-grammar rules. Unit tests separately cover contradictory labels, malformed ASTs, drift, provenance, heldout query exclusion, tampered files and incomplete provider outputs.

## No moving target

Keep disclosed results as development evidence. A future run uses a new immutable directory. Do not tune on heldout errors and call a replay a new confirmation. The existing real-Dolibarr factorial protocol is preserved separately and must not be amended using this package's results.
