# Executable policy memory — study protocol

## Research question

Can a black-box learner convert synthetic OrgSim trial-and-error into exact, executable ERP workflow guards, without changing the frozen provider model's parameters? When does such knowledge beat conventional feature-engineered classifiers under equal labeled evidence?

## Scope and known inductive bias

The experiment assumes four publicly named synthetic operations: `validate_order`, `pay_invoice`, `reserve_stock`, and `close_order`. Public guards are available to all learners via `predict_generic`: state eligibility, resource sufficiency, shipping and invoicing requirements. The active learner knows which public integer fields it can perturb. It **does not** receive the target's private upper-bound numerical constants or policy manifest. Its hypothesis grammar allows an upper-bound guard, a sum upper bound, and at most one monotone two-feature veto per workflow.

This is a strong, *explicit* inductive bias. The learner probes the two/three pertinent fields using a neutral, public-admissible synthetic starting state and monotone binary search, then challenges the inferred rule with random boundary examples and out-of-pattern monotonicity checks. It trusts a program only when all audits and historical validation decisions agree. Otherwise it declines to authorize hidden-rule-dependent accepts. Public guard violations can still be rejected safely.

A separate **non-monotone adversary** changes real OrgSim workflow preconditions (not just labels in a Python wrapper). It introduces isolated veto holes; all audit failures are reported. Results from such environments remain outside the trained grammar. The adversarial oracle's source is never passed through the learning callback.

## Training, comparison, freezing

Each fresh organization is generated deterministically from a seed, with randomized private thresholds. All labels and entity-state deltas are produced by executing OrgSim's SQLite transactional workflow runtime. Negative actions must roll back. Train, router, validation, test, and shifted stress partitions are group-disjoint and 50/50 class-balanced per operation. Artificial class balance is not an estimate of a real organization's base rate.

For each organization:

1. Generate dataset partitions and record all oracle calls, including discarded draws used in balancing.
2. Fit **passive** conventional tabular learners on training examples, with selection using validation only.
3. Perform counted active black-box interventions, freeze a JSON program and collect every query's labels.
4. Fit **matched-active** conventional tabular learners on exactly the original training examples plus *all* active probe labels; use the same validation for model selection. The structured policy receives no extra labeled examples that the matched comparator lacks.
5. Fit a state-effect learner using historical train/validation before opening test cases.
6. Score original test and shifted stress *once* with per-case decision, incorrect accept, incorrect reject, abstention, state-effect and end-to-end metrics. Every prediction is on the same case identities.
7. Test new non-monotone ERP seeds separately and measure coverage and unsafe accepts. Test unbound cross-tenant transfer as a **negative control**, while the tenant-bound adapter refuses that transfer.

The suite includes local SHA-256 freezes binding the source dataset, compiled program and result files. These checks detect ordinary local modifications; they do not externally timestamp results or independently attest to the simulator. Reusing a revealed test set to tune the algorithm would invalidate claims of a new blind holdout; a fresh seed is then necessary.

## Safety and software boundaries

The policy-sidecar is external to Model Refiner 0.5rc1's immutable `SuiteArtifact`: it is a **new experimental adaptation mechanism**, not an unreviewed modification to the native fit artifact. The optional LLM comparator uses the real Model Refiner `Recipe` and `Predictor` inference classes; it is not a native `suite.fit` success. A verified synthetic policy is explicitly tenant-bound, carries synthetic-only provenance and cannot enable real ERP writes. Nonconforming rule families remain uncertified. The model's suggestion is advisory to the deterministic program adapter.

## Metrics

`accuracy_all_cases = (true accepted + true rejected) / total cases`, counting abstentions as uncorrect. `false_accept_rate = false accepts / actually rejected cases`. `end_to_end_exact_all_cases` requires a correct rejection or a correct acceptance with exact predicted changes to every affected state field; misses count as errors. `coverage` is the non-abstained fraction. Separate per-operation results expose any operation-specific regression. **Zero observed false accepts on synthetic data is not proof of zero risk in a real ERP.**
