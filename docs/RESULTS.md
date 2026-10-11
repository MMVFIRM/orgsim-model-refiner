# v0.7 measured results

## Evidence scope

One fresh, source-frozen offline study, seed **887041**, after development fixes. **24 synthetic tenant parameter sets from eight declared formula families**. No provider calls, no live Dolibarr requests, no autonomous ERP transactions. Raw inputs, not private oracle thresholds/flags/formula names, were provided to the learner.

This is **active region reconstruction and raw-input generalization**, not a replication of the original 96-case hard-transfer study. The framework prioritizes accuracy by acquiring additional observed training outcomes.

## Main results

There were **6,144 heldout cases plus 6,144 boundary-focused stress cases**. All 24 policies met their independent calibration criteria. Each set has 3,072 approvals and 3,072 rejections. No heldout raw input was queried during learning.

| Method | Holdout accuracy | Boundary stress accuracy | Holdout false accepts | Stress false accepts |
|---|---:|---:|---:|---:|
| Passive raw-feature tree | 74.25% | 72.17% | 769 | 851 |
| Same-evidence, scope-filtered kNN | 84.70% | 76.63% | 466 | 661 |
| Raw-feature tree with the same active labels | 93.23% | 91.73% | 214 | 248 |
| Shared-partition tree reference | 100.00% | 100.00% | 0 | 0 |
| Active executable policy | 100.00% | 100.00% | 0 | 0 |

The active policy answered **12,288/12,288 correctly**, with **100% coverage**, **0 false accepts** and **0 false rejects** across the two sets. The separately learned effect model reproduced all declared synthetic state fields on **6,144/6,144 correctly accepted actions**. These are simple synthetic stock/counter/status effects, not LLM-generated state predictions or verified Dolibarr transitions.

The shared-partition reference ties exactly because it uses the same feature partition and compiler. This is an important control: **no unique superiority over an equally informed conventional executable representation has been shown**. Raw-feature CART and kNN receive the same observed labels, but not the learned interval representation; the comparison includes that representational advantage.

## Evidence expenditure and novelty

- Initial learning: 32 labels per tenant, **768** total.
- Additional active/audit simulator queries: **20,195**, mean **841.46**, range **383–1025** per tenant.
- Independent calibration: 128 per tenant, **3,072** total; promotion only.
- Dataset-generator oracle calls, including discarded balancing draws: **28,792**. These are distinct from the learner query budget.
- Exact raw-input novelty: **12,288/12,288** against admitted learning evidence.
- Evaluator-only condition-signature novelty: **177/12288 (1.44%)**. Thus **12,111/12,288** had signatures already represented in learning evidence.

That last distinction prevents a misleading compositional-generalization claim. Most examples test application to new numerical inputs within learned regions. The small newly signed subset is descriptive, not a separate preregistered unseen-combination confirmation. The original hard-transfer benchmark remains a negative textual-compilation result.

## Resource and scope controls

Two additional diagonal-boundary tenants used deliberately tight 64-query/64-cell limits. Both policies were unpromoted; all **128** local evaluation cases were `UNKNOWN`, with no observed false accepts or false rejects. Foreign-scope predictions also abstained on all **128** cases. These controls are separate from the main high-accuracy denominator. They demonstrate conservative behavior under those tested limits, **not complete detection of hidden rules or future safety**.

## Historical regression

The disclosed archive's three hashes and 288 unique case/arm pairs were reproduced. Its logged retrieval lists still show **236 same-tenant and 532 other-tenant IDs**, with 96 mixed sets. The package does not invent its missing `f1`–`f6` oracle or rescore it as a blind test. See `results/HISTORICAL_REGRESSION.json`.

## Verification

The completed study passes local source, artifact, evidence-chain, training-replay, split-disjointness and prediction-before-score checks. The full regression suite contains **288 tests**. The new model integration was tested using explicit test doubles only. Clean-package verification details are recorded in `results/RELEASE_VERIFICATION.json` after assembly.

These hashes establish internal consistency, not independent timestamps or authenticated server observations. Eight known synthetic formula families are not evidence of arbitrary organizational rules. Finite calibration samples do not prove a universal 100% true accuracy.

## Reproduce or inspect

```powershell
py -m orgsim_refiner verify --root results/active_confirmatory
py -m orgsim_refiner study --seed 910273 --out build/active_study
```

The bundled confirmation is disclosed. A new run is a separate fixed study, not another opportunity to choose favorable outcomes. Detailed per-tenant predictions, evidence, calibration, query counts and paired tenant-bootstrap comparisons are in `results/active_confirmatory/REPORT.json` and its companion artifacts.
