# Confirmatory results — frozen seed 20261237

The frozen machine-readable source of truth is [`../results/structured_confirmatory/STUDY_SUMMARY.json`](../results/structured_confirmatory/STUDY_SUMMARY.json), with its own [`FINAL_FREEZE.json`](../results/structured_confirmatory/FINAL_FREEZE.json). Hash integrity checked locally; no external timestamp or independently supplied ERP oracle was involved.

Five fresh, independently randomized *synthetic OrgSim* environments, four operations each. Each organization: 128 balanced training examples/operation; 32 router; 32 validation; 128 held-out test; 128 shifted stress. 12,850 simulator executions to create the training/evaluation datasets, plus **1,452 active black-box policy-induction queries** (matched-active tabular learner was given the same additional labels).

| Architecture | Test correct | Test accuracy | Stress correct | Stress accuracy | Incorrect accepts, test / stress |
|---|---:|---:|---:|---:|---:|
| Public guards | 2,068/2,560 | 80.7813% | 1,992/2,560 | 77.8125% | 492 / 568 |
| Passive tabular | 2,498/2,560 | 97.5781% | 2,492/2,560 | 97.3438% | 53 / 54 |
| Matched-active tabular | 2,512/2,560 | 98.1250% | 2,517/2,560 | 98.3203% | 38 / 35 |
| **Executable program memory** | **2,560/2,560** | **100.0000%** | **2,560/2,560** | **100.0000%** | **0 / 0** |

The executable policy also correctly reproduced state effects on accepted actions in the paired holdouts. The full set of candidate fields and all hidden rules were not demonstrated from a real Dolibarr version; this is recovery within a compact, predeclared rule grammar. There is no provider-LLM inference in the confirmatory result.

## Adversarial out-of-class policies

Three independent non-monotone synthetic environments, four operations, 128 cases per operation = **1,536 cases**. All 12 per-operation rule audits rejected the learned monotone program. The adapter answered only public-guard rejections: **543/1,536 answered, 35.35% coverage**. All these answered cases were correct, with **zero incorrect accepts**. This is conservative abstention, *not* 100% out-of-class decision accuracy.

## Cross-tenant reuse negative control

Four hypothetical transfers each scored on 512 *other-tenant* test inputs. An improperly unbound prior-tenant policy achieved 87.11–92.19% accuracy and between 12 and 23 **incorrect accepts** per transfer. Therefore the actual adapter checks synthetic tenant fingerprint and abstains on all such cross-tenant requests. Adaptation must be performed for each new organization. A real production organization identifier needs authenticated tenant scoping beyond this research hash.

## Historical real LLM findings — not directly comparable

The immediately preceding, distinct Model Refiner 0.5rc1 real GPT-6 Luna experiment used 128 synthetic classification test cases from a smaller per-operation training split. The native selected static recipe scored **69.53%**, retrieval **68.75%**, frozen baseline **65.63%** and separate Artificial Weights **75.00%**. The native quality gate failed. That run is **historical evidence only**; it was not rerun by the v0.2 confirmatory protocol and therefore cannot be treated as a matched baseline for the 100% executable-program result. The optional native `Predictor`/`Recipe` comparison in this release must be explicitly executed with provider credentials to measure whether a frozen LLM itself uses the learned structured rules accurately.

## Major limitations

- Hypothesis class and active intervention axes are supplied and closely fit the generator's synthetic policy family. This is not arbitrary software reverse engineering.
- Public preconditions are coded; the learned program discovers hidden thresholds/interactions rather than all rules from zero.
- A same-family holdout tests novel samples and numerical policy settings, not independent software or real ERP workflow fidelity.
- The balanced data distribution differs from real enterprise traffic.
- Out-of-class cases requiring hidden-rule reasoning were all abstained, so useful coverage remains unsolved there.
- Exact state effects were learned within the small declared affine-effect family.
- No real ERP mutations or OpenAI API calls were made in the frozen confirmatory experiment.
