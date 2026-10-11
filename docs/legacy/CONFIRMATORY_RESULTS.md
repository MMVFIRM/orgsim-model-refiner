# Confirmatory offline results — 2026-10-10

Reproduced with `simlearn full --environments 3 --seed 20261011 --train 400 --router 100 --validation 100 --test 200 --stress 200` on the packaged code. Test and stress contain **2,400 cases each** (3 distinct organizations × 4 workflows × 200 cases). All reported examples are synthetic.

| Predictor | Test accuracy | Stress accuracy | Test incorrect accepts | Stress incorrect accepts |
|---|---:|---:|---:|---:|
| Always reject | 50.00% | 50.00% | 0 | 0 |
| Public guards | 81.13% | 77.25% | 453 | 546 |
| Model Refiner TF-IDF memory, nearest-neighbor vote | 66.75% | 64.21% | 423 | 493 |
| **Conventional simulation-trained classifier** | **99.17%** | **99.33%** | **16** | **11** |

The simulation-trained model had 4 incorrect rejects on test and 5 on stress. Exact changed-field prediction was 100% **on its true accepted predictions**, and the end-to-end exact result equals decision accuracy on this bounded additive-effect grammar. Effect induction was audited on independent validation positives in all twelve operation × organization pairs.

### Per-organization accuracy

| Synthetic environment | Test accuracy | Shifted stress accuracy |
|---|---:|---:|
| 1 | 99.125% | 99.00% |
| 2 | 98.50% | 99.125% |
| 3 | 99.875% | 99.875% |

All three organizations individually exceed 98% on test. Across the balanced test there were 1,200 rejected ground-truth cases; the 16 incorrect accepts correspond to a **1.33% false-accept rate** on those cases. That is not a safe real-world autonomous-execution rate.

The entire suite used **16,889 oracle requests**, including additional balancing/discarded draws, to record 12,000 final labeled episodes across all five partitions. Model candidates are conventional CPU classifiers trained on a subset of those episodes. No genuine GPT-6 calls were made in this confirmatory comparison.

### Native Model Refiner SDK smoke

A separate offline `--mock` execution invoked the actual Model Refiner 0.5.0rc1 SDK; it produced a valid immutable artifact and budget/audit report, exercised 281 mock-provider calls in the small test, and selected a static recipe rather than Artificial Weights. The SDK release gate **failed** `live_provider` (and, in that undersized smoke, minimum test-set size). This is correct behavior. It is not evidence that a real LLM can achieve the 99.17% result.

### What remains unproven

Model Refiner's actual adapter gain over prompting or retrieval on a real frozen model; out-of-grammar causal generalization; another tenant's hidden policy; sim-to-real transfer; and safe authorization for live Dolibarr writes. The model-provider run is provided as a separate **opt-in action within the same integrated experiment**, not as a series of invented passed gates.

Each report/freeze pair with SHA-256 checks is stored in `results/confirmatory`. Use the full offline command to regenerate trajectory inputs and verify those hashes in a new output directory.
