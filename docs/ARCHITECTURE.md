# Architecture

## Two distinct learning problems

OrgSim models what an environment does. The external policy memory tells a frozen model what observed rules it should apply. Neither should silently borrow ground truth from the other: a simulated trajectory predicted by a learned policy is not new authoritative evidence about the real environment that policy is trying to reconstruct.

The v0.7 path is:

```
Observed outcomes + explicit public schema
                 |
Scope-bound evidence ledger and contradiction checks
                 |
Executable candidate bank -> active training queries
                 |                    |
Counterexample boundary refinement <-+
                 |
Complete learned-region table -> compact typed policy AST
                 |
Independent calibration (promotion only, never repair)
                 |
Frozen policy + separate state-effect model
          /                      Deterministic simulator     Model Refiner Recipe/Predictor
          \                      /
        Shared held-out evaluation and provenance receipts
```

## Representation

`Schema` accepts bounded integers, explicit enums and Boolean fields. Store money in declared integer minor units and other measurements in declared integer units; do not silently round floating-point inputs. The AST interpreter supports comparisons, arithmetic, Boolean operators, membership, threshold counts, and conditional branches. Programs are data, not Python source.

The active learner's main discovery representation is a finite axis-aligned partition, not an arbitrary arithmetic theorem prover. A sampled committee of ordinary decision trees ranks disagreement queries. The committee is **not a complete version space** and its consensus is not a formal certificate. Optional model-proposed ASTs are admitted only if they fit every provided learning example; their agreement still does not prove generalization.

## Learning

When opposite labels appear in the same proposed region, the learner walks between the two inputs, identifies a coordinate with an observed label change, and scans its declared one-dimensional domain. This can reveal multiple boundaries and isolated intervals; monotonicity is not assumed. It explores representatives of discovered regions, replays all observations, and uses a separate random training audit to expose false equivalences. Additional evidence may cause internal refinement. The entire procedure is bounded by query, region and refinement limits.

The compact compiler learns splits in **region indices**, then translates them back to the exact learned raw cutpoints. Ordinary midpoint splits between region representatives are not sufficient: they can place a decision boundary in the wrong location.

## Promotion and uncertainty

Independent labeled calibration is used only to promote or decline the completed artifact. Default requirements are at least 32 calibration examples, at least four in each class, at least 99% all-case accuracy, at least 95% coverage, zero observed false accepts, and a completed training audit. Failure returns an unpromoted artifact. There is no repair using validation labels and no holdout feedback to training.

These are empirical criteria, not statistical guarantees of 99% future accuracy. `UNKNOWN` means unsupported inference. It must never be relabeled as `REJECT` merely to increase safety or accuracy statistics.

## Persistence and isolation

Every policy and evidence row has a `tenant`, `operation`, and `configuration` identity. The retrieval engine filters all three before ranking. Native legacy Model Refiner APIs are retained unchanged; the isolation guarantee applies to the new integration path, not to arbitrary direct calls into a historical global memory index.

Artifacts and evidence use canonical hashes. SQLite triggers prevent ordinary update/delete through the ledger interface and a chained digest detects inconsistent edits. An owner with file-system control can replace files and recompute hashes: these are local integrity mechanisms, not independently authenticated records.

## State effects

`EffectModel` searches simple constant, unchanged, increment and additive/subtractive expressions on observed before-state and action fields. It retains consistent candidates and refuses a prediction when they disagree on a new input. It models only declared observed fields; unknown external side effects are not certified. Raw LLM decisions do not inherit effect accuracy from this component.

`PolicySimulator` is a pure in-process simulation using the frozen decision and effects. Its generated trajectories carry `MODEL_PSEUDOLABEL` provenance. They may be rehearsal material, but cannot be admitted back as observed truth about the same unknown environment.

## Deliberate limits

High dimensionality, huge numeric domains, curved boundaries, stochastic labels, omitted variables, policy drift, asynchronous external effects, and sparse rare exceptions can defeat the learner or exceed its budget. In particular, high accuracy after exploring discovered regions does not establish unseen logical-combination transfer. The benchmark reports exact raw-input novelty and evaluator-only logical-signature novelty separately.
