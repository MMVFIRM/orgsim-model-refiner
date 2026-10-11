# OrgSim × Model Refiner — Dolibarr Factorial Transfer v0.6.1

A **single preregistered, full-factorial, same-status, sim-to-real research experiment** for a **disposable** Dolibarr 24.0.1 tenant. This release directly follows the successful v0.5 sale-disabled-product observation, but deliberately demands a second, independently verified condition before committing a new evaluation.

### What is new

The v0.6 runner `simlearn.dolibarr_factorial` learns an **executable two-factor policy** from manual, actual Dolibarr intervention outcomes. Candidate factors are a product's on-sale eligibility and sufficient physical stock at order validation. The second factor is **configuration-dependent**: run the three-case feasibility pilot and stop if it is not enforced. No previous synthetic threshold is treated as real authority. Neither training, validation, nor the pilot includes the `00` double-failure combination; it is genuinely withheld until the four-cell holdout.

The full study freezes **6 training blocks (18 cases), 3 validation blocks (9 cases), and 16 four-condition holdout blocks (64 cases)**. All 91 main orders/products plus three pilot orders/products must be fresh, independent and disposable. Assignment is seeded, randomized within blocks, and fixed in a bound CSV. A business-rule refusal needs an actual writer 500 receipt, a distinct case-specific response file, a *specific reason in either the HTTP response or the local diagnostic*, a separate local corroborating rollback/attempt record when only the HTTP body supplies the reason, and a GET-confirmed unchanged before/after state. The read-only client performs GETs only.

It evaluates the learned executable program and transparent factor baselines. An **optional, consented Model Refiner Recipe/Predictor inference** compares frozen GPT-6 Luna, retrieval of the same real pre-holdout evidence, pseudo-labeled simulated rehearsal, and the compiled program. At most 256 logical model calls are planned for the 64 holdout cases; no model calls occur unless `--execute-model --share-sanitized-fields` is explicitly specified. The provider learns no model weights and `suite.fit()` is not run.

### v0.6.1 evidence fix — resume the existing feasibility pilot

Dolibarr can put the **insufficient-stock/negative-stock-disallowed reason** in the HTTP 500 response while writing only a stock-movement attempt and database rollback to its local log. v0.6 rejected that valid evidence configuration. v0.6.1 accepts it as **operator-correlated** evidence, not an independently authenticated receipt. A bare 500, generic rollback, or log without stock-movement evidence still fails. The original pre-registration and scoring are unchanged.

If you already ran `pilot-start` and your manual pilot actions, you can replace just `src/simlearn/dolibarr_factorial.py` in the existing installed project and rerun `pilot-finish` against its existing `build/factorial_pilot_001` folder. **Do not execute the original write actions a second time.** Run `py -m simlearn.dolibarr_factorial check-evidence --help` for a read-only offline preview.

### Start here

[One-shot PowerShell operator protocol](docs/ONE_SHOT_DOLIBARR_FACTORIAL_PROTOCOL.md)

[Pre-registered scoring and stopping criteria](docs/FACTORIAL_READOUT_CRITERIA.md)

The full package retains the original OrgSim deterministic simulator, Model Refiner v0.5.0rc1 integration, inherited real-Dolibarr controls, other research demos, and new tests. It can be installed offline from source when existing dependencies are installed:

```powershell
py -m pip install -e ".[test]"
py -m pytest -q tests/test_dolibarr_factorial.py
py -m simlearn.dolibarr_factorial pilot-template --out build/factorial_pilot_IDS.csv
```

To see all commands:

```powershell
py -m simlearn.dolibarr_factorial --help
```

### Definition of success

A perfect answer on a single known rule is insufficient. Our locked **incremental adaptation** criterion requires the compiled-policy GPT-6 condition to exceed **ordinary GPT-6 + real retrieval by at least ten accuracy percentage points**, with a strictly positive 95% *block*-bootstrap paired interval, at least 95% scored accuracy and coverage and zero false accepts. The pre-registered direct policy learner has a separate 95%/zero-false-accept decision test. The transparent conjunction is also evaluated; if it ties at 100%, it demonstrates that a simple non-LLM rule may be equally effective.

A failed second-rule pilot or an unqualified real holdout is a valid negative result; do not re-label errors or create extra gates to improve the score. Manual evidence is not an independently certified transaction audit, and this single-tenant bounded-rule test does not establish arbitrary ERP fidelity. No production credentials or real ERP write authorization should be inferred.

### Package verification

`RELEASE_MANIFEST.json` contains file-level SHA-256 digests. The experiment's own local event chain and public report freeze are verified separately. See the operator guide for the public artifacts to share; raw server responses, logs, private IDs, keys and records remain local.
