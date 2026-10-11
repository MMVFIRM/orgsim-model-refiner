# v0.8 — learning and replicating real ERP actions

v0.7 could only reconstruct **axis-aligned** rules ("credit <= 17"). Real ERP rules compare
fields with each other (`stock >= qty`), add lines (`qty1 + qty2 <= stock`) and multiply
(`outstanding + price*qty <= limit`). On a real Dolibarr order-validation task v0.7 produced
**no policy at all** (it exhausted its region budget after 2,266 queries). v0.8 adds:

| Component | What it does |
|---|---|
| `oblique.py` | Exact small-integer **oblique decision trees**. Numeric splits are *fitted from the data* (local linear separator over raw + pairwise-product features, rounded to small integers, accepted only if it still separates the node's data) and drawn from a column bank. Several build strategies run (impurity at several simplicity tolerances, plus a separate-and-conquer "largest pure group" mode); the **smallest consistent tree wins** (Occam / description length). |
| `expr_learner.py` | Counterexample-guided loop: probe the exact integer boundary of every split (one step either side), probe committee-disagreement points, audit at random, add every counterexample, refit; stop only after several consecutive rounds find none. Training queries never touch reserved evaluation inputs. |
| `effects.py` | The effect model now learns `before ± (sum of up to three action fields)` and single-field conditionals ("stock decrements for goods, is unchanged for services"). |
| `erp/dolibarr_live.py` | Oracle for a **disposable local** Dolibarr (loopback only). Labels come from the HTTP receipt **and** a read-back of the order status. |
| `tools/erp_experiment.py` | Learn → freeze → *then* fetch held-out labels from the ERP → score. |
| `tools/erp_replicate.py` | Predict decision **and exact resulting stock/status**, perform the order, compare. |
| `tools/erp_drift_check.py` | Re-verify a frozen policy against the ERP; exit code 3 = stop using it. |
| `tools/erp_rule_suite.py` | Five further ERP-style rule families on a fast twin. |

## Measured on a real Dolibarr 24.0.1 (isolated lab container)

Task: validate a sales order with a primary product (on-sale flag, good/service, stock, up to two
lines of it) and an optional second product — 8 input fields. Held-out sets are drawn **before**
learning (hashes written first), excluded from every query, and labelled by the real ERP only after
the policy is frozen: 300 uniform + 300 boundary-focused orders per seed.

| Experiment | Seeds | Held-out result | False accepts | Learning queries (mean) |
|---|---|---|---|---|
| Decision accuracy (`erp_experiment`) | 301–304 | **2,400 / 2,400** | 0 | 1,124 (817–1,363) |
| Action replication: decision **and** exact post-state (`erp_replicate`) | 401–403 | **900 / 900** | 0 | 1,329 (1,243–1,419) |
| Drift monitor, unchanged ERP | — | 0 / 80 mismatches (consistent) | 0 | — |
| Drift monitor, stock check switched off | — | 21 / 80 mismatches (**drift flagged**) | 0 | — |
| Drift monitor, on-sale check bypassed | — | 27 / 80 mismatches (**drift flagged**) | 0 | — |
| Drift monitor, config restored | — | 0 / 80 (consistent again) | 0 | — |

Learning takes 3–6 minutes per policy against the ERP. Raw reports, frozen policies and frozen
predictions are in `results/erp_live_v08/`. `development/` holds the earlier learner generations
for transparency (see "What failed on the way").

Real Dolibarr, default configuration, enforces **only two** order-validation rules: the product
must be on sale, and goods need `stock >= total quantity of that product across the order's
lines` (services are exempt). Credit limits, minimum price and closed customers were probed and
are *not* enforced. The learner recovered exactly that, including the cross-line sum.

### Other rule families (fast twin, `tools/erp_rule_suite.py`)

Credit limit with a product term, a coefficient-2 ratio cap, a four-term line sum, and tier
branches converge to **100 %** held-out accuracy (≈400–2,200 queries). An **integer-rounding
discount rule** (`price - price*discount//100 >= min`) converged in 1 of 3 seeds and otherwise
**failed closed** (not promoted, answers UNKNOWN, never a guess).

## Findings about real ERPs that matter for anyone running this

1. **Concurrency corrupts labels.** Serially, the same input returned the same answer 12/12
   times. With four parallel clients the *same* input was spuriously rejected "not enough stock"
   in up to 50 % of repeats. The live oracle therefore serialises whole transactions by default.
   Never label from concurrent clients without proving determinism first.
2. **HTTP 200 is not an approval.** About 1 in 400 validate calls under load returned 200 while
   the order stayed a draft. A verdict requires *both* the right receipt and the matching read-back
   state; anything else is retried or refused, never labelled.
3. **A policy belongs to a configuration.** Flipping one ERP setting made 26–34 % of re-checked
   orders disagree. Re-run `erp_drift_check` before relying on a stored policy.

## Run it against your own disposable ERP

```powershell
py -m venv .venv; .\.venv\Scripts\python.exe -m pip install -e ".[test]"
# 1. start a disposable lab (see lab/README.md), put DOLAPIKEY in the environment
$env:DOLAPIKEY = "<lab key>"
# 2. learn, freeze, and score on sealed held-out orders (≈5 min)
.\.venv\Scripts\python.exe tools/erp_experiment.py --oracle live --seed 1 --budget 4000 --out build/erp_live_1
# 3. replicate the action (decision + exact stock/status)
.\.venv\Scripts\python.exe tools/erp_replicate.py --oracle live --seed 1 --out build/rep_live_1
# 4. later: has the ERP drifted?
.\.venv\Scripts\python.exe tools/erp_drift_check.py --policy build/erp_live_1/POLICY.json --out build/drift_1
```

To learn a *different* operation, define its `Schema` and a callback returning an `Observation`
(see `erp/orders.py` and `erp/dolibarr_live.py` as the template). Everything else is reused.

## What failed on the way (kept visible)

* **v0.7 learner:** no policy on this task (region explosion).
* **Enumerated form bank (gen 1):** worked on this task (2,400/2,400 live over seeds 102–105) but
  could not express credit-limit, coefficient-2, or 4-term rules (no convergence in 6,000 queries).
* **First oblique trees (gen 2, plus an intermediate build):** general but unstable — spurious
  high-coefficient forms and 1.2–2× the queries. On the twin, one seed failed to converge and one
  intermediate build was promoted with 6 false accepts. Live, gen 2 still scored 600/600 on seeds
  201 and 202 (1,676 and 3,111 queries); that batch was stopped because the improved build
  superseded it. The false-accept case is why the final build uses Occam selection and a
  separate-and-conquer mode.
* **Current build (gen 3):** Occam selection over several build modes. All claims above are from this build.
* Early live attempts with parallel clients produced corrupted labels; that is finding 1.

## Limits — read before trusting this on your ERP

* **Scope of the evidence:** one tenant, one operation, one configuration, an 8-field schema, and a
  rule class (thresholds over small-integer sums/differences/products plus categorical branches).
  100 % on 3,300 sealed cases is strong for *this* task, not a guarantee for another ERP or rule.
* **The schema must contain every relevant feature.** A hidden driver (a user right, a custom
  trigger, time of day) shows up as non-convergence or failed calibration — the learner fails
  closed — but it cannot discover variables you did not declare.
* **Deterministic, stationary oracle assumed.** Label noise and drift are caught, not corrected.
* **Finite audit, not a proof.** "Counterexample-free rounds" and boundary probes are empirical.
  A rare exception far from every boundary can evade them.
* **Integer-rounding / non-linear arithmetic** can exceed the learner (see above).
* **Disposable tenants only.** The live oracle creates real records and refuses non-loopback
  hosts. Nothing here authorises writes to a production ERP.
