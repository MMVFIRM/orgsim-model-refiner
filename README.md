# OrgSim × Model Refiner v0.9.0 — hardened Dolibarr release candidate

**Start with [DELIVERY_README.md](DELIVERY_README.md), [SECURITY.md](SECURITY.md), and [docs/PRODUCTION_DEPLOYMENT.md](docs/PRODUCTION_DEPLOYMENT.md).**

This is a *complete v0.8-derived integrated research framework* with a dedicated
**production GET-only observer** and a separately gated **disposable-clone
learning oracle**. It is **not certified for unattended production operations**.
No historical benchmark score authorizes production writes.

### Quick start (Windows)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe tools/verify_delivery.py
.\.venv\Scripts\python.exe tools/test_delivery.py
```

Read-only production observation, with a *separate server-enforced read-only API key*:

```powershell
$env:DOLIBARR_READ_KEY = "<read-only-key>"  # configure privately; never commit
.\.venv\Scripts\python.exe tools/dolibarr_observe.py `
  --url https://your-erp.example.com --allow-remote-read `
  --family orders --id 123
```

This prints a digest and minimal metadata, **not private ERP records**. Use `--out`
only for a private local evidence path. The observer does not support write methods.
For active probing, use **only a disposable local clone**, the expiring permit and
unique on-clone product sentinel documented in [clone instructions](docs/CLONE_LEARNING.md).

### Historical evidence (v0.8 only)

The bundled v0.8 Dolibarr experiments reported 2,400/2,400 correct decision labels
in one lab task and 900/900 decision-plus-scored-state cases. On the latter,
703 rejected cases were previously scored using an *assumed* unchanged post-state
rather than observed post-action state. The hardened v0.9 scorer fixes this but
those historical numbers have **not been rerun or independently verified**.
See [docs/ERP_LEARNING.md](docs/ERP_LEARNING.md).

---

## Preserved earlier framework documentation

# OrgSim × Model Refiner v0.8.0
## Learns and replicates real ERP actions (oblique executable policies + live-Dolibarr oracle)

**New in v0.8:** the learner now reconstructs rules real ERPs actually use — field-vs-field
comparisons, summed lines, products — which v0.7 could not (it produced no policy on a real
Dolibarr task). On an isolated **real Dolibarr 24.0.1**, sealed held-out orders were labelled
by the ERP only after the policy was frozen:

* **2,400 / 2,400** decisions correct, 0 false accepts (4 seeds, ~1,100 learning queries each);
* **900 / 900** actions replicated **exactly** — decision *and* resulting stock/status (3 seeds);
* a drift monitor that flagged real configuration changes (26–34 % mismatch) and cleared after restore.

Read **[docs/ERP_LEARNING.md](docs/ERP_LEARNING.md)** first — it includes what failed on the way
and the limits (one tenant/operation/configuration; schema must declare all relevant fields;
concurrent labelling corrupts results on Dolibarr; disposable tenants only). Lab setup: `lab/README.md`.

---

## v0.7.0 notes (retained)
## Integrated active executable-policy framework

A complete research package for turning **observed environment outcomes** into persistent, scoped executable policies. The package includes the OrgSim runtime, Model Refiner source, prior Dolibarr research tools, and the new `orgsim_refiner` integration.

**What changed:** the primary learner is no longer an English rule summarizer. It maintains executable hypotheses, rejects contradictions against all learning evidence, asks additional training queries where its representation is unresolved, learns numerical boundaries, and freezes a policy before held-out scoring. A frozen language model can consume that policy through Model Refiner's actual `Recipe` and `Predictor` classes. The deterministic policy result and raw model output remain separate.

**Accuracy has a price:** the high-fidelity mode observes representatives of every discovered logical region. New held-out raw inputs can therefore share learned condition combinations. Results are **empirical interpolation within learned regions**, not proof of unseen-combination transfer, arbitrary ERP reconstruction, or Model Refiner superiority. See [measured results](docs/RESULTS.md).

## Start here

Use a **new directory**. This is not an overlay for an in-progress Dolibarr study.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe tools/verify_release.py
.\.venv\Scripts\python.exe tools/run_tests.py

# Zero model calls and zero Dolibarr calls.
.\.venv\Scripts\python.exe -m orgsim_refiner plan
.\.venv\Scripts\python.exe -m orgsim_refiner study --seed 910273 --out build/active_study
.\.venv\Scripts\python.exe -m orgsim_refiner verify --root build/active_study
```

The defaults run one study with 24 synthetic tenants, eight rule families, balanced training/calibration/evaluation, matched-evidence comparators, and resource-limit controls. Internal learning refinement happens automatically, within a fixed query budget; there is no sequence of new human-run gates. Output directories are immutable: never reuse a disclosed result directory for a new claim.

## Included capabilities

| Component | Implementation |
|---|---|
| Typed executable policies | Boolean, branching, membership, counting and arithmetic AST interpreter; no `eval` or generated code execution |
| Active environment learning | Raw integer/enum/bool inputs; counterexample boundary scans, candidate-disagreement ranking, finite region exploration, independent training audits |
| Evidence certification | Replay all admitted observations; contradictory evidence quarantines a scope; validation is promotion-only |
| Tenant isolation | Hard filter on tenant **and** operation **and** configuration before retrieval; no cross-tenant fallback |
| Persistent external adaptation | Hashed policy artifacts and SQLite append-only evidence ledger; advisory and calibrated inference separated |
| State effects and simulator | Separately learned empirical effects, pure in-process predicted simulator, no ERP authorization |
| Model Refiner bridge | Optional three-arm real-provider comparison, raw predictions versus deterministic verification, attempt ceilings, incomplete-output accounting |
| Observed-outcome inbox | Pause for an external simulator/operator answer and resume deterministically without editing earlier outcomes |
| Dolibarr compatibility | Existing v0.6.1 protocol retained; saved receipt/diagnostic checker; no automated transactional writes |
| One-study evaluation | Frozen inputs/source, prediction commitments, shared cases, scope-safe kNN, matched-data trees, boundary stress and failure controls |

## Commands

`plan` previews budgets. `study` trains and evaluates once. `verify` checks the completed study. `predict` executes a frozen policy. `init`, `learn`, and `submit` implement the offline observation inbox. `propose` optionally asks a model for a training-consistent AST. `audit-history` checks the disclosed hard-transfer archive without inventing its missing oracle. `dolibarr-check` and `dolibarr-status` inspect saved evidence/workspaces.

## Actual model comparison (explicit opt-in)

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[openai]"
.\.venv\Scripts\python.exe -m orgsim_refiner plan --execute-model --model gpt-6-luna
```

This preview still sends **no requests**. The default model comparison uses eight held-out cases per tenant, three prompt conditions and at most 576 provider attempts. The model identifier is forwarded to your provider account; availability is not asserted here. Set `OPENAI_API_KEY` privately, then use the execution command in [the model guide](docs/MODEL_REFINER_INTEGRATION.md). No dollar price is assumed.

## Documentation

- [Architecture and boundaries](docs/ARCHITECTURE.md)
- [One-study method](docs/METHOD.md)
- [Measured results](docs/RESULTS.md)
- [Windows quickstart](docs/WINDOWS_QUICKSTART.md)
- [Model Refiner integration](docs/MODEL_REFINER_INTEGRATION.md)
- [SDK and observed-outcome inbox](docs/SDK.md)
- [Existing Dolibarr study compatibility](docs/DOLIBARR_COMPATIBILITY.md)
- [Security and provenance](SECURITY.md)
- [Release changes](CHANGELOG.md)

All historical documents are in `docs/legacy`. Their old release claims are historical, not evidence for v0.7. The new benchmark output is `results/active_confirmatory`; previous bundled synthetic results remain separately named.

## Limits

The new learner needs a declared bounded schema containing relevant features, a trustworthy stationary observation source, and enough queries. It does not automatically discover every relevant enterprise variable, reproduce hidden external services, authenticate operator receipts, or prove correctness of unobserved effects. Region counts can grow exponentially. A rare hidden condition can evade finite audits. Non-promoted or mismatched scopes return `UNKNOWN`; that is not a business-rule rejection.

**No new real-provider or live Dolibarr efficacy is claimed by the included offline results.** Neural weights are unchanged. This is not a production authorization system or an independently certified digital twin.
