# Gate 2 — No-live-test preflight

As of OrgSim **v0.2.0**, no additional call to the real Dolibarr 24.0.1 tenant has been made. This release fixes known **simulator expressivity and evaluation harness limitations**, not undiscovered Dolibarr rules.

## Current benchmark evidence

- Previous Gate 1C, on the local disposable tenant: 249 documented operations, 38 conservatively modeled by the Gate 1D frozen candidate (the previous generic importer modeled 45), 2 development discovery probes with aligned list counts and selected synthetic value checks, **0/2 full response matches**.
- v0.2.0 offline catalog: **249 operations classified**. **69** are classified as special workflow/relationship proposals within orders, invoices, products and third parties, of which 9 are named lifecycle transitions across the complete catalog. **None is marked executable in the generated blueprint.**
- v0.2.0 candidate: Gate 1D manifest plus three explicitly **unverified** alternate-key GET adapters (`product_by_ref`, `product_by_barcode`, `thirdparty_by_email`), 41 total simulator tools. This does not establish 41 correct Dolibarr behaviors.
- `preflight_offline/probe_priorities.json`: 18 **proposed, unexecuted, read-only** probes with fixture prerequisites. Treat as a menu for human review, not a script to fire automatically.
- v0.2.0 engine synthetic regression tests and original demonstration scenarios must pass before returning to the live sandbox.

## Offline steps (PowerShell)

Extract the ZIP and run from its top-level engine folder:

```powershell
py -m pip install -e ".[dev]"
py -m pytest -q
py examples/franzen_workflow_lab_run.py
py examples/demo_run.py
py -m orgsim.cli compile gate_dolibarr/gate2_reconstruction/model_candidate_v020_unreviewed.json --dest build/gate2_candidate --allow-draft
```

To recompute the offline classification from your **local sanitized Swagger export** (leave this file on your machine):

```powershell
py -m orgsim.franzen_gate `
  --swagger build/dolibarr/v013_sanitized_live_swagger.json `
  --manifest gate_dolibarr/gate2_reconstruction/model_candidate_v020_unreviewed.json `
  --discovery gate_dolibarr/gate2_reconstruction/discovery_summary.json `
  --out build/dolibarr/gate2_offline_preflight
```

The output contains `catalog.json`, `workflow_hypotheses.json`, `probe_priorities.json`, `model_reconstruction_brief.json`, and `FREEZE.json`. These do not contain raw observed customer field values and cannot on their own establish ERP fidelity.

## Gate 2 validation criteria before another live test

- A model-produced workflow rule must compile and execute in the new declarative action DSL.
- At least one **synthetic** positive and negative example per proposed transition should be tested, with no partial state on rejection.
- Predictions must include status outcome, selected returned business values, state deltas and permissions; lexical endpoint coverage alone is not a score.
- Model and baseline receive the same permitted discovery evidence and computational/test budget.
- Live write probes require a resettable local tenant, explicit permissions and per-case reset or isolated clones. Final `orgsim_gate` rights stay read-only when not actively testing.
- **Do not rerun old holdouts as new blind evidence.** Use new holdouts after freezing a candidate, and keep them unavailable to the reconstruction model.

## Important caveat

The new workflow DSL is capable of representing actions such as `orders.validate`, but neither Swagger nor two list probes specify the true Dolibarr validation preconditions, side effects or state representation. **It would be dishonest to claim those unknowns have been fixed without gathering more permitted behavioral evidence.** The revised probe planner is intended to make each later observation informative rather than costly guesswork.
