# Real frozen-model / Model Refiner evaluation

**Actual GPT-6 performance is unknown until you execute this step.** The included `--mock` is a software integration test only; the high-scoring conventional baseline in the offline report is a separate algorithm.

## Dry run (no API key needed)

```powershell
py -m simlearn.cli refiner --data examples/simlearn_prebuilt `
  --task classification --model gpt-6
```

This prints the native Model Refiner `plan_fit` worst-case logical call and provider-attempt count, along with the exact balanced split sizes. The default limits are **32 train, 8 router, 8 validation, and 32 test per workflow** (128/32/32/128 total). The 128 test examples exceed Model Refiner's default minimum of 100; 32 router examples can fit a contextual selector but are modest for calibration. No provider calls happen in the dry-run.

The simulator-trained tabular baseline used 400 training examples *per operation* in its confirmatory experiment. **Its 99.17% score is not a matched-label-budget comparison with this 32-example model run.** A matched-data tabular comparison is calculated automatically for each real classification fit and saved alongside the native report.

## Opt-in provider fit

Configure `OPENAI_API_KEY` securely on the machine running the experiment, not inside a source file or chat upload. Install `.[openai]`. Then:

```powershell
py -m simlearn.cli refiner --data examples/simlearn_prebuilt `
  --task classification --model gpt-6 --execute-model `
  --max-attempts 2500 --max-reserved-tokens 40000000 `
  --out build/refiner_gpt6_decisions
```

Run in a fresh workspace. You can set `--optimizer-model` separately. Large reasoning-model output-token limits may be necessary for model completion; defaults here are 8,192 response tokens and at least 8,192 optimizer compile tokens. The attempt and token-reservation ceilings are **not a dollar-cost cap**. Actual billed tokens/prices depend on the provider.

The native SDK uses train examples for demonstrations and error-rule compilation, router examples to fit contextual Artificial Weights, validation to select static/default/weights recipes, and only then its disjoint test split. It saves:

- `artifact.json`: immutable, content-hashed Model Refiner recipe bank and optional Artificial Weights selector, with frozen original provider model unchanged.
- `report.json`: native test summary for baseline, retrieval-only, best static and weights/selected; confidence intervals, attempted calls, slices, release gate.
- `SIMLEARN_BRIDGE_REPORT.json`: condensed native comparisons and **matched-evidence** conventional comparator scores on exactly the same test IDs.
- `ledger.sqlite3`: Model Refiner's budget reservations and scoped response cache; keep private.

**Pass criteria** require the native quality gate to pass, improved selected accuracy over frozen baseline and retrieval, sound coverage, and no unacceptable increase in incorrect accepts. The native gate's threshold settings are not permission to execute real enterprise mutations.

## Exact-state-effects model task

Run this only after assessing decision-task results; the JSON task can require more output tokens and be more expensive:

```powershell
py -m simlearn.cli refiner --data examples/simlearn_prebuilt `
  --task effects --model gpt-6 --execute-model `
  --out build/refiner_gpt6_effects
```

The output contract is `{"decision":"ACCEPT|REJECT","changes":{...}}`, with exact changed-field names and resulting values, compared to OrgSim's executed outcomes. It is **not** a response-shape-only task. A wrong acceptance or incorrect numeric effect counts as incorrect. A model trained on an incorrect simulator can still perform very well on that simulator and badly on real Dolibarr; this experiment does not eliminate sim-to-real error.

If either model run fails the native gate, keep its artifact staged and treat this as a **negative experimental result**, not as a successful training method.
