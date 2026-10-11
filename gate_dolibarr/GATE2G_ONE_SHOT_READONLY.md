# Gate 2G — One-shot Dolibarr read-only operator guide

This is the **single consolidated** post-Gate-2F live read test. It tests new query combinations and previously unexplored relationships with a stronger conditional learner and corrected scoring. **No live writes are supported.** It is not a business-workflow test.

## Prerequisites

- Disposable local Dolibarr sandbox, using the same read-only `orgsim_gate` role as earlier discoveries. Do not use production records or an admin key.
- Some synthetic third parties, orders, an invoice and a product remain populated. Empty/missing resources reduce cases scored; do not count skips as passes.
- The source report in `gate_dolibarr/gate2f_inputs/gate2d_discovery_report.json` and the **already disclosed** Gate 2F development report `gate_dolibarr/gate2g_inputs/gate2f_development_results.json` are the only behavioral evidence used in the prepared comparison. Their hashes are in the freeze.
- In Windows PowerShell, use a new output folder for every evaluation. Do not run the disclosed Gate 2F test again and call it blind.

## 1. Verify package and preview, **zero ERP traffic**

```powershell
py -m pip install -e ".[dev]"
py -m pytest -q
py -m orgsim.gate2g_readonly evaluate `
  --frozen gate_dolibarr/gate2g_seed_frozen `
  --out build/gate2g_preview
```

The preview does not require a Dolibarr key. The supplied `gate2g_seed_frozen` contains the three fully frozen local arms: `contract`, `symbolic`, `conditional`. They all target the **same 24 read-only query cases**. The `conditional` and naive `symbolic` arms have access to identical disclosed discovery and development evidence. The contract baseline sees Swagger only.

## 2. Optional: freeze the genuine GPT-6 arm FIRST (before ANY Gate 2G live response)

If you want to evaluate the LLM, configure your personal `OPENAI_API_KEY` locally, then run:

```powershell
py -m orgsim.gate2g_readonly prepare `
  --discovery gate_dolibarr/gate2f_inputs/gate2d_discovery_report.json `
  --swagger gate_dolibarr/gate2f_inputs/sanitized_live_swagger.json `
  --development gate_dolibarr/gate2g_inputs/gate2f_development_results.json `
  --execute-model --model gpt-6 `
  --out build/gate2g_model_frozen
```

Use **`--frozen build/gate2g_model_frozen` consistently** in the following steps if you did this. Otherwise use the supplied `gate_dolibarr/gate2g_seed_frozen` for the three local arms. The provider request contains previously disclosed, sanitized schema/status/count metadata and new query templates; it does not contain customer values or API identifiers. Model integration is mock-tested only and may incur API cost. It is one predictor-generation call, **not** interactive LLM exploration.

## 3. One bounded Dolibarr GET-only evaluation

Set the `DOLIBARR_API_KEY` securely in your **local environment**, never in the command line. Then:

```powershell
py -m orgsim.gate2g_readonly evaluate `
  --frozen gate_dolibarr/gate2g_seed_frozen `
  --base-url http://127.0.0.1:8088 `
  --budget 32 --execute-readonly `
  --out build/gate2g_live

py -m orgsim.gate2g_readonly verify `
  --frozen gate_dolibarr/gate2g_seed_frozen `
  --evaluation build/gate2g_live

Remove-Item Env:DOLIBARR_API_KEY -ErrorAction SilentlyContinue
```

This normally makes **28 GET requests**: four unscored selectors and up to 24 scored cases, with no POST, PUT, PATCH, DELETE or redirects. The command is fail-closed to loopback destinations. Some endpoint permissions will be absent and may legitimately return 403; a 403 establishes the tested role's rights, not a global business rule. No account changes are performed by the evaluation code. The default dry-run has zero requests.

Use `build/gate2g_model_frozen` instead of `gate_dolibarr/gate2g_seed_frozen` in both evaluate/verify commands if you prepared the optional real model candidate. Never prepare a new model candidate after viewing the Gate 2G live outcomes and then call the *same* queries a blind evaluation.

## 4. Share these artifacts

- `build/gate2g_live/holdout_report.json`
- `build/gate2g_live/EVAL_FREEZE.json`
- `build/gate2g_live/events.jsonl` (optional, for deeper chain verification)

Do not share keys, raw ERP record bodies, Docker database snapshots, or an `.env` file. The local verifier establishes internal hash integrity and prediction-before-read ordering; it does not provide an external cryptographic timestamp.

**Interpretation:** Aim for ≥95% predicted-field precision, ≥75% correct-target recall and positive net-utility improvement over the matched-evidence unconditional profile cache, with case counts, abstentions and wrong claims reported. These are useful-performance **targets**, not promised Dolibarr results. An otherwise perfect shape-score still does not prove ERP workflow causality. The synthetic outside-grammar causal family failed and live writes remain disabled.
