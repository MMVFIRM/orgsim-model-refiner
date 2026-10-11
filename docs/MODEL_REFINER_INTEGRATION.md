# Model Refiner integration

The new bridge uses the bundled Model Refiner `Recipe`, `Predictor`, `TaskSpec` and `OpenAIResponsesBackend`. It does not run the native `suite.fit` optimizer and does not modify neural weights. Existing native APIs remain included, but the new scoped memory and executable learner form a separate external adaptation path.

## Explicit provider experiment

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[openai]"
.\.venv\Scripts\python.exe -m orgsim_refiner plan `
  --execute-model --model gpt-6-luna `
  --tenants 24 --model-cases 8 --max-output-tokens 4096
```

The preview makes no calls. The default request ceiling needed for this comparison is 24 tenants × 8 heldout cases × 3 prompt conditions = 576 attempts. Set `OPENAI_API_KEY` privately using your preferred local secret mechanism. The model identifier is an explicit user choice and must exist in your provider account. To execute:

```powershell
.\.venv\Scripts\python.exe -m orgsim_refiner study `
  --seed 910273 --tenants 24 `
  --execute-model --model gpt-6-luna --model-cases 8 `
  --max-output-tokens 4096 --max-provider-calls 576 `
  --out build/active_with_model
```

No ERP requests occur. The provider receives schema, scoped synthetic features and either same-scope demonstrations or a compiled AST. It never receives heldout labels. Model outputs that are incomplete, invalid, refused or failed become `UNKNOWN`; there is no invisible retry or relabel. The wrapper records attempted requests and token usage without assuming prices. Failed responses with absent usage remain explicitly unpriced.

## Arms and attribution

`llm_zero_shot` receives the common task and current features. `llm_tenant_retrieval` receives up to 16 examples filtered by tenant/operation/configuration before ranking. `llm_executable_context` receives the frozen, calibrated AST. `program_verified` is a separate deterministic output; any override is recorded. Do not attribute the verifier's accuracy to the LLM.

The model comparison is only attempted for promoted policies. Other tenants appear as unexecuted/unknown for all model conditions; inspect the explicit unexecuted count rather than treating these as baseline model failures. Pure-program results remain available for every tenant.

## Candidate synthesis

`propose` can ask an opted-in model for a JSON AST from training data. A candidate must parse under the bounded DSL and match **every** provided learning label. The SDK can pass accepted candidates through `ActivePolicyLearner.fit(..., extra_candidates=[ast])`. This only adds a hypothesis to disagreement search; it cannot certify truth, bypass calibration, authorize transactions, or label its own examples as ground truth.

No actual provider calls were made while preparing the v0.7 offline benchmark. Mock integrations test the call path and isolation, not the efficacy of any provider model.
