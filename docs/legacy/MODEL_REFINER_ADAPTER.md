# Model Refiner external executable-program memory

## What the integration is

`simlearn.structured.ModelRefinerPolicyAdapter` holds a program for each synthetic operation. Every program contains JSON guard expressions and an audited status; it has no Python code evaluation, no side effects, and no write tools. The adapter binds a program to its specific synthetic tenant fingerprint. Model-proposed `ACCEPT`/`REJECT` labels are advisory only. The sidecar's `MODEL_REFINER_PROGRAM_MEMORY.json` can be versioned and locally integrity-checked alongside a native Model Refiner `SuiteArtifact`, but it does not modify that artifact or the provider model weights.

The executable program knows only what the public guard checks and counterexample-audited black-box simulator results support. A failed audit leads to `ABSTAIN` for otherwise public-admissible requests. Explicitly public-invalid requests can still be `REJECT` even when private rules remain unknown. This is intentionally a three-valued policy, not a classifier that guesses on every sample.

## Actual model-provider experiment (opt-in)

`simlearn.model_assisted.run` uses real `model_refiner.suite.runtime.Predictor`, `model_refiner.suite.artifact.Recipe` and the `OpenAIResponsesBackend`. On the *same* held-out synthetic examples it asks the provider for a baseline decision and a decision prompted with the frozen structured program as lower-trust adaptation data. It measures both raw LLM outputs and the policy-verified outputs. The latter are deterministic program outcomes and **must not be reported as an improvement in the LLM's intrinsic ability**.

Dry-run shows two requests per example, because repair and extra verification are disabled. The explicit `--execute-model` flag, private `OPENAI_API_KEY`, and a hard request ceiling are required before any provider request is made. No password or token is serialized; no real ERP is contacted. The native `suite.fit` optimizer is not retrained by this command. The optional test is valuable for measuring whether executable rule material helps GPT-6 *follow* verified policies; a broader non-LLM-to-LLM integration would require updates to Model Refiner's native training and promotion workflow, and independent validation.

```powershell
py -m simlearn.structured_cli llm `
  --data results/structured_confirmatory/env1_data `
  --memory results/structured_confirmatory/env1_policy/MODEL_REFINER_PROGRAM_MEMORY.json `
  --limit 64 --model gpt-6-luna

# Explicit, costly opt-in
py -m simlearn.structured_cli llm `
  --data results/structured_confirmatory/env1_data `
  --memory results/structured_confirmatory/env1_policy/MODEL_REFINER_PROGRAM_MEMORY.json `
  --limit 64 --max-calls 128 --execute-model `
  --model gpt-6-luna --out build/llm_program_assist
```

The bundled frozen holdout is already disclosed by this package; a real fresh blind confirmation requires newly generated seeds, unchanged algorithm settings, and an untouched test partition.
