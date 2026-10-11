# SDK and external observation workflow

## Host responsibilities

Supply a bounded `Schema`, an exact `Scope(tenant, operation, configuration)`, labeled learning observations, reserved calibration data, and a callback returning **observed** outcomes. The callback is trusted code provided by the host; the framework does not sandbox arbitrary callbacks. Do not connect it to automatic production writes.

```python
from orgsim_refiner import ActivePolicyLearner, LearnConfig
from orgsim_refiner.evidence import TrainingOracle

# schema, scope, initial_observations and calibration are prepared by your host.
# observe(sample) must return an Observation with the exact scope and sample,
# an actual observed label, and authoritative provenance—not a model guess.
oracle = TrainingOracle(
    schema, scope, observe, budget=8192,
    forbidden_hashes={row.sample_hash for row in calibration},
)
learner = ActivePolicyLearner(schema, scope, LearnConfig())
policy = learner.fit(initial_observations, oracle)
policy = learner.calibrate(policy, calibration)
policy.save("new_policy.json")
result = policy.predict(new_features, scope)
```

Explicitly reserve holdout input hashes too when your evaluator supplies them. Never give holdout labels to the callback or learner. The supplied benchmark handles these boundaries itself. `advisory=True` is a diagnostic prediction mode for unpromoted artifacts, not permission to act.

## Command-line observed-outcome inbox

For a new, separately planned integration, create a spec JSON with `scope` and `schema`, a training JSONL and validation JSONL of `Observation.to_dict()` records. Then:

```powershell
py -m orgsim_refiner init --spec private/spec.json `
  --training private/train.jsonl --validation private/validation.jsonl `
  --root build/new_observation_workspace
py -m orgsim_refiner learn --root build/new_observation_workspace
```

When the learner needs a new observation it returns `awaiting_observation` and writes a request under `private/requests/`. An external resettable simulator can execute it. For a real environment, only an appropriately authorized operator may decide whether and how a request can be mapped to a safe disposable fixture; arbitrary requested combinations may be infeasible. The inbox itself never contacts the ERP.

After obtaining an actual result, write an Observation JSON and submit:

```powershell
py -m orgsim_refiner submit --root build/new_observation_workspace `
  --request ACTUAL_REQUEST_HASH --observation private/observed_answer.json
py -m orgsim_refiner learn --root build/new_observation_workspace
```

`SIMULATOR_OBSERVED` means observed output of an independent reference simulator, not output of the candidate being evaluated. `REAL_OPERATOR_CORROBORATED` requires before/after state, a declared outcome category, operator correlation, and separate local response/diagnostic files via `--receipt` and `--diagnostic`. File existence and operator declarations do not independently authenticate causality. The legacy Dolibarr-specific reason checker remains available separately.

Unknown outcomes stop certification rather than being forced into a business refusal. Submitted outcomes cannot be overwritten. Once frozen, start a new explicitly versioned workspace for new learning. This inbox is **not** a mechanism for amending the existing frozen v0.6.1 factorial study.

## Pure predicted simulator

Fit `EffectModel(scope)` on historical observed transitions and disjoint validation. Pass a calibrated policy, effect model, state and explicit feature map to `PolicySimulator`. `step` returns a predicted transition and swaps local state only when both decision and declared effects resolve. It returns `UNKNOWN` otherwise. Generated trajectories have `MODEL_PSEUDOLABEL` provenance and cannot bootstrap themselves into authoritative learning evidence.

The effect learner supports a limited expression family and all declared observed fields, not arbitrary asynchronous services or financial accounting. Inspect the effect certificate and coverage.
