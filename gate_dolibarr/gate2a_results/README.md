# Frozen Gate 2A results

- `budget8/`, `budget20/`, `budget32/`: 24-seed independent confirmatory comparison using base seed **20261231**. Each includes the public contract, aggregate per-seed metrics and the summarized report. No model service or live ERP calls.
- `compiled_smoke/`: one learned **provisional** order-validation manifest compiled into the actual OrgSim runtime, its compiled registry, and a differential report on 160 independent synthetic heldout cases. **Safety gate failed due to 23 false accepts.**
- All files are development/research artifacts. The real Dolibarr 24.0.1 tenant was not contacted in this release.

See `docs/GATE2A_METHOD.md` and `docs/GATE2A_RESULTS.md` for definitions, uncertainty and limitations.
