# v0.6 offline-only harness audit (not real Dolibarr evidence)

New automated tests simulate a throwaway Dolibarr read server and a separately executed synthetic writer with two enforced restrictions. The code's read path receives no mutation privileges.

- A three-case feasibility pilot detects the individual sale and insufficient-stock restrictions and confirms two distinct no-transition refusals.
- Precommitted training and validation include `11`, `01`, `10`; `00` is excluded from pilot/train/validation.
- The program compiler recovers the conjunction from labeled outcomes and freezes it before opening heldout blocks.
- A 48-case/12-block synthetic dry run exercises all four factorial cells. It yields 48/48 learned-program decisions and 12/12 stock-effect predictions, while always-reject scores 36/48. These are **mock/harness-only results**, not measured model or ERP performance.
- A mock provider that always rejects verifies the incremental Model Refiner quality gate fails when no advantage is observed.
- A separate mock provider that gets only the policy condition correct verifies the gate passes a >=10-point statistically positive contrast. Again this is a **scorer test**, not real model improvement.
- Tests reject tampered event files, changed assignments, duplicated product identities, invalid business receipts, the wrong tenant's frozen policy, and a stock pilot that fails in the environment.

Nothing here establishes that a real Dolibarr tenant will enforce insufficient-stock refusal, or that GPT-6 outperforms retrieval on real data. Both require the user's separate run, and the real ERP interactions are always manual for writes.
