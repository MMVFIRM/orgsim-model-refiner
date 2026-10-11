# OrgSim × Model Refiner v0.6 — One-shot real Dolibarr factorial study

**Research operator protocol; Dolibarr 24.0.1; disposable Docker tenant only.**

## What this one experiment can and cannot settle

Primary question, pre-registered **before the main observations**: does a frozen GPT-6 Luna supplied with a learned executable policy outperform GPT-6 with **real labeled-example retrieval** on the same real Dolibarr cases requiring the composition of **two separately established constraints**, after learning only from other combinations?

Secondary questions: can the executable policy reject the withheld double-failure combination (`00`); does it meet high decision coverage, accuracy and false-accept controls; and can a training-derived, non-status **stock-effect formula** predict product stock changes when validation succeeds?

**Do not call the result universally definitive.** One tenant, one operation, one two-factor family, operator-attested receipts and correlated records cannot prove arbitrary enterprise simulation fidelity. A tie or underpowered improvement is a valid negative/inconclusive outcome; do **not** introduce a new gate to rescue the numbers.

## Two candidate business factors, and mandatory feasibility pilot

- `A`: a product is **on sale** (GET `/products/{id}` `status=1`). In Dolibarr 24, customer-order validation can reject products not active for sale unless `ORDER_NOCHECK_ONSALE_PRODUCTS_ONVALID` disables this check. This was independently piloted in the earlier v0.5 experiment.
- `B`: the product has sufficient real stock (`stock_reel >= order line quantity`), **only if the target tenant actually prevents negative stock on order validation**. Dolibarr 24 can perform stock movement at order validation when `STOCK_CALCULATE_ON_VALIDATE_ORDER=1`; that alone does not guarantee an insufficient-stock rejection. Use a single valid warehouse and verify its stock policy. The user must test whether B is genuinely enforced before declaring the pilot successful.

Read the code specific to your deployed version and verify configuration; stock policy may depend on inventory module and installation. Relevant upstream Dolibarr 24.0.1 code: `Commande::valid()`, `api_orders::validate()` and `checkActiveProductInLines()`.

The Python harness has **no POST, PATCH, PUT or DELETE calls**. The only real ERP mutations are manual operations by a separate authorized user, after a snapshot-backed operator check. The watcher key must be read-only, stored in the environment, and never sent to ChatGPT. Use fresh, disposable data with a restorable database **and file volumes**, only on `http://127.0.0.1:8088`.

### Set up

Open PowerShell in the extracted package folder:

```powershell
py -m pip install -e ".[test]"
py -m pytest -q tests/test_dolibarr_factorial.py

# Configure DOLIBARR_API_KEY securely with a GET-only key in your environment.
py -m simlearn.dolibarr_factorial pilot-template --out build/factorial_pilot_IDS.csv
```

In `build/factorial_pilot_IDS.csv`, populate **only** `order_id` and `product_id`. Create three matched draft customer orders, each with exactly one line of the same positive quantity, product type, product price, tax and total. Use distinct products, distinct orders, the same authorized writer and warehouse, and no concurrent transactions. Required experimental cells:

| Cell | Product on sale | Stock sufficient | Purpose |
|---|---|---|---|
| `11` | Yes | Yes | Positive control; expect actual 2xx and observed status change |
| `01` | No | Yes | Tests the sale rule with stock sufficient |
| `10` | Yes | No | Tests the independent stock rule with sale eligible |
| `00` | No | No | **Do not create, run, or reveal until the frozen holdout** |

Stock must be visible through GET `/products/{id}` as the `stock_reel` numeric field, and order quantity must be available through GET `/orders/{id}?withLines=true`. Stock *availability* refers to the same physical warehouse used in the writer validation; otherwise aggregate stock may be misleading.

```powershell
py -m simlearn.dolibarr_factorial pilot-start `
  --csv build/factorial_pilot_IDS.csv `
  --root build/factorial_pilot_001
```

**Only after the above returns successfully**, manually run `POST /api/index.php/orders/{id}/validate` for all three disposable orders with your **separate writer identity**. Pass the same appropriate `idwarehouse` and `notrigger=0` for every case. The supplied Python never issues this POST; do not reuse the watcher identity to write. Retain the real HTTP response body and one distinct, correlated local Dolibarr/PHP diagnostic for each failure in **redacted local files**.

A qualifying pilot requires: `11` actual 2xx with GET-observed change, `01` actual application-level 500 with documented product-on-sale reason and unchanged state, `10` actual application-level 500 with documented insufficient-stock reason and unchanged state. **v0.6.1 evidence fix:** the specific business reason may occur in the *HTTP validation response* rather than the local Dolibarr log. In that case the separate log must show a corresponding **stock movement attempt + transaction rollback** for the stock refusal (or order validation attempt + transaction rollback for the sale refusal). A reason-specific local diagnostic remains acceptable under the original route. These are operator-correlated, not independently authenticated, artifacts. 403/304/404/generic 500 **never** qualify. Do not invent evidence or combine files from different attempts.

```powershell
py -m simlearn.dolibarr_factorial pilot-finish `
  --root build/factorial_pilot_001 `
  --success-status 200 --sale-status 500 --stock-status 500 `
  --sale-receipt private_receipts/sale_response.txt `
  --sale-diagnostic private_receipts/sale_log.txt `
  --stock-receipt private_receipts/stock_response.txt `
  --stock-diagnostic private_receipts/stock_log.txt
```

Replace status values with **actual** observed values. To check a redacted receipt and log *without changing the pilot state*, use:

```powershell
py -m simlearn.dolibarr_factorial check-evidence `
  --receipt private_receipts/stock_response.txt `
  --diagnostic private_receipts/stock_log.txt `
  --reason insufficient_stock
```

If your existing v0.6 `pilot-finish` failed **before** creating `PILOT_CONFIRM.json`, update the single `src/simlearn/dolibarr_factorial.py` source file and rerun the same `pilot-finish` with the existing `--root` and evidence files. **Do not repeat the writer actions**, do not reset those three records, and do not scaffold until qualification succeeds. The resumed command takes fresh GET snapshots only. The original `PILOT_START.json`, committed before-state, receipt files, and snapshot remain in place.

If the stock-negative order actually succeeds, has a 304, a permission problem, or an unrelated 500, **stop**. That means this factorial test is not feasible in this tenant as configured. Do not invent a result or repurpose v0.5 receipts.

## Build one frozen, fresh study

Only after pilot qualifies:

```powershell
py -m simlearn.dolibarr_factorial scaffold `
  --pilot build/factorial_pilot_001/public/PILOT_CONFIRM.json `
  --train-blocks 6 --validation-blocks 3 --holdout-blocks 16 `
  --seed 20261011 --out build/factorial_template.json
```

This pre-registers **91 fresh disposable orders and 91 fresh products**: 18 training, 9 validation, 64 holdout. The holdout has sixteen four-record matched blocks, containing `11`, `01`, `10`, `00`, with randomized order within each block. Product and order IDs are never reused. The 3 pilot orders/products are separate and must **not** be reused. Each block has identical quantity and totals across members, and differs only in the two deliberately manipulated business factors. The program checks the exposed safe fields and real order line/product associations at runtime.

In `build/factorial_template_IDS.csv`, edit only `order_id` and `product_id`. The condition assignments, split, IDs and quality scoring are fixed. Check that `stock_reel >= qty` matches the `stock_sufficient` column, keeping all products in the same relevant warehouse. Once all fixtures are independently created and the full database/volume snapshot exists:

```powershell
py -m simlearn.dolibarr_factorial bind `
  --template build/factorial_template.json `
  --csv build/factorial_template_IDS.csv `
  --out build/factorial_private_plan.json

py -m simlearn.dolibarr_factorial init `
  --plan build/factorial_private_plan.json `
  --root build/dolibarr_factorial_001
```

**Nothing from the holdout is allowed to train the system.** There are zero `00` cases in the pilot, training, or validation. After each `start-block` command returns, use your authorized writer account for all its orders, then finish the cases *based on what actually happened*. Do not manually act before the block-start command completes.

Training example:

```powershell
py -m simlearn.dolibarr_factorial start-block `
  --root build/dolibarr_factorial_001 `
  --block train-block-001
```

For each observed successful case, after the separate actual writer action:

```powershell
py -m simlearn.dolibarr_factorial finish-case `
  --root build/dolibarr_factorial_001 `
  --case train-block-001-x1 `
  --outcome SUCCESS --http-status 200
```

For each **actual** app-level `500` failure caused by the product-not-for-sale or insufficient-stock rule, supply its own **redacted** writer response and related local diagnostic (don't infer the reason from the intended cell alone). A specific business reason in the HTTP body plus a separate relevant transaction rollback log is valid evidence even when the local log doesn't repeat the reason:

```powershell
py -m simlearn.dolibarr_factorial finish-case `
  --root build/dolibarr_factorial_001 `
  --case train-block-001-x2 `
  --outcome BUSINESS_REJECTED --http-status 500 `
  --receipt-file private_receipts/actual_response_for_case.txt `
  --diagnostic-file private_receipts/actual_log_for_case.txt
```

Replace the example case ID and HTTP status with the real ones; assignments vary by block. **Each refused action must have its own independently saved redacted HTTP receipt with a distinct digest** (include a nonsecret request/time identifier if the sanitized error text would otherwise be identical); the runner blocks reuse of identical refusal receipts. The tool derives which mechanism must be demonstrated. A rollback log without the business reason in either artifact, a generic API 500, or a stock-refusal HTTP body with no local stock-movement/rollback evidence **does not qualify**. If neither file combination supports the real error, record `OTHER_ERROR` instead of silently counting a false rejection. For permission problems, record `ACCESS_DENIED`, for 304 use `NO_OP`, and for uncertain status use `UNKNOWN`. **Unknown/no-op/errors prevent a qualified fitted study; that is a legitimate finding, not a reason to edit labels.**

For each train block `train-block-001` ... `train-block-006` and each validation block `validation-block-001` ... `validation-block-003`, run the same start/finish sequence, recording all three cases. Then:

```powershell
py -m simlearn.dolibarr_factorial status --root build/dolibarr_factorial_001
py -m simlearn.dolibarr_factorial fit --root build/dolibarr_factorial_001
```

`fit` refuses to proceed unless all 27 training/validation outcomes qualify, the two independent refusal mechanisms are observed as planned, the hidden `00` cell was never used, and a **unique executable program** is consistent with training and validation. It freezes the program, its provenance and any learned status/stock effect equations. Training labels remain operator-attested, not authenticated by a third party.

## Frozen holdout, optionally compare GPT-6

**Commit all predictions for a holdout block before performing its actions.** E.g.:

```powershell
py -m simlearn.dolibarr_factorial start-block `
  --root build/dolibarr_factorial_001 `
  --block holdout-block-001
```

To compare actual GPT-6 Luna against its retrieved examples and executable policy in a *single study*, run every holdout block with **explicit model opt-in** instead:

```powershell
py -m pip install -e ".[openai]"
# Configure OPENAI_API_KEY securely. Provider receives only sanitized numeric/status facts.
py -m simlearn.dolibarr_factorial start-block `
  --root build/dolibarr_factorial_001 `
  --block holdout-block-001 `
  --execute-model --share-sanitized-fields --model gpt-6-luna
```

**Choose once before starting any holdout** whether the study includes the provider comparison. Never run `start-block` twice for the same block; the command refuses overwrites. Four LLM conditions per case (`baseline`, `real retrieval`, `program-labeled synthetic rehearsal`, and `compiled policy`) mean **up to 256 logical model calls for 64 cases**. Model calls are disabled by default; real provider attempts and billing may differ from logical call counts. There are **zero automated Dolibarr writes** in either mode.

Save all actual writer receipts as private local files. For `00` cases, both factors fail; the underlying Dolibarr validation may report only the first failure. The study scores the actual outcome and checks a *real*, correlated product-sale diagnostic rather than asserting that both errors must be returned.

## Pre-registered scoring and stopping rule

The study is committed to a single **64-case/16-block** evaluation, with four equally represented factorial cells. Main accuracy and balanced accuracy include all 64 planned cases via a separate planned-coverage accounting; uncertain labels reduce usable coverage. Never exclude difficult cases and then report only an inflated conditional score.

The direct executable learner must achieve **>=95% correctness and coverage** and **zero false accepts** on qualified holdout decisions. The incremental Model Refiner claim has a stronger requirement: model-policy **>=95% accuracy, >=95% coverage, zero false accepts**, and an absolute improvement **>=10 percentage points** over **GPT-6 + real-example retrieval**, with the 95% **block-bootstrap paired confidence interval entirely above zero**. Compare to unadapted GPT-6 as a secondary contrast; the transparent conjunction `A∧B` and individual-factor baselines are also scored and may match or outperform the learner. If retrieval or unadapted GPT-6 tie at 100%, the unique-adaptation claim **fails**, even if the compiled policy is perfect.

For genuine sim-to-real state effects, the training-derived program must have a frozen prediction of the change in product `stock_reel` for accepted validation. Only count true before/after GET-confirmed non-status effects; no GPT-generated state effect score is claimed by this protocol.

```powershell
py -m simlearn.dolibarr_factorial verify --root build/dolibarr_factorial_001
py -m simlearn.dolibarr_factorial report --root build/dolibarr_factorial_001
```

Upload **only**:

- `build/dolibarr_factorial_001/public/FACTORIAL_REPORT.json`
- `build/dolibarr_factorial_001/public/FACTORIAL_AUDIT_FREEZE.json`
- Optionally `build/factorial_pilot_001/public/PILOT_CONFIRM.json`

Keep all CSV IDs, private observations, financials, raw HTTP bodies, logs, API keys, and Docker snapshots **local**. The freeze uses hash-linked local evidence, not external timestamps or independently certified transactions.

## What the results can mean

- **Support for scope:** program learning from separate A and B observations correctly handles their unseen conjunction/overlap in real order validation.
- **Strongest incremental support:** learned-policy prompt passes the exact preregistered paired superiority test versus both standard retrieval and the frozen model.
- **Valid negative result:** retrieval, the transparent two-factor control, or the frozen model matches the policy; the system did not demonstrate a unique benefit despite successfully learning this bounded environment.
- **Infeasible experiment:** the stock constraint does not actually prevent validation in the user's configured tenant. Don't proceed to the main study.

This experiment does not authorize production actions or claim that other ERP rules were discovered.
