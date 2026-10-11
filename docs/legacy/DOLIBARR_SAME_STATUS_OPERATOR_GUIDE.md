# OrgSim × Model Refiner v0.5 — Dolibarr same-status causal transfer

**Research operator guide · Dolibarr 24.0.x · disposable snapshot-backed local sandbox only**

The objective is to learn a **non-status** business restriction from real before/action/after evidence and measure held-out prediction accuracy. The Python runner uses a **read-only** Dolibarr identity and sends GET requests only. No programmatic Dolibarr write is executed. All POST actions are performed **separately by the authorized human** in the local disposable Docker tenant, and all server receipts/logs stay private.

## Why the target changed

v0.4's 10/10 holdout pairs produced 20/20 correct decisions for the frozen GPT-6 baseline **and** every adapted arm. Ten negative cases were HTTP 304 non-transitions of an already-validated order. There was no measured benefit from simulated rehearsal or the learned policy. v0.5 **forbids changing order status within a pair** and **does not accept a 304 as evidence of a business-rule refusal**.

Dolibarr 24 source provides a more causal candidate. The `Commande::valid()` method calls `checkActiveProductInLines()` unless configuration `ORDER_NOCHECK_ONSALE_PRODUCTS_ONVALID` bypasses it; failed checks return a negative result to `api_orders`, which may expose them as HTTP 500. That makes a same-draft-status, single-line pair of otherwise matched orders using a sale-enabled versus sale-disabled product a **candidate** business-rule experiment. The exact behavior depends on configuration and installed modules. Therefore **pilot before you prepare the study**.

References:
- [Dolibarr 24 Commande::valid implementation](https://doxygen.dolibarr.org/dolibarr_24.0/dev/build/html/dc/df5/commande_8class_8php_source.html)
- [Dolibarr 24 orders REST validation endpoint](https://doxygen.dolibarr.org/dolibarr_24.0/dev/build/html/d9/d7b/api__orders_8class_8php_source.html)

## 0. Preconditions

Ensure these conditions before starting:

- Dolibarr is a **throwaway** Docker instance with a restorable database/file-volume snapshot; no production or shared business records.
- Two accounts/keys: `DOLIBARR_API_KEY` **read-only** for automated GET snapshots and a **separate authorized writer identity** for manual validations. Never give the watcher key mutation rights. Do not paste either key here.
- You can create draft customer orders, order lines and products. Use product price, type and stock that are the same on both sides. Set both orders to **draft status (0)**; each must contain **exactly one product line**, with identical quantity, line totals and customer properties. Start with two products with the same numeric attributes; after building the orders, disable **selling / on sale** for just one product. Keep buying status, stock, product type and prices matched.
- The operator is the only one making mutations during each pair. Take an image or Docker/database backup **before** any test actions. Do not run actions against a tenant you cannot restore.

On Windows PowerShell, from the extracted release folder:

```powershell
py -m pip install -e ".[test]"
py -m pytest -q tests

# Keep a private watcher key in your environment, not on the command line or in output.
$env:DOLIBARR_API_KEY = '<local read-only watcher key>'
```

## 1. Two-order pilot: do not seed dozens of fixtures yet

Prepare **two otherwise matched fresh draft customer orders**. Each must have exactly one line referencing its own product. One product is active for sale (`products/{id}` `status` 1), the other inactive for sale (`status` 0). The order status for **both** must still be 0. The other exposed safe values should match. Record their IDs locally.

```powershell
py -m simlearn.dolibarr_causal pilot-start `
  --root build/causal_pilot_001 `
  --order-a 101 --product-a 201 `
  --order-b 102 --product-b 202
```

**Replace 101/102/201/202 with the actual disposable IDs.** This contacts only the loopback Dolibarr URL with GET requests and verifies both raw single-line-to-product links. It reports which side is sale-enabled; **do not assume `a` is the successful side**.

Next, **you manually** attempt `POST /api/index.php/orders/{id}/validate` on *both* orders with your separate authorized writer. The operation accepts JSON request fields such as `idwarehouse` and `notrigger`; use your tenant's actual Swagger/API explorer for precise syntax. If using a manual client, save the **actual** HTTP statuses; never use the watcher credential for POST. Capture a **redacted** copy of the failed response in a local file such as `private_receipts/pilot_500.txt` and a **separate, correlated** product-status diagnostic from Dolibarr/PHP logs as `private_receipts/pilot_validation_log.txt`. Remove names, emails, API keys and other secrets from both files. Do not manufacture an error or substitute a generic PHP exception for this diagnostic.

For the pilot to qualify, the sale-enabled order must undergo an observable state transition with an actual 2xx action result; the disabled-product order must return an actual **500 containing Dolibarr's `Error when validating Order:` application prefix**, and its diagnostic must specifically identify a product-on-sale validation failure (`checkActiveProductInLines` or an equivalent, correlated error). It must also leave the order's observable state unchanged. HTTP 403, 304, uncorroborated 500, and 2xx without a state change **do not qualify**.

```powershell
py -m simlearn.dolibarr_causal pilot-finish `
  --root build/causal_pilot_001 `
  --active-status 200 --inactive-status 500 `
  --receipt-file private_receipts/pilot_500.txt `
  --diagnostic-file private_receipts/pilot_validation_log.txt
```

**The 200/500 in this command are examples of actual statuses the protocol requires, not a claim your tenant will return them.** If your own observations differ, **stop** rather than substitute expected statuses. Report the **sanitized error category** and we can select a new business condition. In particular, if `ORDER_NOCHECK_ONSALE_PRODUCTS_ONVALID` is enabled or the product status is not exposed through GET, this experiment is unsuitable until you legitimately adjust the disposable sandbox.

On success, the read-only pilot writes `build/causal_pilot_001/public/PILOT_CONFIRM.json`, which contains evidence hashes **but no raw order IDs**. Its evidence is operator-attested, not independently authenticated.

## 2. Freeze the fixture plan before creating the main study

The pilot examples are development evidence only. **Never reuse pilot orders or products in the study.**

```powershell
py -m simlearn.dolibarr_causal scaffold `
  --pilot build/causal_pilot_001/public/PILOT_CONFIRM.json `
  --train-pairs 5 --validation-pairs 3 --holdout-pairs 10 `
  --out build/causal_template.json
```

This creates `build/causal_template_IDS.csv` with 36 case rows (18 pairs), each requiring an independent disposable customer order. CSV columns are:

`case_id,pair_id,operation,split,expected_product_sale_status,record_id,product_id`

**Fill in only `record_id` and `product_id`.** The script alternates the intended on-sale side across pairs to prevent an “always a” shortcut. The expected product status is a **fixture-creation instruction**, not the truth label. Both orders in each pair must have identical status 0, line quantity, customer setup, price, monetary totals and other readable attributes, with exactly one line each linking to the respective `product_id`. Exposed product status should be the *only* observed difference (`watch_1.status`). Use fresh products for independent cases where practical. If products must be reused, keep them unmodified during heldout actions; product reuse reduces independence.

You may enter private fixture identifiers into the CSV using a spreadsheet or text editor. Once the CSV is complete:

```powershell
py -m simlearn.dolibarr_causal bind `
  --template build/causal_template.json `
  --csv build/causal_template_IDS.csv `
  --out build/causal_plan_PRIVATE.json

py -m simlearn.dolibarr_causal init `
  --plan build/causal_plan_PRIVATE.json `
  --root build/dolibarr_causal_001
```

The binder rejects duplicate order IDs, missing product IDs and changed case/split assignments. Keep all filled plans, CSVs and watcher snapshots locally.

## 3. Run train and validation pairs

Run each pair **before either member is changed**:

```powershell
py -m simlearn.dolibarr_causal start-pair `
  --root build/dolibarr_causal_001 `
  --pair train-order_validate-001
```

If matched-field values, lifecycle status, single-line association or the observed sale-status contrast do not satisfy the protocol, `start-pair` stops before committing. Fix the disposable fixtures rather than editing the comparator. The system captures both commitments before returning control to the operator.

After a manual **actual success** with the writer identity and observed GET change, run for the applicable case ID:

```powershell
py -m simlearn.dolibarr_causal finish-case `
  --root build/dolibarr_causal_001 `
  --case train-order_validate-001-a `
  --outcome SUCCESS --source UI --operator-role authorized_writer
```

For an **actual documented application-level business refusal with HTTP 500**, when BOTH private redacted files exist and you have seen the product validation issue:

```powershell
py -m simlearn.dolibarr_causal finish-case `
  --root build/dolibarr_causal_001 `
  --case train-order_validate-001-b `
  --outcome BUSINESS_REJECTED --source MANUAL_API `
  --operator-role authorized_writer --receipt-status 500 `
  --business-reason product_not_on_sale `
  --receipt-file private_receipts/pair001_response.txt `
  --diagnostic-file private_receipts/pair001_diagnostic.txt
```

**Replace every case ID with the actual observed case. For correlated 500 responses no machine-readable business code is assumed; the redacted response and local diagnostic are mandatory.** Each pair's sale-enabled side may be `a` or `b`; check your CSV and the runner output. Do not use the success or negative command unless that action really happened. The business code is an actual *nonsecret* server token, not a label invented to pass validation.

For a genuine 403 access denial or a 500 with an unrelated/uncertain cause, use `ACCESS_DENIED` or `OTHER_ERROR`, not `BUSINESS_REJECTED`:

```powershell
py -m simlearn.dolibarr_causal finish-case `
  --root build/dolibarr_causal_001 `
  --case train-order_validate-001-b `
  --outcome OTHER_ERROR --source MANUAL_API `
  --operator-role authorized_writer --receipt-status 500
```

HTTP 304 must be recorded as `NO_OP` with its local receipt; **it cannot count as a business-rule reject in v0.5**. Finish every training and validation case, then check readiness:

```powershell
py -m simlearn.dolibarr_causal status --root build/dolibarr_causal_001
py -m simlearn.dolibarr_causal fit --root build/dolibarr_causal_001
```

At least four valid **success + business refusal** training pairs and two equivalent validation pairs are mandatory. The fit fails closed otherwise. Frozen policies are tenant-bound, and no holdout may be opened before a qualified fit.

## 4. Held-out paired test, with optional GPT-6

The ten holdout pairs are fresh, disjoint disposable orders and products, with the same controlled intervention. For each pair, `start-pair` freezes both predictions **before you perform either writer action**.

```powershell
py -m simlearn.dolibarr_causal start-pair `
  --root build/dolibarr_causal_001 `
  --pair holdout-order_validate-001
```

The optional four-arm GPT-6 Luna comparison requires the provider key and explicit consent to send **sanitized but potentially sensitive numerical fields**:

```powershell
py -m pip install -e ".[openai]"
# Configure OPENAI_API_KEY securely in your local environment.
py -m simlearn.dolibarr_causal start-pair `
  --root build/dolibarr_causal_001 `
  --pair holdout-order_validate-001 `
  --execute-model --share-sanitized-financials --model gpt-6-luna
```

Each pair makes up to **8 logical provider calls** (4 conditions × 2 cases). Ten holdout pairs therefore involve **up to 80 logical calls**, not counting vendor-side retry/billing details. Do not run `start-pair` twice for the same pair. Model calls are disabled by default.

Record actual outcomes as in step 3. Do not alter the already frozen study plan, the fitted policy, training labels or scorer after seeing holdout results.

## 5. Verify and report

```powershell
py -m simlearn.dolibarr_causal verify --root build/dolibarr_causal_001
py -m simlearn.dolibarr_causal report --root build/dolibarr_causal_001
```

Send back **only**:

- `build/dolibarr_causal_001/public/PAIRED_REPORT.json`
- `build/dolibarr_causal_001/public/PAIRED_AUDIT_FREEZE.json`
- Optionally the public `build/causal_pilot_001/public/PILOT_CONFIRM.json` (contains no raw IDs)

Do **not** upload raw HTTP bodies, Docker/PHP logs, API keys, private fixture IDs, database snapshots, private prediction files or `private/EVENTS.jsonl`.

## How the result will be interpreted

The report distinguishes a business-rule **transition-quality gate** (>=95% accuracy and coverage, zero observed false accepts, >=8 qualified held-out pairs) from an **incremental Model Refiner gate** (the adapted frozen GPT-6 must actually beat the unadapted GPT-6 on the *same* cases). A 100% unadapted baseline **does not constitute evidence of additional adaptation benefit**. Both are reported separately. It also includes an explicit status-only baseline; because both sides have the same initial status, such a classifier should not distinguish them.

**Scope caveats:** A perfect score from the structured tree on one known, explicitly varied product-status rule does not establish generalized ERP policy discovery. Manual interventions and logs are not cryptographically authenticated. Records within the same Dolibarr tenant are correlated. Unobserved plugins, hidden fields, concurrency and configuration can still confound the result. `effect_exact_rate` is learned by a separate effect program, not independently predicted by the LLM. No automatic ERP write authorization follows.
