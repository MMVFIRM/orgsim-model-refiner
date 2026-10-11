# Authorized disposable-clone learning only

The live research oracle intentionally **writes** records to learn order
validation. Never run it on production. An on-host reverse proxy to production
would not be safe even with a loopback URL.

## Preparation

Use a disposable isolated clone matching the Dolibarr production version,
modules, relevant settings and schema. Disable outgoing payments, email,
third-party webhooks, fulfillment and scheduled jobs; use clone-specific API
credentials, network isolation, and an operator-controlled restore snapshot.

Produce an expiring local permit (offline; no ERP write):

```
py tools/prepare_clone_permit.py   --url http://127.0.0.1:8189   --expires-hours 8 --max-mutations 30000   --ack "I CONFIRM THIS IS A DISPOSABLE CLONE"   --out private/clone-permit.json
```

The script prints a random `ORGSIM-CLONE-...` product reference and unique
label. **Manually create a product with exactly this reference and label only
in the disposable clone.** Do not add the marker to production. Before the
first mutation the oracle GETs this marker and verifies both fields. This is
an additional operator safety check, *not cryptographic proof* of identity.

Run the clone learning benchmark using clone-specific writer credentials:

```
py tools/erp_experiment.py --oracle live --url http://127.0.0.1:8189   --api-key-env DOLIBARR_CLONE_KEY --clone-permit private/clone-permit.json   --workers 1 --seed 1 --budget 4000 --out build/clone-study-001
```

Other live tools `erp_replicate.py` and `erp_drift_check.py` accept the same
`--clone-permit` option. Every mutation is charged against the permit budget.
Writes that time out are **not retried**; stop and reconcile/restore first.
Post-validation GETs must be available to verify both approvals and named
business-rule refusals. Generic HTTP 500 errors remain unclassified.

Do not move or copy raw clone receipts, marker labels, writer keys or private
business data into shared public reports. Clone predictions are not a license
for live production actions. Active learning can consume thousands of writes.
