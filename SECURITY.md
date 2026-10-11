# Security and production compatibility — v0.9.0

**Research release candidate, NOT security certified or a production write controller.**

- `tools/dolibarr_observe.py` and `DolibarrReadOnly` only issue GET to
  allow-listed object record endpoints. Non-loopback connections require HTTPS
  and explicit `--allow-remote-read`. Redirects and environment HTTP proxies
  are disabled; response sizes are bounded. Treat records as sensitive.
- Production server-side credentials MUST have read-only privileges. The client
  control alone is not sufficient authorization enforcement. Use a private
  secrets store or environment variables, never command-line key values.
- Automated probes create real third parties, products, stock movements, and
  sales orders. They run ONLY on disposable local clones with a separately
  provisioned clone account, an expiring local permit, a unique product sentinel
  that is verified before mutations, and a bounded mutation budget.
- `localhost` can reverse-proxy to a production system. The marker is only an
  additional operator-confirmation control, not cryptographic proof of isolation.
  Ensure no production routing, secrets, email, payments, CRM integrations or
  automated outgoing jobs are active on the clone. A database/document-store
  snapshot and documented restore are mandatory before experiments.
- A failed, timed-out or interrupted POST may have COMMITTED. Such requests are
  not retried automatically; record the uncertainty and reconcile locally.
- A generic HTTP 500 is not a business rejection. Only named stock/sale errors
  with verified unchanged order status and product stock are recognized.
  Ambiguous outcomes are UNKNOWN and cannot certify an accuracy claim.
- Model Refiner predictions and learned policies **never confer ERP authority**.
  Neural weights are not changed; verification is limited to observed scope and
  configuration. The package contains legacy research tools; follow the
  hardened entry points rather than historical commands.
- **Never** use the active clone oracle on production, even for drift testing.
  Use read-only production monitoring and separately authorized clone tests.
- Release hashes demonstrate file integrity within a local copy, not publisher
  signatures or external, tamper-proof provenance. Do not share API keys,
  original business data or private receipts.

See `docs/PRODUCTION_DEPLOYMENT.md`, `docs/CLONE_LEARNING.md`, and
`release_validation/BUILD_STATUS.json` for tested and untested boundaries.
