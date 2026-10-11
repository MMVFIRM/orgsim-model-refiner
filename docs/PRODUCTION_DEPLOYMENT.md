# Safe operation against a production Dolibarr release

## Scope

The production interface is a minimal observer; no production ERP write path
is available through `tools/dolibarr_observe.py`. It supports GET for one
`orders`, `products`, `invoices` or `thirdparties` record ID at a time. It
requires HTTPS for a non-loopback host, refuses redirects, removes implicit
HTTP proxy behavior and caps response size. It does not auto-discover schemas,
write transactions, read entire customer datasets, or replicate every module.

## Before connecting

1. Create a **dedicated read-only API identity on the Dolibarr server** and
   verify its permissions; do not reuse an administrator/writer key.
2. Restrict server/network access and confirm valid TLS certificates. Avoid
   intermediaries that rewrite authentication or redirect to different hosts.
3. Set `DOLIBARR_READ_KEY` privately. **Never** store it in a file shared with
   a model or commit it to source control.
4. Treat output files containing raw ERP JSON as confidential. Default
   console output is limited to response SHA-256 and byte count.
5. Keep clone accounts, the clone permit, and clone marker inaccessible from
   the production deployment. Code-level controls are defense in depth,
   not proof of independent production isolation.

## Example

```
py tools/dolibarr_observe.py --url https://dolibarr.example.com   --allow-remote-read --family orders --id 123
```

Optional `--out private/order-123.json` writes sensitive raw JSON to a new
private file. The script refuses overwriting an existing evidence file.
Do not send the private file to model providers without independent approval.

## Deploy acceptance

Run tests and release hash verification; validate with a dedicated test
record and read-only identity in the actual target installation. Confirm TLS,
permissions and field shape. All policies imported from a clone remain
**advisory**; compare against production observations, do not execute actions
on production based on a simulated approval alone.

This release has not undergone penetration testing, independent security
certification, or full compatibility testing across Dolibarr versions/modules.
