# Dolibarr: preserve the existing study

Your v0.6.1 factorial plan, partitions, seed, scoring rules and immutable outcomes remain unchanged. The v0.7 package includes the original `simlearn.dolibarr_factorial` and related source; it is not an overlay to an interrupted workspace. Do not relabel `OTHER_ERROR`, invent a missing log, rerun an action over an altered before-state, or replace old receipts.

The full historical protocol is in `legacy/ONE_SHOT_DOLIBARR_FACTORIAL_PROTOCOL.md`. The new active synthetic results are a separate experiment and cannot be used to claim the existing Dolibarr holdout has passed.

## Read-only local checks

```powershell
py -m orgsim_refiner dolibarr-status --root PATH_TO_EXISTING_RUN
py -m orgsim_refiner dolibarr-check `
  --receipt PATH_TO_ACTUAL_RESPONSE `
  --diagnostic PATH_TO_SEPARATE_CORRELATED_LOG `
  --reason insufficient_stock
```

`dolibarr-check` examines saved evidence only. A passing saved log check does not prove syslog is currently enabled or that the next action will generate a usable diagnostic. Before resuming an administratively restarted study, the operator must verify logging on a separate authorized calibration fixture. Restore snapshots only under the operator's control, preserve the interrupted run, and disclose any training-only exposure. Nothing here restores a database, enables logging, changes permissions or performs writes.

The new offline inbox is a future integration interface. It does not automatically know how to construct all requested ERP states, and its feature schema is not a complete behavioral twin of Dolibarr. Raw live records and writer credentials must stay local. No new live Dolibarr calls were made to build or evaluate v0.7.
