# Disposable Dolibarr lab (for v0.8 live experiments)

Isolated from any other Dolibarr you run: its own compose project name, its own volumes, and
port **8189** bound to `127.0.0.1` only. Use a throwaway password; never reuse real credentials.

```powershell
cd lab
Copy-Item .env.example .env        # edit the three passwords
docker compose -p orgsim_lab --env-file .env up -d
# wait until http://127.0.0.1:8189/api/index.php/status answers 401 (installed)
```

1. **Enable the Stock module** (it is not in the default module list for a fresh install):
   `docker cp enable_mod.php orgsim_lab-web-1:/tmp/enable_mod.php`
   then `docker exec orgsim_lab-web-1 php /tmp/enable_mod.php` (on Git Bash prefix with `MSYS_NO_PATHCONV=1`).
2. **Configure stock-on-validation** and create a warehouse:
   `UPDATE llx_const SET value='1' WHERE name='STOCK_CALCULATE_ON_VALIDATE_ORDER'` (insert it if absent),
   then `POST /api/index.php/warehouses {"label":"LAB-WH","ref":"LABWH","statut":1}` (must become warehouse id 1).
3. **Create an API key** for the admin user (password login is disabled in recent Dolibarr):
   `UPDATE llx_user SET api_key='<40 hex chars>' WHERE login='<DOLI_ADMIN_LOGIN>'`, then
   `$env:DOLAPIKEY = "<that key>"`. Do not commit it.

Tear down: `docker compose -p orgsim_lab --env-file .env down` (add `-v` to delete the lab data).

The scripts refuse non-loopback hosts. The oracle serialises transactions by default because
concurrent order validation on Dolibarr 24.0.1 returned spurious stock refusals (see docs/ERP_LEARNING.md).
