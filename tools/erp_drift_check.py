"""Re-verify a frozen policy against the live (disposable) ERP; flag configuration drift.

  python tools/erp_drift_check.py --policy build/erp_live_1/POLICY.json --samples 120 --out build/drift_1

Samples a mix of uniform and boundary-focused orders, asks the policy first, then the ERP,
and reports mismatches. Exit code 3 if the mismatch rate exceeds --tolerance, meaning the
ERP no longer behaves as the policy was learned (configuration, version, customisation or
data drift): stop using the policy and re-learn it.
"""
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from erp_experiment import uniform_inputs, boundary_inputs, dedupe
from orgsim_refiner.artifact import PolicyArtifact
from orgsim_refiner.erp import orders
from orgsim_refiner.erp.dolibarr_live import DolibarrLive
from orgsim_refiner.util import digest, write_json


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--samples', type=int, default=120)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--tolerance', type=float, default=0.0, help='max tolerated mismatch rate')
    ap.add_argument('--url', default='http://127.0.0.1:8189')
    ap.add_argument('--api-key-env', default='DOLAPIKEY')
    ap.add_argument('--clone-permit', help='Expiring private permit tied to disposable clone marker; required for live mode')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if not a.clone_permit:
        raise SystemExit('Mutating drift checker requires --clone-permit for disposable clone; never production')
    key = os.environ.get(a.api_key_env)
    if not key:
        raise SystemExit(f'Set {a.api_key_env} in the environment.')
    policy = PolicyArtifact.load(a.policy)
    if policy.scope != orders.SCOPE or policy.schema != orders.SCHEMA:
        raise SystemExit('Policy scope/schema does not match the order-validation task.')
    rng = random.Random(a.seed * 31 + 1)
    half = a.samples // 2
    xs = dedupe(uniform_inputs(rng, half * 2), [])[:half] + dedupe(boundary_inputs(rng, half * 3), [])[:a.samples - half]
    erp = DolibarrLive(a.url, key, clone_permit=a.clone_permit)
    bad, rows = [], 0
    for x in xs:
        pred = policy.predict(x, orders.SCOPE)['decision']
        actual = erp.callback(x).label
        rows += 1
        if pred != actual:
            bad.append({'input': x, 'predicted': pred, 'actual': actual})
    rate = len(bad) / rows
    report = {'policy_sha256': policy.fingerprint, 'checked': rows, 'mismatches': len(bad), 'mismatch_rate': round(rate, 6),
              'tolerance': a.tolerance, 'drift_detected': rate > a.tolerance,
              'false_accepts': sum(b['predicted'] == 'APPROVE' for b in bad), 'examples': bad[:10],
              'action': 'STOP using this policy and re-learn' if rate > a.tolerance else 'policy still consistent with the ERP'}
    Path(a.out).mkdir(parents=True, exist_ok=False)
    write_json(Path(a.out) / 'DRIFT_REPORT.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'examples'}, indent=1))
    return 3 if report['drift_detected'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
