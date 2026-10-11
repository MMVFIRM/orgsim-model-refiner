"""Learn Dolibarr order validation, freeze, then score on sealed held-out cases.

  python tools/erp_experiment.py --oracle twin --seed 1 --budget 4000 --out build/erp_twin_1
  python tools/erp_experiment.py --oracle live --seed 1 --budget 4000 --out build/erp_live_1 \
        --url http://127.0.0.1:8189 --api-key-env DOLAPIKEY --workers 6

Sequence (cannot be reordered by flags): build sealed holdout inputs -> learn with
holdout hashes reserved -> promote on an independent calibration set -> write the
frozen policy and its predictions -> only then obtain held-out labels and score.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from orgsim_refiner.erp import orders
from orgsim_refiner.expr_learner import ExprPolicyLearner, ExprConfig
from orgsim_refiner.evidence import TrainingOracle
from orgsim_refiner.util import digest, write_json


def uniform_inputs(rng, n):
    return [orders.SCHEMA.sample(rng) for _ in range(n)]


def boundary_inputs(rng, n):
    """Evaluator-designed stress: sit on and next to the stock-vs-quantity boundaries."""
    out = []
    S = orders.SCHEMA
    while len(out) < n:
        x = S.sample(rng)
        x['sale_a'] = rng.random() < .85
        x['sale_b'] = rng.random() < .85
        x['kind_a'] = 'good' if rng.random() < .8 else 'service'
        total = x['qty_a1'] + x['qty_a2']
        x['stock_a'] = min(40, max(0, total + rng.choice([-2, -1, -1, 0, 0, 1, 2])))
        if x['qty_b'] > 0:
            x['stock_b'] = min(40, max(0, x['qty_b'] + rng.choice([-2, -1, -1, 0, 0, 1, 2])))
        out.append(x)
    return out


def dedupe(xs, taken):
    seen, out = set(taken), []
    for x in xs:
        h = digest(x)
        if h not in seen:
            seen.add(h)
            out.append(x)
    return out


def score(pairs):
    n = len(pairs)
    ok = sum(p == y for p, y in pairs)
    fa = sum(p == 'APPROVE' and y == 'REJECT' for p, y in pairs)
    fr = sum(p == 'REJECT' and y == 'APPROVE' for p, y in pairs)
    unk = sum(p == 'UNKNOWN' for p, y in pairs)
    return {'cases': n, 'correct': ok, 'accuracy': round(ok / n, 6) if n else None, 'false_accepts': fa,
            'false_rejects': fr, 'unknown': unk,
            'approvals': sum(y == 'APPROVE' for p, y in pairs), 'rejections': sum(y == 'REJECT' for p, y in pairs)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--oracle', choices=['twin', 'live'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--budget', type=int, default=4000)
    ap.add_argument('--holdout', type=int, default=300, help='per held-out set')
    ap.add_argument('--calibration', type=int, default=80)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--url', default='http://127.0.0.1:8189')
    ap.add_argument('--api-key-env', default='DOLAPIKEY')
    ap.add_argument('--clone-permit', help='Expiring private permit tied to disposable clone marker; required for live mode')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)

    out = Path(a.out)
    if out.exists():
        raise SystemExit('Output directory exists; results are immutable. Use a new --out.')
    out.mkdir(parents=True)
    rng = random.Random(a.seed * 7919 + 13)

    if a.oracle == 'live':
        if not a.clone_permit:
            raise SystemExit('Live oracle disabled without --clone-permit; never use production')
        if a.workers != 1:
            raise SystemExit('Live worker parallelism disabled; use --workers 1')
        from orgsim_refiner.erp.dolibarr_live import DolibarrLive
        key = os.environ.get(a.api_key_env)
        if not key:
            raise SystemExit(f'Set {a.api_key_env} in the environment (never pass keys on the command line).')
        live = DolibarrLive(a.url, key, clone_permit=a.clone_permit)
        callback = live.callback
    else:
        callback = orders.twin_callback

    # 1. sealed held-out INPUTS (labels are not requested until the policy is frozen)
    uni = dedupe(uniform_inputs(rng, a.holdout * 2), [])[:a.holdout]
    bnd = dedupe(boundary_inputs(rng, a.holdout * 3), [digest(x) for x in uni])
    bnd = dedupe(bnd, [])[:a.holdout]
    cal = dedupe(uniform_inputs(rng, a.calibration * 3), [])
    reserved = {digest(x) for x in uni + bnd}
    cal = [x for x in cal if digest(x) not in reserved][:a.calibration]
    reserved |= {digest(x) for x in cal}
    write_json(out / 'HOLDOUT_INPUTS_SHA256.json', {'uniform': [digest(x) for x in uni], 'boundary': [digest(x) for x in bnd],
                                                  'calibration': [digest(x) for x in cal]})

    # 2. learn
    cfg = ExprConfig(seed=a.seed, workers=a.workers)
    learner = ExprPolicyLearner(orders.SCHEMA, orders.SCOPE, cfg)
    oracle = TrainingOracle(orders.SCHEMA, orders.SCOPE, callback, budget=a.budget, forbidden_hashes=reserved)
    t0 = time.time()
    artifact = learner.fit([], oracle) if False else learner.fit([_seed_row(callback, rng)], oracle)
    learn_seconds = time.time() - t0

    # 3. promotion on independent calibration labels (never used for learning)
    from orgsim_refiner.types import Observation
    def label_many(xs, split):
        rows = []
        if a.workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(a.workers) as ex:
                got = list(ex.map(callback, xs))
        else:
            got = [callback(x) for x in xs]
        for r in got:
            rows.append(Observation.from_dict({**r.to_dict(), 'split': split}))
        return rows
    cal_rows = label_many(cal, 'validation')
    artifact = learner.calibrate(artifact, cal_rows)
    artifact.save(out / 'POLICY.json')

    # 4. freeze predictions BEFORE obtaining held-out labels
    preds = {}
    for name, xs in (('uniform', uni), ('boundary', bnd)):
        preds[name] = [artifact.predict(x, orders.SCOPE)['decision'] for x in xs]
    write_json(out / 'FROZEN_PREDICTIONS.json', {'policy_sha256': artifact.fingerprint, 'predictions': preds})

    # 5. held-out labels from the oracle, then score
    results = {}
    mistakes = []
    for name, xs in (('uniform', uni), ('boundary', bnd)):
        rows = label_many(xs, 'holdout')
        pairs = list(zip(preds[name], [r.label for r in rows]))
        results[name] = score(pairs)
        for x, p, r in zip(xs, preds[name], rows):
            if p != r.label:
                mistakes.append({'set': name, 'input': x, 'predicted': p, 'actual': r.label, 'evidence': r.evidence})
    both = results['uniform']['correct'] + results['boundary']['correct']
    tot = results['uniform']['cases'] + results['boundary']['cases']
    report = {'oracle': a.oracle, 'seed': a.seed, 'budget': a.budget, 'learning_queries': oracle.calls,
              'learn_seconds': round(learn_seconds, 1), 'converged': bool(artifact.data['evidence']['random_audit_passed']),
              'stop_reason': artifact.data['reason'] if not artifact.ready else learner.stop_reason,
              'promoted': artifact.ready, 'calibration': artifact.data['calibration'],
              'results': results, 'combined_accuracy': round(both / tot, 6),
              'total_false_accepts': results['uniform']['false_accepts'] + results['boundary']['false_accepts'],
              'policy_sha256': artifact.fingerprint, 'mistakes': mistakes[:50],
              'live_api_calls': getattr(locals().get('live'), 'calls', 0)}
    write_json(out / 'REPORT.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('mistakes', 'calibration')}, indent=1))
    return 0


def _seed_row(callback, rng):
    x = orders.SCHEMA.sample(rng)
    return callback(x)


if __name__ == '__main__':
    raise SystemExit(main())
