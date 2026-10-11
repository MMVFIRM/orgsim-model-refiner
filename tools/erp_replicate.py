"""Learn an ERP action (decision AND state change), freeze, then replicate it on held-out orders.

  python tools/erp_replicate.py --oracle twin --seed 1 --out build/rep_twin_1
  python tools/erp_replicate.py --oracle live --seed 1 --out build/rep_live_1 --api-key-env DOLAPIKEY

For every held-out order the frozen model predicts (a) APPROVE/REJECT and (b) the exact
resulting state (order status, stock of each product). The action is then performed on the
ERP and the observed result compared. "Replication accuracy" counts a case correct only if
the decision AND every state field match exactly.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from erp_experiment import uniform_inputs, boundary_inputs, dedupe
from orgsim_refiner.erp import orders
from orgsim_refiner.expr_learner import ExprPolicyLearner, ExprConfig
from orgsim_refiner.evidence import TrainingOracle
from orgsim_refiner.effects import EffectModel
from orgsim_refiner.types import Observation
from orgsim_refiner.util import digest, write_json


def twin_with_state(x):
    obs = orders.twin_callback(x)
    tracked_a = x['kind_a'] == 'good'
    before = {'order_status': 0, 'stock_a': x['stock_a'] if tracked_a else 0, 'stock_b': x['stock_b'] if x['qty_b'] > 0 else 0}
    after = dict(before)
    if obs.label == 'APPROVE':
        after['order_status'] = 1
        if tracked_a:
            after['stock_a'] = x['stock_a'] - x['qty_a1'] - x['qty_a2']
        if x['qty_b'] > 0:
            after['stock_b'] = x['stock_b'] - x['qty_b']
    action = {k: x[k] for k in ('kind_a', 'qty_a1', 'qty_a2', 'qty_b')}
    return Observation(obs.scope, x, obs.label, provenance='SIMULATOR_OBSERVED', before=before, after=after, action=action)


def before_of(x):
    return {'order_status': 0, 'stock_a': x['stock_a'] if x['kind_a'] == 'good' else 0, 'stock_b': x['stock_b'] if x['qty_b'] > 0 else 0}


def action_of(x):
    return {k: x[k] for k in ('kind_a', 'qty_a1', 'qty_a2', 'qty_b')}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--oracle', choices=['twin', 'live'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--budget', type=int, default=4000)
    ap.add_argument('--holdout', type=int, default=150)
    ap.add_argument('--calibration', type=int, default=120)
    ap.add_argument('--url', default='http://127.0.0.1:8189')
    ap.add_argument('--api-key-env', default='DOLAPIKEY')
    ap.add_argument('--clone-permit', help='Expiring private permit tied to disposable clone marker; required for live mode')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit('Output directory exists; results are immutable.')
    out.mkdir(parents=True)
    rng = random.Random(a.seed * 104729 + 7)
    if a.oracle == 'live':
        if not a.clone_permit:
            raise SystemExit('Live oracle disabled without --clone-permit; never use production')
        from orgsim_refiner.erp.dolibarr_live import DolibarrLive
        key = os.environ.get(a.api_key_env)
        if not key:
            raise SystemExit(f'Set {a.api_key_env} in the environment.')
        live = DolibarrLive(a.url, key, capture_state=True, clone_permit=a.clone_permit)
        callback = live.callback
    else:
        live = None
        callback = twin_with_state

    uni = dedupe(uniform_inputs(rng, a.holdout * 2), [])[:a.holdout]
    bnd = dedupe(dedupe(boundary_inputs(rng, a.holdout * 3), [digest(x) for x in uni]), [])[:a.holdout]
    cal = dedupe(uniform_inputs(rng, a.calibration * 3), [])
    reserved = {digest(x) for x in uni + bnd}
    cal = [x for x in cal if digest(x) not in reserved][:a.calibration]
    reserved |= {digest(x) for x in cal}

    learner = ExprPolicyLearner(orders.SCHEMA, orders.SCOPE, ExprConfig(seed=a.seed + 5000))
    oracle = TrainingOracle(orders.SCHEMA, orders.SCOPE, callback, budget=a.budget, forbidden_hashes=reserved)
    first = orders.SCHEMA.sample(rng)
    while digest(first) in reserved:
        first = orders.SCHEMA.sample(rng)
    t0 = time.time()
    art = learner.fit([callback(first)], oracle)
    cal_rows = [Observation.from_dict({**callback(x).to_dict(), 'split': 'validation'}) for x in cal]
    art = learner.calibrate(art, cal_rows)
    train_rows = [r for r in learner.evidence.rows if r.before is not None]
    effects = EffectModel(orders.SCOPE).fit(train_rows, cal_rows)
    art.save(out / 'POLICY.json')
    write_json(out / 'EFFECTS.json', effects.to_dict())
    learn_seconds = time.time() - t0

    # ---- freeze predictions (decision + exact post-state) before touching held-out orders
    def predict(x):
        d = art.predict(x, orders.SCOPE)['decision']
        before = before_of(x)
        if d == 'APPROVE':
            after = effects.predict(before, action_of(x), orders.SCOPE)
            return d, (after if after is not None else None)
        return d, (before if d == 'REJECT' else None)
    frozen = {name: [predict(x) for x in xs] for name, xs in (('uniform', uni), ('boundary', bnd))}
    write_json(out / 'FROZEN_PREDICTIONS.json', {'policy_sha256': art.fingerprint, 'effects_ready': effects.ready,
                                                  'predictions': {k: [[d, s] for d, s in v] for k, v in frozen.items()}})

    # ---- replicate on the ERP and compare
    results, mistakes = {}, []
    for name, xs in (('uniform', uni), ('boundary', bnd)):
        decision_ok = state_ok = full_ok = approvals = 0
        for x, (pd, ps) in zip(xs, frozen[name]):
            obs = callback(x)
            ok_d = pd == obs.label
            actual_state = obs.after  # Use actual observed state on BOTH success and refusal; never assume unchanged
            ok_s = ps is not None and ps == actual_state
            decision_ok += ok_d
            state_ok += ok_s
            full_ok += ok_d and ok_s
            approvals += obs.label == 'APPROVE'
            if not (ok_d and ok_s):
                mistakes.append({'set': name, 'input': x, 'predicted': [pd, ps], 'actual': [obs.label, actual_state]})
        n = len(xs)
        results[name] = {'cases': n, 'approvals': approvals, 'decision_correct': decision_ok, 'state_correct': state_ok,
                         'replication_exact': full_ok, 'replication_accuracy': round(full_ok / n, 6)}
    tot = sum(r['cases'] for r in results.values())
    ex = sum(r['replication_exact'] for r in results.values())
    report = {'oracle': a.oracle, 'seed': a.seed, 'learning_queries': oracle.calls, 'learn_seconds': round(learn_seconds, 1),
              'policy_promoted': art.ready, 'effects_ready': effects.ready, 'effects': effects.certificate,
              'results': results, 'replication_accuracy': round(ex / tot, 6), 'mistakes': mistakes[:40],
              'effect_programs': {k: v[0] for k, v in effects.programs.items() if v}}
    write_json(out / 'REPORT.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('mistakes', 'effect_programs', 'effects')}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
