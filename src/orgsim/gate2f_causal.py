"""Gate 2F-B: offline, matched-budget, hidden synthetic ERP causal learning.

The oracle is evaluator-only. Both arms use the SAME audited rule/effect learner;
only selection of discovery interventions differs. Held-out states are generated
and scored AFTER both arms have completed their exploration. NOT live Dolibarr.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any

from .gate2a_lab import (_SyntheticOracle, public_operations, reference_state,
                         _make_pool, one_factor_candidates)
from .gate2c_counterexamples import (_InteractionOracle, _fingerprint, _observation,
                                    fit_audited_model, run_cegis_arm, selective_scores)

FORMAT = 'orgsim-gate2f-synthetic-causal-v1'


def _uniform_arm(op, oracle, budget: int, seed: int) -> dict:
    """Common positive anchor, then uniformly sampled public counterfactual pool.

    No oracle outcomes, thresholds or hidden state are consulted in selecting the
    next query. The fit procedure is identical to the active arm.
    """
    rng = random.Random(seed)
    anchor = reference_state(op)
    proposals = one_factor_candidates(op, anchor)
    proposals += _make_pool(op, seed + 876, n=max(250, budget * 8))
    rng.shuffle(proposals)
    seen = {_fingerprint(op, anchor)}
    chosen = [anchor]
    for state in proposals:
        sig = _fingerprint(op, state)
        if sig in seen:
            continue
        seen.add(sig)
        chosen.append(state)
        if len(chosen) == budget:
            break
    if len(chosen) != budget:
        raise RuntimeError('Insufficient unique exploration states')
    rows = [_observation(op, oracle, state) for state in chosen]
    return {'queries': len(rows), 'observations': rows,
            'model': fit_audited_model(op, rows),
            'positive_examples': sum(bool(x['accepted']) for x in rows)}


def run(*, seeds: int = 8, budgets: tuple[int, ...] = (20, 32),
        holdout: int = 160, freeze_seed: int = 20261028,
        outdir: str | Path = 'build/gate2f_causal') -> dict:
    if not (2 <= seeds <= 40 and 80 <= holdout <= 1000 and budgets
            and all(20 <= x <= 80 for x in budgets) and len(set(budgets)) == len(budgets)):
        raise ValueError('Invalid experiment dimensions')
    out = Path(outdir)
    if (out / 'CAUSAL_FREEZE.json').exists():
        raise FileExistsError('Results are frozen. Choose a new output directory.')
    operations = public_operations()
    records = []
    for family in ('simple', 'interaction'):
        for si in range(seeds):
            for oi, op in enumerate(operations):
                oracle = (_SyntheticOracle(freeze_seed + si) if family == 'simple'
                          else _InteractionOracle(freeze_seed + si))
                trainings = {}
                observed_signatures = set()
                for budget in budgets:
                    # Common anchor and budget; the intervention choice is the
                    # independent variable. The evaluator returns observations.
                    active = run_cegis_arm(op, oracle, budget=budget,
                                           seed=freeze_seed + si * 991 + oi * 31)
                    uniform = _uniform_arm(op, oracle, budget,
                                           seed=freeze_seed + si * 991 + oi * 31)
                    trainings[budget] = {'uniform': uniform, 'counterexample': active}
                    for arm in (uniform, active):
                        observed_signatures.update(_fingerprint(op, x['before'])
                                                   for x in arm['observations'])
                pool = _make_pool(op, freeze_seed*3 + si*1607 + oi*19,
                                  n=holdout+700)
                unseen = [s for s in pool if _fingerprint(op,s) not in observed_signatures]
                evaluation = unseen[:holdout]
                if len(evaluation) < holdout:
                    # Finite state domains cannot always supply 160 unique
                    # feature combinations. Report exact novelty explicitly.
                    seen_ids = {json.dumps(s, sort_keys=True) for s in evaluation}
                    for s in pool:
                        k=json.dumps(s,sort_keys=True)
                        if k not in seen_ids:
                            evaluation.append(s);seen_ids.add(k)
                        if len(evaluation) >= holdout:
                            break
                if len(evaluation) != holdout:
                    raise RuntimeError('Could not construct bounded synthetic holdout')
                novel = sum(_fingerprint(op,s) not in observed_signatures for s in evaluation)
                truth = [(state, *oracle.query(op,state,phase='evaluation')[:2])
                         for state in evaluation]
                for budget, arms in trainings.items():
                    for arm_name, training in arms.items():
                        score = selective_scores(op, training['model'], truth)
                        records.append({'family':family,'seed':si,'operation':op.name,
                                        'arm':arm_name,'budget':budget,
                                        'discovery_queries':training['queries'],
                                        'positive_discovery':training['positive_examples'],
                                        'novel_business_states':novel,**score})
    aggregate = []
    for family in ('simple','interaction'):
        for budget in budgets:
            for arm in ('uniform','counterexample'):
                group = [r for r in records if r['family']==family and r['budget']==budget
                         and r['arm']==arm]
                total = sum(r['cases'] for r in group)
                positive = sum(r['positive_truth'] for r in group)
                tp = sum(r['accepted_true_positives'] for r in group)
                false_accepts = sum(r['false_accepts'] for r in group)
                answered = total-sum(r['abstentions'] for r in group)
                effects_ok = sum(round(r['effect_exact_on_true_accepts'] * r['accepted_true_positives'])
                                 for r in group if r['effect_exact_on_true_accepts'] is not None)
                effects_total = tp
                a={'oracle_family':family,'budget':budget,'arm':arm,
                   'cases':total,'operation_seed_pairs':len(group),
                   'discovery_observations':sum(r['discovery_queries'] for r in group),
                   'false_accepts':false_accepts,
                   'false_rejects':sum(r['false_rejects'] for r in group),
                   'coverage':round(answered/total,5),
                   'true_positive_recall':round(tp/max(1,positive),5),
                   'effect_exact_when_correctly_accepted':(round(effects_ok/effects_total,5)
                                                            if effects_total else None),
                   'effects_scored':effects_total,
                   'audited_ready':sum(bool(r['audit_ready']) for r in group),
                   'novel_state_fraction':round(sum(r['novel_business_states'] for r in group)/total,5)}
                a['causal_gate']='PASS_SYNTHETIC_ONLY' if (
                    a['false_accepts']==0 and a['coverage']>=.8
                    and a['true_positive_recall']>=.8
                    and a['effect_exact_when_correctly_accepted']==1.0
                    and a['audited_ready'] >= int(.8*len(group))) else 'FAIL'
                aggregate.append(a)
    result = {'format':FORMAT,'phase':'offline_causal','live_calls':0,
              'live_writes':0,'llm_tested':False,'oracle_is_synthetic':True,
              'seeds':seeds,'freeze_seed':freeze_seed,'budgets':list(budgets),
              'heldout_per_operation_seed':holdout,
              'operation_types':len(operations), 'results':aggregate,
              'known_limitations':[
                  'The learner knows candidate critical axes and monotonicity assumptions.',
                  'The synthetic oracles have finite predefined domains, not production complexity.',
                  'Some heldout cases have fresh IDs but already seen feature combinations.',
                  'No cross-system model transfer, no real model API calls, no live mutation permission.',
                  'A failed interaction coverage gate must remain a failure even with zero false accepts.']}
    out.mkdir(parents=True,exist_ok=True)
    by = (json.dumps(records,sort_keys=True,indent=2)+'\n').encode()
    summary = (json.dumps(result,sort_keys=True,indent=2)+'\n').encode()
    (out/'per_operation.json').write_bytes(by)
    (out/'causal_report.json').write_bytes(summary)
    freeze = {'format':FORMAT+'-freeze','phase':'offline_causal',
              'report_sha256':sha256(summary).hexdigest(),
              'records_sha256':sha256(by).hexdigest(),
              'live_writes':0,'independent_software_holdout':False}
    (out/'CAUSAL_FREEZE.json').write_text(json.dumps(freeze,sort_keys=True,indent=2)+'\n')
    return result


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description='Gate 2F synthetic causal benchmark (no live I/O)')
    p.add_argument('--seeds',type=int,default=8)
    p.add_argument('--budgets',nargs='+',type=int,default=[20,32])
    p.add_argument('--holdout',type=int,default=160)
    p.add_argument('--freeze-seed',type=int,default=20261028)
    p.add_argument('--out',default='build/gate2f_causal')
    a=p.parse_args(argv)
    result=run(seeds=a.seeds,budgets=tuple(a.budgets),holdout=a.holdout,
               freeze_seed=a.freeze_seed,outdir=a.out)
    print(json.dumps({'offline_only':True,'live_writes':0,
                      'results':result['results'],'out':a.out},indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())
