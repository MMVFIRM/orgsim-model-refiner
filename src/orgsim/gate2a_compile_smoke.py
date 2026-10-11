"""Test a *learned* synthetic workflow in the actual OrgSim runtime.

Single predetermined seed and discovery budget. Evidence is synthetic only.
The exporter refuses incomplete effects. No real Dolibarr API access.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .gate2a_lab import (public_operations, _SyntheticOracle, _make_pool, run_arm,
                         predict_model, features_for)
from .gate2a_export import model_to_manifest
from .compiler import compile_spec
from .engine import SimulationEngine, SimError


def smoke(*, seed:int=20261231, budget:int=20, withheld:int=160,
          out:str|Path='build/gate2a_compiled_smoke')->dict:
    target=next(o for o in public_operations() if o.name=='orders.validate')
    oracle=_SyntheticOracle(seed)
    pool=_make_pool(target,seed+1000,n=240)
    observed=run_arm(target,oracle,pool,arm='franzen',budget=budget,seed=seed)
    model=observed['model']
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=model_to_manifest(target,model)
    content=json.dumps(manifest,indent=2)+'\n'
    (out/'learned_unreviewed_manifest.json').write_text(content)
    compile_spec(manifest,out/'compiled',allow_draft=True)
    # Entirely new independently seeded synthetic cases, withheld from learning.
    eval_pool=_make_pool(target,seed*3+999,n=withheld+100)
    seen={tuple(s.items()) for s in pool}
    eval_states=[s for s in eval_pool if tuple(s.items()) not in seen][:withheld]
    if len(eval_states)<withheld:raise AssertionError('Insufficient disjoint tests')
    sim=SimulationEngine(manifest,allow_draft=True)
    tp=tn=fp=fn=effects_ok=0
    transition_error=0
    try:
        for case in eval_states:
            live_code,live_after,_=oracle.query(target,case,phase='evaluation')
            created=sim.call('lab.cases.create',case,role='test_agent')
            rid=str(created['result']['id'])
            try:
                sim.call('lab.workflow.run_action',{'id':rid},role='test_agent')
                predicted_accept=True
            except SimError as err:
                predicted_accept=False
                if err.code not in {'POLICY_DENIED'}:transition_error+=1
            sim_record=sim.get('lab','cases',rid)
            actual_accept=live_code==200
            tp+=int(predicted_accept and actual_accept)
            tn+=int(not predicted_accept and not actual_accept)
            fp+=int(predicted_accept and not actual_accept)
            fn+=int(not predicted_accept and actual_accept)
            if predicted_accept and actual_accept:
                effects_ok+=int(all(sim_record[k]==live_after[k] and type(sim_record[k])==type(live_after[k])
                                   for k in target.outcome_fields))
            if not predicted_accept:
                assert all(sim_record[k]==case[k] for k in target.outcome_fields),'Rejected action mutated state'
    finally:sim.close()
    metrics={'experiment':'SYNTHETIC_ORACLE_NOT_DOLIBARR',
             'seed':seed,'discovery_queries':budget,'successful_discovery_queries':observed['positives'],
             'heldout_cases':withheld,
             'accepted_truth':tp+fn,'rejected_truth':tn+fp,
             'correct_accept':tp,'correct_reject':tn,'false_accept':fp,'false_reject':fn,
             'heldout_outcome_accuracy':round((tp+tn)/withheld,4),
             'safety_gate':'FAIL_FALSE_ACCEPTS' if fp else 'NO_FALSE_ACCEPTS_IN_THIS_SMALL_TEST',
             'balanced_accuracy':round((tp/max(1,tp+fn)+tn/max(1,tn+fp))/2,4),
             'effects_correct_given_true_positives':effects_ok,'true_positives':tp,
             'runtime_unexpected_errors':transition_error,'provisional_policy':True,
             'compiled_manifest_sha256':sha256(content.encode()).hexdigest(),
             'live_dolibarr_calls':0,
             'note':'Checks whether a discovered rule can be executed and how it predicts a synthetic oracle; not a vendor reconstruction.'}
    (out/'runtime_differential_report.json').write_text(json.dumps(metrics,indent=2)+'\n')
    return metrics


def main():
    import argparse
    p=argparse.ArgumentParser(description='Compile and evaluate one learned synthetic OrgSim workflow')
    p.add_argument('--out',default='build/gate2a_compiled_smoke')
    p.add_argument('--seed',type=int,default=20261231)
    args=p.parse_args()
    print(json.dumps(smoke(seed=args.seed,out=args.out),indent=2))

if __name__=='__main__':main()
