"""Gate 2G consolidated causal benchmark. Offline-only, three oracle families.

This is a single larger decision gate, not an assertion of real ERP fidelity.
Candidate rules are trained only against oracle discovery calls. Withheld
observations are generated after all models/budgets are frozen.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import argparse
import json
import random
from statistics import mean
from typing import Any

from .gate2a_lab import _SyntheticOracle, public_operations, _make_pool
from .gate2c_counterexamples import _InteractionOracle, _fingerprint, run_cegis_arm, selective_scores
from .gate2f_causal import _uniform_arm
from .interaction_learning import run_interaction_arm, score_interaction

FORMAT='orgsim-gate2g-causal-v1'

class VariableVetoOracle(_SyntheticOracle):
    """Fresh thresholds, same declared scoped veto grammar; evaluator-only."""
    def __init__(self,seed:int):
        super().__init__(seed)
        r=random.Random(seed*3419+9377)
        self.private_veto={
            'orders.validate':(r.choice([3,4,5]),r.choice([50,70,90])),
            'invoices.payment':r.choice([2,3,4]),
            'orders.close':r.choice([4,5,6]),
            'inventory.reserve':(r.choice([4,5,6]),r.choice([3,4,5]))}

    def query(self,op,before,*,phase='discovery'):
        status,after,response=super().query(op,before,phase=phase)
        if status!=200:return status,after,response
        name=op.name
        if name=='orders.validate':
            a,b=self.private_veto[name];deny=before['qty']>=a and before['amount']>=b
        elif name=='invoices.payment':
            deny=before['payment_count']>=self.private_veto[name]
        elif name=='orders.close':
            deny=before['ordered_qty']>=self.private_veto[name]
        elif name=='inventory.reserve':
            a,b=self.private_veto[name];deny=before['qty']>=a and before['allocated']>=b
        else:deny=False
        return (422,before.copy(),{'ok':False,'error':'POLICY_DENIED'}) if deny else (status,after,response)


class OutOfGrammarOracle(_SyntheticOracle):
    """Non-monotone and multi-guard interactions; should trigger uncertainty."""
    def query(self,op,before,*,phase='discovery'):
        status,after,response=super().query(op,before,phase=phase)
        if status!=200:return status,after,response
        n=op.name
        if n=='orders.validate':deny=before['qty'] in (2,5) and before['amount']>=60
        elif n=='invoices.payment':deny=before['payment_count'] in (1,4) and before['payment']>=20
        elif n=='orders.close':deny=before['ordered_qty']==3
        elif n=='inventory.reserve':deny=(before['qty']==2 or before['allocated']==4) and before['stock']>=8
        else:deny=False
        return (422,before.copy(),{'ok':False,'error':'POLICY_DENIED'}) if deny else (status,after,response)


def _score_legacy(score:dict):
    return {'cases':score['cases'],'answered':score['cases']-score['abstentions'],
            'coverage':score['coverage'],'false_accepts':score['false_accepts'],
            'false_rejects':score['false_rejects'],
            'true_accepts':score['accepted_true_positives'],
            'true_rejects':score['true_negatives'],'abstentions':score['abstentions'],
            'positive_truth':score['positive_truth'],
            'positive_recall':score['positive_accept_recall'],
            'effect_exact':score['effect_exact_on_true_accepts'],
            'audited':score['audit_ready']}


def run(*,seeds:int=12,budget:int=48,holdout:int=200,freeze_seed:int=20261210,
        outdir:str|Path='build/gate2g_causal',include_out_of_grammar:bool=True)->dict:
    if not (2<=seeds<=30 and 32<=budget<=80 and 80<=holdout<=500):
        raise ValueError('Invalid experiment dimensions')
    out=Path(outdir)
    if (out/'CAUSAL_FREEZE.json').exists():raise FileExistsError('Already frozen')
    records=[];ops=public_operations()
    families=['known_interaction','variable_threshold']
    if include_out_of_grammar:families.append('outside_grammar')
    for family in families:
        for si in range(seeds):
            for oi,op in enumerate(ops):
                seed=freeze_seed + 5003*si + 17*oi
                oracle={'known_interaction':_InteractionOracle,
                        'variable_threshold':VariableVetoOracle,
                        'outside_grammar':OutOfGrammarOracle}[family](seed)
                # Public contract and seed same for each exploration arm.
                # Baselines and hybrid are trained BEFORE any evaluation call.
                # A finite, tiny contract domain cannot support 48 different
                # business-state probes; share a smaller ceiling, no padding.
                domain=1
                for k,vals in op.fields.items():
                    if k!='record_id':domain*=len(vals)
                effective=32 if domain<2*budget else budget
                uniform=_uniform_arm(op,oracle,effective,seed)
                cegs=run_cegis_arm(op,oracle,budget=effective,seed=seed)
                hybrid=run_interaction_arm(op,oracle,budget=effective,seed=seed)
                arms={'uniform':uniform,'counterexample_conjunction':cegs,
                      'interaction_audited':hybrid}
                signatures={_fingerprint(op,row['before']) for arm in arms.values()
                            for row in arm['observations']}
                pool=_make_pool(op,seed*2+127,n=holdout+650)
                unique=[x for x in pool if _fingerprint(op,x) not in signatures]
                cases=unique[:holdout]
                if len(cases)<holdout:
                    seen={json.dumps(x,sort_keys=True) for x in cases}
                    for s in pool:
                        key=json.dumps(s,sort_keys=True)
                        if key not in seen:
                            cases.append(s);seen.add(key)
                        if len(cases)>=holdout:break
                if len(cases)!=holdout:raise RuntimeError('Insufficient heldout states')
                new_states=sum(_fingerprint(op,x) not in signatures for x in cases)
                truth=[(s,*oracle.query(op,s,phase='evaluation')[:2]) for s in cases]
                for name,r in arms.items():
                    scores=(score_interaction(op,r['model'],truth)
                            if name=='interaction_audited' else
                            _score_legacy(selective_scores(op,r['model'],truth)))
                    records.append({'family':family,'seed':si,'operation':op.name,
                                    'arm':name,'budget_max':budget,
                                    'observations_used':len(r['observations']),
                                    'novel_business_states':new_states,**scores})
    results=[]
    for family in families:
        for name in ('uniform','counterexample_conjunction','interaction_audited'):
            group=[x for x in records if x['family']==family and x['arm']==name]
            total=sum(x['cases'] for x in group)
            true_pos=sum(x['positive_truth'] for x in group)
            tps=sum(x['true_accepts'] for x in group)
            fp=sum(x['false_accepts'] for x in group)
            fn=sum(x['false_rejects'] for x in group)
            abst=sum(x['abstentions'] for x in group)
            passed_effects=[x for x in group if x['true_accepts']>0]
            # No effect correctness claim where the learner abstained entirely.
            effect=(round(sum(x['true_accepts']*x['effect_exact'] for x in passed_effects)/tps,5)
                    if tps else None)
            result={'family':family,'arm':name,'cases':total,
                    'observations_total':sum(x['observations_used'] for x in group),
                    'mean_observations_per_operation':round(mean(x['observations_used'] for x in group),2),
                    'novel_business_state_fraction':round(sum(x['novel_business_states'] for x in group)/total,5),
                    'coverage':round((total-abst)/total,5),
                    'true_positive_recall':round(tps/max(1,true_pos),5),
                    'false_accepts':fp,'false_rejects':fn,
                    'effect_exact_on_true_accepts':effect,
                    'audited_ready':sum(bool(x['audited']) for x in group),
                    'operation_seed_pairs':len(group)}
            result['gate']='PASS_SCOPED_SYNTHETIC_ONLY' if (
                result['coverage']>=.8 and result['true_positive_recall']>=.8
                and fp==0 and effect==1.0 and result['audited_ready']>=int(.8*len(group))) else 'FAIL'
            results.append(result)
    report={'format':FORMAT,'oracle_is_synthetic':True,'real_ERP_behavior_verified':False,
            'model_in_loop':False,'live_reads':0,'live_writes':0,
            'seeds':seeds,'budget_max_per_operation':budget,'holdout_size':holdout,
            'freeze_seed':freeze_seed,'results':results,
            'hypothesis_class':'monotone base conjunction + at most one unary/pair upper veto',
            'known_limitations':['Same oracle family as development is not independent confirmation.',
                  'Variable thresholds are fresh but remain in the scoped hypothesis class.',
                  'Outside-grammar tests are deliberately non-monotone or multi-interaction.',
                  'The action catalog exposes all base critical axes and derived arithmetic.',
                  'Zero observed false accepts in synthetic data is not a live safety guarantee.',
                  'Not a causal Dolibarr verification or actual LLM experiment.']}
    out.mkdir(parents=True,exist_ok=True)
    pb=(json.dumps(report,sort_keys=True,indent=2)+'\n').encode()
    rb=(json.dumps(records,sort_keys=True,indent=2)+'\n').encode()
    (out/'causal_report.json').write_bytes(pb)
    (out/'per_operation.json').write_bytes(rb)
    (out/'CAUSAL_FREEZE.json').write_bytes((json.dumps({'format':FORMAT+'-freeze',
               'report_sha256':sha256(pb).hexdigest(),'rows_sha256':sha256(rb).hexdigest(),
               'live_reads':0,'live_writes':0,'independent_software_holdout':False},
               indent=2,sort_keys=True)+'\n').encode())
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description='Consolidated synthetic causal audit; offline only')
    parser.add_argument('--seeds',type=int,default=12)
    parser.add_argument('--budget',type=int,default=48)
    parser.add_argument('--holdout',type=int,default=200)
    parser.add_argument('--freeze-seed',type=int,default=20261210)
    parser.add_argument('--out',default='build/gate2g_causal')
    args=parser.parse_args(argv)
    result=run(seeds=args.seeds,budget=args.budget,holdout=args.holdout,
               freeze_seed=args.freeze_seed,outdir=args.out)
    print(json.dumps({'results':result['results'],'out':args.out,'live_writes':0},indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
