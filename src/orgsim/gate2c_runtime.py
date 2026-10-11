"""Replay learned certified synthetic transitions through the real OrgSim runtime."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import argparse
import json

from .gate2a_lab import _SyntheticOracle,_make_pool,public_operations
from .gate2c_counterexamples import run_cegis_arm,_fingerprint
from .gate2c_export import certified_synthetic_manifest
from .engine import SimulationEngine,SimError
from .compiler import compile_spec


def replay(*,seeds:tuple[int,...]=(20271019,20271020),budget:int=20,
           heldout:int=160,outdir:str|Path='build/gate2c_runtime')->dict:
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    rows=[];total=0
    for seed in seeds:
        oracle=_SyntheticOracle(seed)
        for index,op in enumerate(public_operations()):
            discovery=run_cegis_arm(op,oracle,budget=budget,seed=seed*31+index)
            model=discovery['model']
            if not model['audit']['ready_for_offline_execution']:
                rows.append({'seed':seed,'operation':op.name,'certified_for_synthetic_replay':False,
                             'reason':'audit_not_ready','cases':0})
                continue
            manifest=certified_synthetic_manifest(op,model)
            name=f'{seed}_{op.name.replace(".","_")}'
            path=out/name;path.mkdir(parents=True,exist_ok=True)
            content=json.dumps(manifest,sort_keys=True,indent=2)+'\n'
            (path/'learned_synthetic_unreviewed.json').write_text(content)
            compile_spec(manifest,path/'compiled',allow_draft=True)
            observed={_fingerprint(op,r['before']) for r in discovery['observations']}
            pool=_make_pool(op,seed*77+index*1007+7,n=heldout+400)
            withheld=[s for s in pool if _fingerprint(op,s) not in observed][:heldout]
            if len(withheld)<heldout:raise AssertionError('Holdout not disjoint from discovery')
            engine=SimulationEngine(manifest,allow_draft=True)
            tp=tn=fp=fn=effects_correct=unexpected=0
            try:
                for case in withheld:
                    status,truth_after,_=oracle.query(op,case,phase='evaluation')
                    created=engine.call('lab.cases.create',case,role='test_agent')
                    rec_id=str(created['result']['id'])
                    try:
                        engine.call('lab.workflow.run_action',{'id':rec_id},role='test_agent')
                        got=True
                    except SimError as err:
                        got=False
                        if err.code!='POLICY_DENIED':unexpected+=1
                    actual=status==200
                    after=engine.get('lab','cases',rec_id)
                    tp+=int(got and actual);tn+=int(not got and not actual)
                    fp+=int(got and not actual);fn+=int(not got and actual)
                    if got and actual:
                        effects_correct+=int(all(after[f]==truth_after[f] and type(after[f])==type(truth_after[f])
                            for f in op.outcome_fields))
                    elif not got:
                        assert all(after[f]==case[f] for f in op.outcome_fields), 'Rejected operation changed state'
            finally:
                engine.close()
            total+=heldout
            rows.append({'seed':seed,'operation':op.name,'certified_for_synthetic_replay':True,
                         'manifest_sha256':sha256(content.encode()).hexdigest(),
                         'cases':heldout,'true_accepts':tp,'true_rejects':tn,
                         'false_accepts':fp,'false_rejects':fn,
                         'effects_correct':effects_correct,
                         'unexpected_runtime_errors':unexpected})
    summary={'format':'orgsim-gate2c-compiled-v1','oracle':'SYNTHETIC_ONLY',
             'cases':total,'scored_operations':sum(x['certified_for_synthetic_replay'] for x in rows),
             'false_accepts':sum(x.get('false_accepts',0) for x in rows),
             'false_rejects':sum(x.get('false_rejects',0) for x in rows),
             'effects_correct_on_accepted_truth':sum(x.get('effects_correct',0) for x in rows),
             'true_accepts':sum(x.get('true_accepts',0) for x in rows),
             'unexpected_runtime_errors':sum(x.get('unexpected_runtime_errors',0) for x in rows),
             'live_dolibarr_calls':0,'all_manifest_reviewed_false':True,
             'rows':rows}
    summary['pass_offline_synthetic']=(total==len(seeds)*len(public_operations())*heldout
        and summary['false_accepts']==0 and summary['false_rejects']==0
        and summary['effects_correct_on_accepted_truth']==summary['true_accepts']
        and summary['unexpected_runtime_errors']==0)
    (out/'runtime_report.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main(argv=None)->int:
    p=argparse.ArgumentParser(description='Replay synthetic, unreviewed learned workflows locally only')
    p.add_argument('--out',default='build/gate2c_runtime')
    p.add_argument('--budget',type=int,default=20)
    p.add_argument('--cases',type=int,default=160)
    a=p.parse_args(argv)
    doc=replay(budget=a.budget,heldout=a.cases,outdir=a.out)
    print(json.dumps({k:v for k,v in doc.items() if k!='rows'},indent=2))
    return 0 if doc['pass_offline_synthetic'] else 2
if __name__=='__main__':raise SystemExit(main())
