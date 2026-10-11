"""Prespecified resource-limit and binding controls, not a general safety proof."""
from __future__ import annotations
from pathlib import Path
from .types import Scope
from .learner import ActivePolicyLearner,LearnConfig
from .evidence import TrainingOracle,EvidenceLedger
from .util import write_json,digest

def run_controls(out:Path,seed:int):
    from .benchmark import SyntheticTenant,SCHEMA,dataset,metrics
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    spec={'format':'orgsim-refiner-resource-controls-v1','seed':seed,'tenants':2,
          'family':'diagonal','training_queries_per_tenant':64,'max_cells':64,
          'claim':'Deliberately inadequate query/cell budgets: report failure/abstention, do not claim automatic detection of arbitrary hidden rules.'}
    write_json(out/'CONTROL_FREEZE.json',spec)
    results=[]
    for i in range(2):
        oracle_owner=SyntheticTenant(seed+i*313,'diagonal')
        rows=dataset(oracle_owner,seed+i*17,{'train':16,'validation':32,'holdout':64})
        directory=out/f'control_{i+1}';directory.mkdir()
        ledger=EvidenceLedger(directory/'EVIDENCE.sqlite')
        sh=write_json(directory/'SEALED_EVALUATOR.json',{'oracle':oracle_owner.sealed_spec(),
                     'splits':{k:[r.to_dict() for r in v] for k,v in rows.items()}})
        ledger.append('data_sealed',{'sha256':sh})
        learner=ActivePolicyLearner(SCHEMA,oracle_owner.scope,LearnConfig(seed=seed+i,max_cells=64,audit_samples=16))
        oracle=TrainingOracle(SCHEMA,oracle_owner.scope,oracle_owner.observe,budget=64,ledger=ledger,
                  forbidden_hashes={r.sample_hash for split in ('validation','holdout') for r in rows[split]})
        policy=learner.calibrate(learner.fit(rows['train'],oracle),rows['validation'])
        policy.save(directory/'POLICY.json')
        predictions=[policy.predict(r.sample,r.scope)['decision'] for r in rows['holdout']]
        foreign=Scope('another-tenant',policy.scope.operation,policy.scope.configuration)
        foreign_predictions=[policy.predict(r.sample,foreign)['decision'] for r in rows['holdout']]
        h=write_json(directory/'PREDICTIONS.json',{'local':predictions,'foreign':foreign_predictions})
        ledger.append('predictions_frozen',{'sha256':h})
        result={'ready':policy.ready,'reason':policy.data['reason'],'training_queries':oracle.calls,
                'local':metrics(rows['holdout'],predictions),'foreign_scope_abstentions':sum(p=='UNKNOWN' for p in foreign_predictions),
                'evidence':ledger.verify()}
        write_json(directory/'RESULT.json',result);results.append(result)
    report={'spec_sha256':digest(spec),'controls':results,'new_model_calls':0,'dolibarr_calls':0,
            'warning':'Finite empirical controls. Sparse or omitted hidden effects and policy drift can still escape detection.'}
    write_json(out/'REPORT.json',report)
    return report
