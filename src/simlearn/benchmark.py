"""Single integrated benchmark: accurate decisions + exact causal effects.

Train, router and validation are the only model-selection inputs. Test/stress
are read exactly once during final scoring. All baselines share identical data.
"""
from __future__ import annotations
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import statistics

from .dataset import load_outcomes
from .oracle import OPS,canonical,digest
from .learning import (EffectLearner,fit_models,predict_tabular,predict_generic,
                       predict_retrieval)


def _score(rows:list[dict],pred:list[str],effects:EffectLearner|None=None)->dict:
    if len(rows)!=len(pred):raise ValueError('Predictions and cases misaligned')
    n=len(rows)
    tp=tn=fp=fn=effect_exact=0
    by_op=defaultdict(lambda:dict(n=0,correct=0,fp=0,fn=0,accepted=0,effects=0))
    for row,choice in zip(rows,pred):
        op=row['operation'];item=by_op[op];item['n']+=1
        correct=choice==row['decision']
        item['correct']+=correct
        if row['decision']=='ACCEPT':item['accepted']+=1
        if choice=='ACCEPT' and row['decision']=='ACCEPT':
            tp+=1
            if effects is not None and effects.predict(row['sample'],choice)==row['changed']:
                effect_exact+=1;item['effects']+=1
        elif choice=='REJECT' and row['decision']=='REJECT':tn+=1
        elif choice=='ACCEPT':fp+=1;item['fp']+=1
        else:fn+=1;item['fn']+=1
    assert tp+tn+fp+fn==n
    result={'cases':n,'accuracy':round((tp+tn)/n,6),
            'balanced_accuracy':round((tp/(tp+fn)+tn/(tn+fp))/2,6) if (tp+fn and tn+fp) else None,
            'false_accepts':fp,'false_rejects':fn,
            'false_accept_rate':round(fp/(fp+tn),6) if fp+tn else None,
            'accept_recall':round(tp/(tp+fn),6) if tp+fn else None,
            'accept_precision':round(tp/(tp+fp),6) if tp+fp else None,
            'reject_recall':round(tn/(tn+fp),6) if tn+fp else None,
            'decision_coverage':1.0,
            'effect_exact_on_true_accepts':round(effect_exact/tp,6) if tp and effects is not None else None,
            'end_to_end_exact_accuracy':round((tn+effect_exact)/n,6) if effects is not None else None,
            'per_operation':{k:{**v,'accuracy':round(v['correct']/v['n'],5)} for k,v in sorted(by_op.items())}}
    return result


def _input_set(rows):return {digest(x['sample']) for x in rows}


def run_environment(data_dir:str|Path,outdir:str|Path,*,seed:int=42)->dict:
    data_dir=Path(data_dir);out=Path(outdir)
    if out.exists():raise FileExistsError(f'Output already exists: {out}')
    splits=load_outcomes(data_dir/'outcome.jsonl')
    disjoint=set()
    for name,rows in splits.items():
        sig=_input_set(rows)
        if disjoint & sig:raise RuntimeError('Leakage: state appears across partitions')
        disjoint|=sig
    if any(len({r['group'] for r in rows})!=len(rows) for rows in splits.values()):
        raise ValueError('Duplicate scenario groups within split')
    train=splits['train'];val=splits['validation']
    models,selections=fit_models(train,val,seed=seed)
    effects=EffectLearner().fit(train,val)
    # Tune retrieval K using validation, never test outcomes.
    ks={}
    for k in (1,3,5,9):
        guess=predict_retrieval(train,val,k=k)
        ks[k]=_score(val,guess)['accuracy']
    k_best=max(ks,key=lambda k:(ks[k],-k))
    # Entire labeled evidence is frozen before any test scoring. No refit.
    results={}
    for split in ('test','stress'):
        rows=splits[split]
        arms={
            'always_reject':['REJECT']*len(rows),
            'public_guards':[predict_generic(x) for x in rows],
            'model_refiner_retrieval':predict_retrieval(train,rows,k=k_best),
            'tabular_simtrained':predict_tabular(models,rows),
        }
        results[split]={name:_score(rows,preds,effects) for name,preds in arms.items()}
    source=json.loads((data_dir/'DATASET_FREEZE.json').read_text(encoding='utf-8'))
    report={
        'format':'orgsim-model-refiner-simlearn-v1-offline-benchmark',
        'oracle':'actual_OrgSim_SQLite_workflow_interpreter',
        'model_refiner_retrieval_source':'model_refiner.suite.runtime.Memory',
        'actual_llm_calls':0,'model_refiner_full_fit_run':False,
        'model_refiner_native_aw_tested':False,
        'simulator_ground_truth_is_not_dolibarr':True,'model_source_manifest_not_supplied_to_predictors':True,
        'independent_erp_holdout':False,
        'training_examples':len(train),'router_examples':len(splits['router']),
        'validation_examples':len(val),'test_examples':len(splits['test']),
        'stress_examples':len(splits['stress']),
        'training_oracle_calls_including_discarded':source['oracle_queries_total'],
        'train_data_sha256':digest([r['sample'] for r in train]),
        'dataset_manifest_sha256':sha256((data_dir/'DATASET_FREEZE.json').read_bytes()).hexdigest(),
        'model_selection_only_on_validation':True,
        'selected_models':selections,
        'retrieval_k_selection':{'k':k_best,'validation_accuracy':ks},
        'effect_induction':effects.report,
        'effect_hypotheses_unreviewed':effects.models,
        'scores':results,
        'warnings':[
            'Scores are from a synthetic simulator: not a real-world Dolibarr model or provider-LLM result.',
            'The simulator is generated by code distributed in this package; not an independent black-box software validation.',
            'Known policy family uses threshold guards and at most one two-variable veto. Arbitrary enterprise business rules not covered.',
            'Artificial Weights and provider-model changes require the separately budgeted model-refiner fit experiment.',
            'Effect exactness is based on observed accepted states and is not evidence of real ERP causal accuracy.',
            'Balanced evaluation probabilities differ from naturally occurring enterprise class frequencies.',
        ],
    }
    out.mkdir(parents=True)
    rb=(json.dumps(report,sort_keys=True,indent=2)+'\n').encode('utf-8')
    (out/'benchmark_report.json').write_bytes(rb)
    freeze={'format':'simlearn-benchmark-freeze-v1','dataset_sha256':report['dataset_manifest_sha256'],
            'report_sha256':sha256(rb).hexdigest(),'live_dolibarr_requests':0,
            'live_dolibarr_writes':0,'provider_calls':0,
            'no_claim_of_dolibarr_fidelity':True}
    (out/'BENCH_FREEZE.json').write_text(json.dumps(freeze,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n')
    return report


def summarize(reports:list[dict])->dict:
    if not reports:raise ValueError('No reports')
    keys=list(reports[0]['scores']['test'])
    result={'format':'simlearn-benchmark-summary-v1','environments':len(reports),'scores':{}}
    for split in ('test','stress'):
        result['scores'][split]={}
        for name in keys:
            vals=[r['scores'][split][name] for r in reports]
            cases=sum(s['cases'] for s in vals)
            false_accepts=sum(s['false_accepts'] for s in vals)
            false_rejects=sum(s['false_rejects'] for s in vals)
            accuracy=sum(s['accuracy']*s['cases'] for s in vals)/cases
            end2end=sum(s['end_to_end_exact_accuracy']*s['cases'] for s in vals)/cases
            result['scores'][split][name]={
              'cases':cases,'accuracy':round(accuracy,6),
              'end_to_end_exact_accuracy':round(end2end,6),
              'false_accepts':false_accepts,'false_rejects':false_rejects,
              'per_environment_accuracy':[s['accuracy'] for s in vals],
              'per_environment_end_to_end':[s['end_to_end_exact_accuracy'] for s in vals],
              'min_environment_accuracy':min(s['accuracy'] for s in vals),
              'max_environment_accuracy':max(s['accuracy'] for s in vals)}
    return result
