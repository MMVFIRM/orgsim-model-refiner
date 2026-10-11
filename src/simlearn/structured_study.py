"""One integrated, pre-registered experiment for executable simulation learning.

All four arms see equal historical training data. The matched-active tabular arm
also receives *every* active query label that the structured learner receives.
Test/stress sets are read only after programs are frozen; no refitting.
No LLM or real Dolibarr is contacted by this entrypoint.
"""
from __future__ import annotations
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from .oracle import OPS,Oracle,digest,sample_input
from .dataset import load_outcomes,generate_environment
from .learning import EffectLearner,fit_models,predict_tabular,predict_generic
from .structured import Program,ModelRefinerPolicyAdapter,fit_structured
from .out_of_class import NonMonotoneOracle

FORMAT='orgsim-refiner-executable-policy-study-v1'

def payload_write(path:Path,obj:dict)->str:
    raw=(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n').encode()
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(raw)
    return sha256(raw).hexdigest()

def evaluate(rows:list[dict],decisions:list[str],effects:EffectLearner)->dict:
    if len(rows)!=len(decisions):raise ValueError('Misaligned predictions')
    n=len(rows);positive=sum(r['decision']=='ACCEPT' for r in rows)
    negative=n-positive
    tp=tn=fp=fn=abst=exact=0
    per=defaultdict(lambda:{'cases':0,'true_accepts':0,'false_accepts':0,'false_rejects':0,'abstentions':0})
    for r,pred in zip(rows,decisions):
        if pred not in ('ACCEPT','REJECT','ABSTAIN'):raise ValueError('Invalid prediction')
        op=r['operation'];d=per[op];d['cases']+=1
        if pred=='ABSTAIN':abst+=1;d['abstentions']+=1;continue
        if pred=='ACCEPT' and r['decision']=='ACCEPT':
            tp+=1;d['true_accepts']+=1
            if effects.predict(r['sample'],pred)==r['changed']:exact+=1
        elif pred=='REJECT' and r['decision']=='REJECT':tn+=1
        elif pred=='ACCEPT':fp+=1;d['false_accepts']+=1
        else:fn+=1;d['false_rejects']+=1
    assert tp+tn+fp+fn+abst==n
    return {'cases':n,'answered':n-abst,'coverage':round((n-abst)/n,6),
            'accuracy_all_cases':round((tp+tn)/n,6),
            'accuracy_on_answered':round((tp+tn)/(n-abst),6) if n>abst else None,
            'effect_exact_on_true_accepts':round(exact/tp,6) if tp else None,
            'end_to_end_exact_all_cases':round((tn+exact)/n,6),
            'false_accepts':fp,'false_rejects':fn,'abstentions':abst,
            'false_accept_rate_on_true_rejects':round(fp/negative,6) if negative else None,
            'accept_recall_all_cases':round(tp/positive,6) if positive else None,
            'true_accepts':tp,'true_rejects':tn,'positive_truth':positive,
            'negative_truth':negative,'per_operation':dict(sorted(per.items()))}

def run_environment(data_dir:Path|str,outdir:Path|str,*,seed:int,max_probes:int=160,
                    audit_count:int=32,baseline_seed:int|None=None)->dict:
    data_dir=Path(data_dir);outdir=Path(outdir)
    if outdir.exists():raise FileExistsError(f'Refusing to overwrite: {outdir}')
    splits=load_outcomes(data_dir/'outcome.jsonl')
    manifest=json.loads((data_dir/'DATASET_FREEZE.json').read_text())
    if manifest['seed']!=seed:raise ValueError('Dataset seed and oracle mismatch')
    if manifest['not_real_dolibarr'] is not True:raise ValueError('Unverified provenance')
    seen={digest(r['sample']) for r in splits['train']+splits['router']+splits['validation']}
    sealed={digest(r['sample']) for r in splits['test']+splits['stress']}
    if seen&sealed:raise ValueError('Holdout leakage before learning')
    train=splits['train'];valid=splits['validation']
    oracle=Oracle(seed)
    programs={};probe_rows=[]
    try:
        for op in OPS:
            candidate,examples=fit_structured(oracle.execute,op,seed=seed+len(op),
                 max_queries=max_probes,audit_count=audit_count,validation=valid)
            programs[op]=candidate
            probe_rows+=examples
        # An exact overlap with sealed samples is unacceptable, even if unintentional.
        active={digest(row['sample']) for row in probe_rows}
        if active & sealed:raise ValueError('Active probe overlaps sealed test/stress example')
        if any(row['decision'] not in ('ACCEPT','REJECT') for row in probe_rows):raise ValueError('Invalid probe')
        # Matched evidence includes every active training label, not just positives.
        passive_models,passive_choice=fit_models(train,valid,seed=baseline_seed or seed)
        active_models,active_choice=fit_models(train+probe_rows,valid,seed=baseline_seed or seed)
        effects=EffectLearner().fit(train,valid)
        adapter=ModelRefinerPolicyAdapter(programs,tenant_fingerprint=manifest['manifest_sha256'])
        results={}
        for split in ('test','stress'):
            cases=splits[split]
            arms={
                'public_guards':[predict_generic(r) for r in cases],
                'tabular_passive':predict_tabular(passive_models,cases),
                'tabular_matched_active':predict_tabular(active_models,cases),
                'structured_program':[adapter.decide(r['sample'],tenant_fingerprint=manifest['manifest_sha256'])['decision'] for r in cases],
            }
            results[split]={name:evaluate(cases,pred,effects) for name,pred in arms.items()}
        prog_payload=adapter.frozen_payload()
        outdir.mkdir(parents=True)
        program_sha=payload_write(outdir/'MODEL_REFINER_PROGRAM_MEMORY.json',prog_payload)
        report={'format':FORMAT,'seed':seed,'environment_class':'MONOTONE_BOUNDED_SYNTHETIC',
            'evidence_splits':'group_disjoint_train_router_validation_test_stress',
            'source_manifest_sha256':manifest['manifest_sha256'],
            'dataset_freeze_sha256':sha256((data_dir/'DATASET_FREEZE.json').read_bytes()).hexdigest(),
            'no_real_dolibarr_evidence':True,'no_real_dolibarr_writes':True,
            'provider_model_calls':0,'native_model_refiner_fit_repeated':False,
            'native_model_refiner_recipe_scores_not_measured_by_this_study':True,
            'learner_read_private_policy':False,
            'public_operation_axes_provided':True,
            'rule_grammar':'public guards AND monotone unary, sum, pair upper vetoes',
            'matched_probe_evidence_for_tabular':True,
            'train_size':len(train),'validation_size':len(valid),
            'queries_in_active_adaptation':len(probe_rows),
            'probes_by_operation':{op:programs[op].queries for op in OPS},
            'audits_by_operation':{op:{'ready':p.audit_ready,'evaluated':p.audit_total,
                                      'mismatches':p.audit_mismatches} for op,p in programs.items()},
            'effect_induction_validated':effects.report,
            'passive_tabular_selection':passive_choice,
            'matched_tabular_selection':active_choice,
            'program_artifact_sha256':program_sha,
            'scores':results,
            'cautions':['Scope-limited synthetic program synthesis. Simulator source is not independent real ERP truth.',
                'Monotone-rule grammar and public intervention axes are supplied; model discovers numerical constants.',
                'Active synthetic interventions use additional oracle calls counted separately from passive training.',
                'Model Refiner native GPT/Luna performance was measured in the previous run, not repeated here.',
                'No real-world transactional safety claim; false accepts must be treated as blockers.']}
        report_sha=payload_write(outdir/'STRUCTURED_REPORT.json',report)
        payload_write(outdir/'EXPERIMENT_FREEZE.json',{'format':FORMAT+'-freeze',
              'report_sha256':report_sha,'program_sha256':program_sha,
              'dataset_freeze_sha256':report['dataset_freeze_sha256'],
              'provider_calls':0,'dolibarr_reads':0,'dolibarr_writes':0,
              'no_external_timestamp':True})
        return report
    finally:oracle.close()

def run_ood(seed:int,*,max_probes:int=160,audit_count:int=32,eval_per_op:int=120)->dict:
    """Out-of-grammar adversarial oracle, independently queried after fit.

    The learned program must *abstain* if audits show its grammar is invalid.
    No OOD oracle definition is supplied to the learner callback.
    """
    import random
    o=NonMonotoneOracle(seed)
    ready=counts={}
    results={}
    try:
        for op in OPS:
            p,_=fit_structured(o.execute,op,seed=seed+len(op),
                    max_queries=max_probes,audit_count=audit_count)
            rng=random.Random(seed*7919+OPS.index(op)*113)
            # This split is fresh random population; no post-test fitting.
            eval_rows=[]
            for i in range(eval_per_op):
                s=sample_input(rng,op,mode='stress')
                result=o.execute(s)
                eval_rows.append({**result,'sample':s,'operation':op})
            decisions=[p.classify(r['sample']) for r in eval_rows]
            results[op]={'audit_ready':p.audit_ready,'audit_mismatches':p.audit_mismatches,
                         'probes':p.queries,'coverage':round(sum(x!='ABSTAIN' for x in decisions)/len(decisions),6),
                         'incorrect_accepts':sum(x=='ACCEPT' and r['decision']=='REJECT' for x,r in zip(decisions,eval_rows)),
                         'incorrect_rejects':sum(x=='REJECT' and r['decision']=='ACCEPT' for x,r in zip(decisions,eval_rows))}
        return {'format':'simlearn-nonmonotone-stress-v1','seed':seed,'per_operation':results,
                'adapter_generalization_claim':'NONE','real_erp_verified':False,
                'caution':'Out-of-grammar detection is empirical, not complete proof of safety.'}
    finally:o.close()

def summarize(reports:list[dict],ood:list[dict])->dict:
    if not reports:raise ValueError('Need nonempty experiment')
    output={'format':FORMAT+'-summary','organizations':len(reports),
            'seeds':[r['seed'] for r in reports], 'arms':{},
            'active_oracle_calls_total':sum(r['queries_in_active_adaptation'] for r in reports),
            'all_audits_ready':all(all(t['ready'] for t in r['audits_by_operation'].values()) for r in reports),
            'adversarial_out_of_grammar':ood,
            'provider_model_calls':0,'dolibarr_calls':0,'independent_real_erp_holdout':False}
    for split in ('test','stress'):
        output['arms'][split]={}
        for arm in reports[0]['scores'][split]:
            ss=[r['scores'][split][arm] for r in reports]
            n=sum(s['cases'] for s in ss)
            correct=sum(s['true_accepts']+s['true_rejects'] for s in ss)
            end_to_end=sum(round(s['end_to_end_exact_all_cases']*s['cases']) for s in ss)
            negatives=sum(s['negative_truth'] for s in ss)
            output['arms'][split][arm]={'cases':n,'accuracy_all_cases':round(correct/n,6),
               'end_to_end_exact_all_cases':round(end_to_end/n,6),
               'coverage':round(sum(s['answered'] for s in ss)/n,6),
               'incorrect_accepts':sum(s['false_accepts'] for s in ss),
               'incorrect_rejects':sum(s['false_rejects'] for s in ss),
               'abstentions':sum(s['abstentions'] for s in ss),
               'false_accept_rate_on_true_rejects':round(sum(s['false_accepts'] for s in ss)/negatives,6),
               'per_environment_accuracy':[s['accuracy_all_cases'] for s in ss]}
    return output
