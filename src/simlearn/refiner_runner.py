"""Actual Model Refiner 0.5rc1 SDK integration and opt-in model fitting.

No API calls without --execute-model. The offline mock verifies code paths,
not LLM efficacy. The Model Refiner artifact must pass its own release gate
before any attempt to treat it as deployable.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path

from model_refiner.backend import ModelBackend, OpenAIResponsesBackend
from model_refiner.suite.dataset import Dataset, Record
from model_refiner.suite.tasks import TaskSpec
from model_refiner.suite.config import TrainConfig
from model_refiner.suite.sdk import fit
from model_refiner.suite.training import plan_fit
from model_refiner.types import ModelResult
from .dataset import CLASS_TASK,EFFECT_TASK
from .learning import predict_generic, fit_models, predict_tabular, predict_retrieval, EffectLearner
from .dataset import load_outcomes
from .benchmark import _score


def choose_records(path:Path,task:TaskSpec,limits:dict)->list[Record]:
    """Uniform counts per label/operation in each split; disjoint groups stay disjoint."""
    path=Path(path)
    source=[json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]
    groups={}
    for row in source:
        if row['split']=='stress':continue
        k=(row['split'],row['slice'], 'ACCEPT' if row['target']=='ACCEPT' or '"decision":"ACCEPT"' in row['target'] else 'REJECT')
        groups.setdefault(k,[]).append(row)
    chosen=[]
    for split,quota in limits.items():
        if quota%2 or quota<4: raise ValueError('Refiner quotas must be even and >=4')
        for op in sorted({x['slice'] for x in source}):
            for label in ('ACCEPT','REJECT'):
                bucket=groups.get((split,op,label),[])
                if len(bucket)<quota//2:raise ValueError(f'Insufficient {split} {op} {label} records')
                chosen.extend(bucket[:quota//2])
    records=[Record.from_dict(x,i) for i,x in enumerate(chosen)]
    Dataset.prepare(records,task)
    return records


def build_config(*,model:str,optimizer_model:str|None,task:TaskSpec,rounds:int=1,
                 max_attempts:int=2500,max_reserved_tokens:int=40_000_000,
                 max_output_tokens:int=8192,seed:int=42)->TrainConfig:
    return TrainConfig(
        model=model,task=task,optimizer_model=optimizer_model,
        seed=seed,k_values=(2,),rounds=rounds,max_train_errors=12,max_train_scan=24,
        max_attempts=max_attempts,max_reserved_tokens=max_reserved_tokens,
        max_output_tokens=max_output_tokens,max_compile_tokens=max(8192,max_output_tokens),
        retries=0,repair_attempts=1,enable_weights=True,weight_epochs=30,
        preflight_probe=True,balanced_error_sampling=True,rule_coverage_check=True)


class MockERPBackend(ModelBackend):
    """A fixed deterministic heuristic for SDK wiring tests. It cannot learn.

    Never use its scores as evidence of Model Refiner or frozen-LLM performance.
    """
    def generate(self,*,model:str,messages:list[dict])->ModelResult:
        text=messages[-1]['content']
        if any('REFINER_COMPILE' in x['content'] for x in messages if x['role']=='developer'):
            return ModelResult('Examples suggest that both ACCEPT and REJECT depend on operational preconditions.\n'
                               'Never execute unverified changes. ACCEPT and REJECT are the only valid labels.')
        try:
            source=json.loads(text)
            if isinstance(source,dict) and 'input' in source and 'candidate' in source:source=json.loads(source['input'])
            decision=predict_generic({'sample':source})
        except (ValueError,KeyError,TypeError):decision='REJECT'
        return ModelResult(text=decision,input_tokens=64,output_tokens=4)

    def cache_identity(self):return {'mock_backend':'synthetic_public_guard_v1', 'real_llm':False}


def prepare_run(*,data_dir:str|Path,task_name:str='classification',model:str='gpt-6',
                optimizer_model:str|None=None,limits:dict|None=None,
                max_attempts:int=2500,max_reserved_tokens:int=40_000_000,
                max_output_tokens:int=8192,seed:int=42):
    if task_name not in ('classification','effects'):raise ValueError(task_name)
    task=CLASS_TASK if task_name=='classification' else EFFECT_TASK
    limits=limits or {'train':32,'router':8,'validation':8,'test':32}
    path=Path(data_dir)/f'{task_name}_refiner.jsonl'
    data=choose_records(path,task,limits)
    config=build_config(model=model,optimizer_model=optimizer_model or model,task=task,
                        max_attempts=max_attempts,max_reserved_tokens=max_reserved_tokens,
                        max_output_tokens=max_output_tokens,seed=seed)
    ds=Dataset.prepare(data,task,seed=seed)
    plan=plan_fit(ds,config)
    if plan['provider_attempt_upper_bound']>config.max_attempts:
        raise ValueError(f'Plan upper bound {plan["provider_attempt_upper_bound"]} provider attempts exceeds '
                         f'max_attempts={max_attempts}; raise limit or reduce split sizes/candidates')
    return data,config,plan


def run_refiner(*,data_dir:str|Path,workspace:str|Path,task_name:str='classification',
                model:str='gpt-6',optimizer_model:str|None=None,execute_model:bool=False,
                mock:bool=False,limits:dict|None=None,max_attempts:int=2500,
                max_reserved_tokens:int=40_000_000,max_output_tokens:int=8192,
                seed:int=42):
    if mock and execute_model:raise ValueError('Mock and live model options cannot be combined')
    data,config,plan=prepare_run(data_dir=data_dir,task_name=task_name,model=model,
         optimizer_model=optimizer_model,limits=limits,max_attempts=max_attempts,
         max_reserved_tokens=max_reserved_tokens,max_output_tokens=max_output_tokens,seed=seed)
    safe={'format':'orgsim-refiner-fit-plan-v1','task':task_name,
          'provider':'mock' if mock else 'openai' if execute_model else 'not_executed',
          'model':model,'optimizer_model':optimizer_model or model,
          'plan':plan,'real_dolibarr_data_used':False,
          'synthetic_labels_only':True,
          'note':'Estimated worst-case provider attempts, not a dollar price; no background/model/ERP requests without --execute-model.'}
    if not execute_model and not mock:return safe
    if execute_model:
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError('OPENAI_API_KEY must be configured privately on the operator machine')
        backend=OpenAIResponsesBackend(max_output_tokens=max_output_tokens)
        provider='openai'
    else:
        if task_name!='classification':raise ValueError('Mock tests classification wiring only')
        backend=MockERPBackend();provider='toy'
    workspace=Path(workspace)
    if workspace.exists():raise FileExistsError(f'Use a fresh run directory: {workspace}')
    result=fit(backend,provider,data,config,workspace=workspace)
    safe['run_completed']=True
    safe['native_refiner_artifact_id']=result.artifact.artifact_id
    safe['native_refiner_gate']=result.report.get('gate')
    safe['native_refiner_scores']={name:{'score':metric['score'],'coverage':metric['coverage'],'per_operation':{k:v['score'] for k,v in metric.get('slices',{}).items()}} for name,metric in result.report['arms'].items()}
    safe['native_refiner_comparisons']=result.report.get('comparisons')
    safe['native_selected_mode']=result.report.get('selected_mode')
    safe['model_provider_calls']=result.report.get('budget',{}).get('upstream_attempts')
    safe['native_aw_enabled']=result.artifact.to_dict()['use_weights']
    # Fair comparison: exact same selected training labels and same test identities.
    source=load_outcomes(Path(data_dir)/'outcome.jsonl')
    by_id={r['id']:r for records in source.values() for r in records}
    selected={s:[by_id[r.id] for r in data if r.split==s]
              for s in ('train','router','validation','test')}
    if task_name=='classification':
        tabular_models,_=fit_models(selected['train'],selected['validation'],seed=seed)
        fx=EffectLearner().fit(selected['train'],selected['validation'])
        benchmark_arms={
          'public_guards':[predict_generic(r) for r in selected['test']],
          'native_retrieval_memory_knn5':predict_retrieval(selected['train'],selected['test'],k=5),
          'tabular_simtrained':predict_tabular(tabular_models,selected['test']),
        }
        safe['matched_evidence_conventional_baselines']={name:_score(selected['test'],preds,fx)
                 for name,preds in benchmark_arms.items()}
        safe['matching_test_ids_sha256']=__import__('hashlib').sha256(
            json.dumps([r['id'] for r in selected['test']],sort_keys=True).encode()).hexdigest()
    else:
        safe['matched_evidence_conventional_baselines']=None
        safe['note_about_effect_comparison']='JSON exact effects scored by the native suite; compare with separately run synthetic effect learner on same selected records.'
    (workspace/'SIMLEARN_BRIDGE_REPORT.json').write_text(json.dumps(safe,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    return safe
