"""Optional real-model black-box learning gate for a *synthetic* ERP.

Explicit --execute-model and OPENAI_API_KEY required. Without them this module
runs NO model calls. Model receives only synthetic public contracts and its own
discovery observations; evaluator-held outcomes and oracle implementation are
never in the prompt or model adapter. The deterministic audit shield remains
in force and cannot be overridden by model output.

Gate 2C does not issue Dolibarr API requests, credentials or customer data.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Protocol
from urllib.request import Request, urlopen

from .gate2a_lab import Operation,_SyntheticOracle,_make_pool,public_operations,run_arm,features_for
from .gate2c_counterexamples import (
    _applies,_candidate_axes,_fingerprint,fit_audited_model,predict_audited,
    run_cegis_arm,_compute,
)


class ModelBackend(Protocol):
    def complete_json(self, prompt:dict)->dict: ...


class OpenAIResponsesBackend:
    """One tightly bounded model call; no provider response content is persisted."""
    def __init__(self,model:str):
        api_key=os.getenv('OPENAI_API_KEY','')
        if not api_key:
            raise ValueError('OPENAI_API_KEY is required in the local process environment')
        if not model or len(model)>120 or any(c.isspace() for c in model):
            raise ValueError('Invalid model name')
        self._key=api_key
        self.model=model

    def complete_json(self,prompt:dict)->dict:
        instructions=(
            'You are inferring policy constraints from a synthetic black-box ERP. '
            'Return one JSON object, not prose: {"rule":[{"field":"...","op":"eq|ne|ge|le|gt|lt|in",'
            '"value":...}],"effects":{"output_field":{"kind":"literal|copy|add|subtract|offset",'
            '"value":...}},"uncertainties":["..."]}. '
            'Use only features and fields shown in the evidence. '
            'Never assert that a hypothesis is verified. Minimize erroneous acceptance. '
            'Use discovered examples, not a guess about Dolibarr. '
            'For subtract/add effects use from and source. For copy use from, '
            'for offset use from and delta, for literal use value.'
        )
        body={'model':self.model,'store':False,'max_output_tokens':1200,
              'input':[{'role':'system','content':instructions},
                       {'role':'user','content':json.dumps(prompt,separators=(',',':'))}]}
        request=Request('https://api.openai.com/v1/responses',
                        data=json.dumps(body).encode(),method='POST',
                        headers={'Authorization':'Bearer '+self._key,
                                 'Content-Type':'application/json'})
        with urlopen(request,timeout=80) as response:
            data=response.read(2_000_000)
        if len(data)>1_000_000:raise ValueError('Model response too large')
        payload=json.loads(data)
        text=payload.get('output_text')
        if not text:
            text=''.join(part.get('text','') for msg in payload.get('output',[])
                         for part in msg.get('content',[]) if part.get('type')=='output_text')
        if not text:raise ValueError('No text in model output')
        # Some model configurations wrap JSON in Markdown fences.
        text=text.strip()
        if text.startswith('```'):
            text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
        result=json.loads(text)
        if not isinstance(result,dict):raise ValueError('Expected a JSON object')
        return result


def public_prompt(op:Operation,observations:list[dict])->dict:
    """Serializes ONLY a public contract and the arm's synthetic observations."""
    return {
      'environment':'SYNTHETIC_ERP_RESEARCH_BENCHMARK',
      'operation':op.name,'description':op.description,
      'typed_state_inputs':{k:list(v) for k,v in op.fields.items() if k!='record_id'},
      'derived_features':{k:list(v) for k,v in op.derived.items()},
      'output_fields':list(op.outcome_fields),
      'declared_critical_axes':[a.name for a in _candidate_axes(op)],
      'discoveries':[{'features':o['features'],
                       'accepted':o['accepted'],
                       'changed_output_fields':{f:o['after'][f] for f in op.outcome_fields}
                       if o['accepted'] else {}}
                     for o in observations],
      'important_limit':'No held-out answers or hidden oracle thresholds provided.'
    }


def validate_model_proposal(op:Operation,result:dict,observations:list[dict])->dict:
    """Reject unsupported instructions or models inconsistent with discovered evidence."""
    if set(result)-{'rule','effects','uncertainties'}:
        raise ValueError('Unrecognized model proposal keys')
    rules=result.get('rule')
    effects=result.get('effects')
    if not isinstance(rules,list) or not rules or len(rules)>16:
        raise ValueError('No bounded conjunctive model proposal')
    allowed=set(op.fields)|set(op.derived)
    allowed.discard('record_id')
    cleaned=[]
    for atom in rules:
        if not isinstance(atom,dict) or set(atom)!={'field','op','value'}:
            raise ValueError('Invalid model predicate')
        field,relation,value=atom['field'],atom['op'],atom['value']
        if field not in allowed or relation not in {'eq','ne','ge','le','gt','lt','in'}:
            raise ValueError('Unsupported or unregistered model predicate')
        if relation=='in' and (not isinstance(value,list) or len(value)>20):
            raise ValueError('Invalid membership predicate')
        if relation!='in' and type(value) not in {int,float,str,bool}:
            raise ValueError('Invalid predicate threshold')
        cleaned.append({'field':field,'op':relation,'value':value})
    if not isinstance(effects,dict) or set(effects)!=set(op.outcome_fields):
        raise ValueError('All output effects must be specified')
    for name,effect in effects.items():
        if not isinstance(effect,dict) or effect.get('kind') not in {'literal','copy','add','subtract','offset'}:
            raise ValueError('Invalid effect syntax')
        if effect['kind']=='literal':
            if set(effect)!={'kind','value'} or type(effect['value']) not in {str,int,float,bool}:
                raise ValueError('Invalid literal')
        elif effect['kind']=='copy':
            if set(effect)!={'kind','from'} or effect['from'] not in op.fields:
                raise ValueError('Invalid copy')
        elif effect['kind'] in {'add','subtract'}:
            if set(effect)!={'kind','from','source'} or effect['from'] not in op.fields or effect['source'] not in op.fields:
                raise ValueError('Invalid binary effect')
        elif effect['kind']=='offset':
            if set(effect)!={'kind','from','delta'} or effect['from'] not in op.fields or type(effect['delta']) not in {int,float}:
                raise ValueError('Invalid offset')
    for o in observations:
        if _applies(op,cleaned,o['before'])!=o['accepted']:
            raise ValueError('Model rule contradicts discovery evidence')
        if o['accepted']:
            for field in op.outcome_fields:
                value=_compute(effects[field],o['before'])
                if type(value)!=type(o['after'][field]) or value!=o['after'][field]:
                    raise ValueError('Model effect contradicts observed transition')
    return {'rule':cleaned,'effects':effects}


def safeguarded_prediction(op:Operation,proposed:dict,shield:dict,state:dict)->dict:
    """The model cannot override the independently audited execution shield."""
    gate=predict_audited(op,shield,state)
    if gate['decision']=='abstain':return gate
    model_accept=_applies(op,proposed['rule'],state)
    if (model_accept and gate['decision']!='accept') or (not model_accept and gate['decision']!='reject'):
        return {'decision':'abstain','after':None,'reason':'model_vs_evidence_disagreement'}
    if not model_accept:return {'decision':'reject','after':state.copy()}
    after=state.copy()
    for field,effect in proposed['effects'].items():after[field]=_compute(effect,state)
    return {'decision':'accept','after':after,'unknown_effects':[]}


def _metrics(op:Operation,proposal:dict|None,shield:dict,truth:list[tuple])->dict:
    fp=fn=tp=tn=abstained=correct_effects=0
    for before,status,oracle_after in truth:
        predicted=(safeguarded_prediction(op,proposal,shield,before) if proposal
                   else {'decision':'abstain'})
        if predicted['decision']=='abstain':abstained+=1;continue
        accepted=status==200;pred_accept=predicted['decision']=='accept'
        fp+=int(pred_accept and not accepted);fn+=int(not pred_accept and accepted)
        tp+=int(pred_accept and accepted);tn+=int(not pred_accept and not accepted)
        if pred_accept and accepted:
            correct_effects+=int(all(type(predicted['after'][f])==type(oracle_after[f]) and
                                     predicted['after'][f]==oracle_after[f] for f in op.outcome_fields))
    return {'cases':len(truth),'true_accepts':tp,'true_rejects':tn,
            'false_accepts':fp,'false_rejects':fn,'abstentions':abstained,
            'effect_correct_on_true_accepts':correct_effects,
            'coverage':round((len(truth)-abstained)/len(truth),4)}


def model_gate(backend:ModelBackend,*,seeds:int=2,budget:int=20,heldout:int=140,
               freeze_seed:int=20271207,outdir:str|Path='build/gate2c_model')->dict:
    if not 1<=seeds<=12 or not 20<=budget<=60 or not 80<=heldout<=500:
        raise ValueError('Gate requires 1..12 seeds, 20..60 queries, 80..500 holdouts')
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for sidx in range(seeds):
        oracle=_SyntheticOracle(freeze_seed+sidx)
        for oi,op in enumerate(public_operations()):
            pool=_make_pool(op,freeze_seed+sidx*103+oi*17,n=max(240,budget+100))
            unif=run_arm(op,oracle,pool,arm='uniform',budget=budget,seed=freeze_seed+sidx*103+oi)
            active=run_cegis_arm(op,oracle,budget=budget,seed=freeze_seed+sidx*103+oi)
            observations_by_arm={'uniform':unif['observations'],'counterexample':active['observations']}
            # Holdouts are constructed AFTER discovery, excluding both arm's
            # business-state signatures. No model sees the evaluation states.
            exclude={_fingerprint(op,o['before']) for rows_ in observations_by_arm.values()
                     for o in rows_}
            held_pool=_make_pool(op,freeze_seed*3+sidx*771+oi*89,n=heldout+550)
            held=[state for state in held_pool if _fingerprint(op,state) not in exclude][:heldout]
            if len(held)<heldout:raise AssertionError('Insufficient disjoint holdouts')
            truth=[(state,*oracle.query(op,state,phase='evaluation')[:2]) for state in held]
            for arm,observed in observations_by_arm.items():
                shield=fit_audited_model(op,observed)
                prompt=public_prompt(op,observed)
                proposal=None;error=None
                try:
                    proposal=validate_model_proposal(op,backend.complete_json(prompt),observed)
                except (ValueError,KeyError,TypeError,ArithmeticError) as exc:
                    error=type(exc).__name__+': '+str(exc)[:120]
                # Provider/network errors are not silently interpreted as
                # benchmark abstentions; they abort the run.
                score=_metrics(op,proposal,shield,truth)
                rows.append({'seed_index':sidx,'arm':arm,'operation':op.name,'queries':budget,
                             'model_proposal_valid':proposal is not None,
                             'model_validation_error':error,
                             'shield_ready':shield['audit']['ready_for_offline_execution'],**score})
    report={'format':'orgsim-gate2c-true-model-v1','environment':'SYNTHETIC_ORACLE_ONLY',
            'backend_invocations':len(rows),
            'real_model_calls':len(rows) if isinstance(backend,OpenAIResponsesBackend) else 0,
            'model_provider':'openai_responses' if isinstance(backend,OpenAIResponsesBackend) else 'test_or_custom_backend',
            'seeds':seeds,'budget':budget,'heldout':heldout,'live_dolibarr_calls':0,
            'by_arm':{arm:{'calls':sum(x['arm']==arm for x in rows),
                           'model_valid':sum(x['model_proposal_valid'] for x in rows if x['arm']==arm),
                           'false_accepts':sum(x['false_accepts'] for x in rows if x['arm']==arm),
                           'false_rejects':sum(x['false_rejects'] for x in rows if x['arm']==arm),
                           'abstentions':sum(x['abstentions'] for x in rows if x['arm']==arm),
                           'cases':sum(x['cases'] for x in rows if x['arm']==arm)}
                      for arm in ('uniform','counterexample')},
            'caveats':['Model inputs contain synthetic observations only.',
                       'Model outputs are audited by an independently implemented shield.',
                       'Vendor ERP transfer remains untested.']}
    (out/'model_gate_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'model_gate_metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    return report


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description='Optional LLM-assisted synthetic reconstruction. Never calls Dolibarr.')
    parser.add_argument('--model',default='gpt-6')
    parser.add_argument('--execute-model',action='store_true',help='Explicit opt-in to bill the external model API')
    parser.add_argument('--seeds',type=int,default=2)
    parser.add_argument('--budget',type=int,default=20)
    parser.add_argument('--heldout',type=int,default=140)
    parser.add_argument('--out',default='build/gate2c_model')
    a=parser.parse_args(argv)
    if not a.execute_model:
        print('DRY RUN ONLY: no model called. Add --execute-model and configure OPENAI_API_KEY to run.')
        return 0
    backend=OpenAIResponsesBackend(a.model)
    report=model_gate(backend,seeds=a.seeds,budget=a.budget,heldout=a.heldout,outdir=a.out)
    print(json.dumps(report,indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
