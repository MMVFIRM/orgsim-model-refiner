"""Gate 2G: consolidated, fixed-rubric, strictly GET-only structural prediction.

Contract vs prior-evidence symbolic vs optional real LLM. All arms receive
identical discovery/Swagger evidence; predictions frozen before any live GET.
Only NEW query templates relative to the revealed Gate 2E cases are scored.
No fixture values, IDs, auth tokens, or ERP response bodies are persisted.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .dolibarr_gate import DolibarrClient, _local_host
from .gate2b_readonly import _ID, _REF
from .gate2d_adaptive import (EvidenceLog, hash_json, safe_shape, type_name,
                               canonical, OpenAIProbeAdvisor)
from .gate2e_blind import _source_report, _profiles, _allowed_evidence
from .conditional_read import ConditionalReadModel, TYPE_KINDS

FORMAT = 'orgsim-gate2g-fixed-denominator-v1'
FAMILIES = ('thirdparties','products','orders','invoices')
MAX_BUDGET = 48
# New query variants only. Exact Gate 2D/2E scored probes are intentionally absent.
# The few follow-on detail routes are new to this benchmark, and may be unmodeled.
@dataclass(frozen=True)
class Query:
    id: str
    family: str
    route: str
    evidence_class: str
    selected: tuple[str,...] = ()
    index: int = 0
    special: str = ''        # 'reverse', 'ref_match', or ''

    def public(self):
        x=asdict(self);x['selected']=list(self.selected);return x

QUERIES: tuple[Query,...] = (
    Query('customers_asc_ref','thirdparties','/thirdparties?limit=2&sortfield=t.rowid&sortorder=ASC&properties=id,ref,name,client',
          'unseen_ordering',('id','ref','name','client')),
    Query('customers_empty_page','thirdparties','/thirdparties?limit=2&page=2&properties=id,name,client',
          'unseen_pagination',('id','name','client')),
    Query('customers_desc_limit2','thirdparties','/thirdparties?limit=2&sortorder=DESC&properties=id,ref,name',
          'unseen_combined_modifier',('id','ref','name'),special='reverse'),
    Query('products_projected_new','products','/products?limit=2&properties=id,ref,label,type,price_ttc',
          'unseen_projection',('id','ref','label','type','price_ttc')),
    Query('products_ids_page0','products','/products?limit=2&page=0&ids_only=1',
          'unseen_combined_modifier'),
    Query('products_category_filter','products','/products?limit=3&category=1&properties=id,ref,label',
          'unseen_filter',('id','ref','label')),
    Query('orders_small_projection','orders','/orders?limit=1&properties=id,ref,socid,status,total_ht',
          'unseen_projection',('id','ref','socid','status','total_ht')),
    Query('orders_page2','orders','/orders?limit=1&page=2&properties=id,ref,socid,status',
          'unseen_pagination',('id','ref','socid','status')),
    Query('orders_new_sort','orders','/orders?limit=2&sortfield=t.rowid&sortorder=DESC&properties=id,ref,socid,status',
          'unseen_ordering',('id','ref','socid','status'),special='reverse'),
    Query('invoices_with_lines','invoices','/invoices?limit=1&withLines=true&properties=id,ref,status,lines',
          'unseen_projection',('id','ref','status','lines')),
    Query('invoices_suppress_lines','invoices','/invoices?limit=1&withLines=false&properties=id,ref,status,lines',
          'unseen_combined_modifier',('id','ref','status','lines')),
    Query('invoices_status_cancelled','invoices','/invoices?limit=2&status=cancelled&properties=id,ref,status',
          'unseen_filter',('id','ref','status')),
    Query('customer_outstanding_mode','thirdparties','/thirdparties/{id}/outstandingorders?mode=1',
          'unseen_relationship_modifier',('opened','refs'),index=1),
    Query('customer_notifications','thirdparties','/thirdparties/{id}/notifications',
          'unseen_endpoint'),
    Query('customer_representatives','thirdparties','/thirdparties/{id}/representatives?mode=1',
          'unseen_endpoint'),
    Query('customer_fixed_discount','thirdparties','/thirdparties/{id}/fixedamountdiscounts?mode=0',
          'unseen_endpoint'),
    Query('customer_supplier_categories','thirdparties','/thirdparties/{id}/supplier_categories?limit=2',
          'unseen_endpoint'),
    Query('product_contacts','products','/products/{id}/contacts',
          'unseen_endpoint'),
    Query('product_subproducts','products','/products/{id}/subproducts',
          'unseen_endpoint'),
    Query('product_quantity_prices','products','/products/{id}/selling_multiprices/per_quantity',
          'unseen_endpoint'),
    Query('product_ref_variants','products','/products/ref/{ref}/variants',
          'unseen_endpoint'),
    Query('invoice_contact_filter','invoices','/invoices/{id}/contacts?type=external',
          'unseen_endpoint_modifier'),
    Query('order_contact_filter','orders','/orders/{id}/contacts?type=external',
          'unseen_endpoint_modifier'),
    Query('invoice_detail_suppress_lines','invoices','/invoices/{id}?withLines=false&properties=id,ref,status,lines',
          'unseen_detail_modifier',('id','ref','status','lines')),
)

BY_ID = {q.id:q for q in QUERIES}
VALID_KINDS={'array','object','string','number','null','boolean','unknown'}
ALLOWED_FIELDS={'status','kind','length','item_kind','item_keys','field_types','reverse_matches','ref_matches'}


def _json_bytes(x): return (json.dumps(x,indent=2,sort_keys=True)+'\n').encode('utf-8')


def _check_pred(q:Query,x:dict) -> dict:
    if not isinstance(x,dict) or set(x)-ALLOWED_FIELDS:raise ValueError('Invalid prediction fields')
    res={}
    if 'status' in x:
        if type(x['status']) is not int or not 100<=x['status']<=599:raise ValueError('Invalid status')
        res['status']=x['status']
    for n in ('kind','item_kind'):
        if n in x:
            if x[n] not in VALID_KINDS:raise ValueError('Invalid kind')
            res[n]=x[n]
    if 'length' in x:
        if type(x['length']) is not int or not 0<=x['length']<=20:raise ValueError('Invalid length')
        res['length']=x['length']
    if 'item_keys' in x:
        if (not isinstance(x['item_keys'],list) or len(x['item_keys'])>12
                or any(not isinstance(v,str) or v not in q.selected for v in x['item_keys'])):
            raise ValueError('Invalid projected keys')
        res['item_keys']=sorted(set(x['item_keys']))
    if 'field_types' in x:
        if (not isinstance(x['field_types'],dict) or len(x['field_types'])>12 or any(
                k not in q.selected or not (v in VALID_KINDS if isinstance(v,str)
                        else (isinstance(v,list) and 2<=len(v)<=3 and len(v)==len(set(v))
                              and all(isinstance(t,str) and t in TYPE_KINDS for t in v)))
                for k,v in x['field_types'].items())):
            raise ValueError('Invalid field type projection')
        res['field_types']={k:(sorted(x['field_types'][k]) if isinstance(x['field_types'][k],list)
                                else x['field_types'][k]) for k in sorted(x['field_types'])}
    for n,special in (('reverse_matches','reverse'),('ref_matches','ref_match')):
        if n in x:
            if q.special!=special or type(x[n]) is not bool:raise ValueError('Invalid equality assertion')
            res[n]=x[n]
    return res


def _swagger_ops(swagger:dict) -> dict:
    """Route descriptions stripped to non-secret, public interface facts."""
    if swagger.get('swagger')!='2.0' or not isinstance(swagger.get('paths'),dict):
        raise ValueError('Need sanitized Swagger 2.0 JSON')
    result={}
    for q in QUERIES:
        # Optional query details and dynamic path variables are removed.
        base=q.route.split('?',1)[0]
        operation=swagger['paths'].get(base,{}).get('get',{})
        if not operation:raise ValueError('Swagger has no documented GET for '+base)
        response=operation.get('responses',{}).get('200',{}).get('schema',{})
        schema_type=response.get('type','unknown')
        if '$ref' in response:schema_type='unknown'
        result[q.id]={'success_response_schema':schema_type,
                      'success_code_documented':True,
                      'query_params':[p['name'] for p in operation.get('parameters',[])
                                      if isinstance(p,dict) and p.get('in')=='query'
                                      and isinstance(p.get('name'),str)],
                      'operation_id':operation.get('operationId','')}
    return result


def _contract(q:Query, contract:dict) -> dict:
    # Swagger describes success type, NOT authorization, actual state, or defaults.
    k=contract['success_response_schema']
    pred={'status':200}
    if k in VALID_KINDS-{'unknown'}:pred['kind']=k
    return _check_pred(q,pred)


def _symbolic(q:Query, profiles:dict, contract:dict) -> dict:
    p=profiles.get(q.family+'_list',{})
    kind=p.get('kind')
    pred={}
    if p.get('status')==200 and kind=='array':
        pred={'status':200,'kind':'array'}
    if '{ref}' in q.route:
        # Only the first product-by-ref had previous direct evidence, not these.
        # Structural guess: novel alternate lookup exists and returns object.
        pred={'status':200,'kind':'object'}
        if q.special=='ref_match':pred['ref_matches']=True
    elif q.route.startswith('/thirdparties/{id}/outstanding'):
        origin=('thirdparties_outstanding_orders' if 'orders' in q.route
                else 'thirdparties_outstanding_invoices')
        o=profiles.get(origin,{})
        pred={'status':o['status'],'kind':o['kind']} if o.get('status')==200 else {}
        pred['field_types']={k:v for k in q.selected
                             if (v:=o.get('field_types',{}).get(k)) in VALID_KINDS-{'null','unknown'}}
    elif '{id}' in q.route:
        # New relationship endpoints: no prior observed outcome.
        pred={}
    elif q.special=='reverse':
        # Hypothesis: documented sortorder DESC reverses documented default ASC.
        if p.get('length',0)>=2:pred['reverse_matches']=True
    if q.route.find('ids_only=1')>=0:
        pred['kind']='array';pred['item_kind']='string'
        n=p.get('length')
        if type(n) is int:pred['length']=min(n,3)
    if 'properties=' in q.route and p.get('status')==200:
        pred['kind']='array'
        pred['item_keys']=sorted(k for k in q.selected if k in p.get('field_keys',[]))
        pred['field_types']={k:v for k in q.selected
                             if (v:=p.get('field_types',{}).get(k)) in VALID_KINDS-{'null','unknown'}}
        if 'page=2' in q.route and (p.get('length') or 0)==3:
            pred['length']=1
        elif 'status=' not in q.route and 'mode=' not in q.route:
            n=p.get('length')
            if type(n) is int:pred['length']=min(n, 3 if 'limit=3' in q.route else 1)
    return _check_pred(q,pred)


class OpenAI2FPredictor:
    """Explicit opt-in, one provider request at FREEZE time, never during live GET."""
    def __init__(self,model='gpt-6'):
        self.model=model
        self._advisor=OpenAIProbeAdvisor(model)

    def predict_many(self,cases:list[Query],evidence:dict) -> dict:
        from urllib.request import Request
        system=('You are predicting READ-ONLY black-box ERP API outcomes. '
                'Return exactly one JSON object {"predictions":{"case_id":{...}}}. '
                'Use only evidence supplied. Omit unsupported guesses; abstention is valid. '
                'For each case permitted keys: status, kind, length, item_kind, item_keys, '
                'field_types (types may be lists of 2-3 alternatives), and reverse_matches/ref_matches when applicable. '
                'Do not output or infer actual IDs, field values, credentials, secrets or routes.')
        request={'model':self.model,'store':False,'max_output_tokens':5000,
                 'input':[{'role':'system','content':system},
                          {'role':'user','content':canonical({'evidence':evidence,
                             'case_templates':[q.public() for q in cases]})}]}
        req=Request('https://api.openai.com/v1/responses',method='POST',
                    data=canonical(request).encode(),headers={
                      'Authorization':'Bearer '+self._advisor._key,
                      'Content-Type':'application/json'})
        with self._advisor._opener.open(req,timeout=120) as resp:raw=resp.read(1_000_001)
        if len(raw)>1_000_000:raise ValueError('Provider response exceeds limit')
        response=json.loads(raw)
        content=response.get('output_text') or ''.join(
            part.get('text','') for item in response.get('output',[])
            for part in item.get('content',[]) if part.get('type')=='output_text')
        if content.lstrip().startswith('```'):
            content=content.strip().split('\n',1)[-1].rsplit('```',1)[0].strip()
        parsed=json.loads(content)
        if set(parsed)!= {'predictions'} or not isinstance(parsed['predictions'],dict):
            raise ValueError('Model response must contain predictions only')
        if set(parsed['predictions'])-BY_ID.keys():raise ValueError('Unrecognized model case')
        return {key:_check_pred(BY_ID[key],v) for key,v in parsed['predictions'].items()}


def _augment_profile_cache(profiles:dict,development:dict)->dict:
    """Naive cache uses SAME revealed evidence as the conditional learner.

    It aggregates latest non-null field types but does not learn applicability,
    polymorphism, guards, or parameter-specific omissions. This is a useful
    matched-evidence ablation, not a third weaker-information baseline.
    """
    merged=json.loads(json.dumps(profiles))
    for row in development.get('cases',[]):
        family=row.get('family')
        observation=row.get('observation',{})
        if family not in FAMILIES or observation.get('status')!=200:
            continue
        route=row.get('route_template','').split('?',1)[0]
        origin_routes={
            '/thirdparties/{id}/outstandingorders':'thirdparties_outstanding_orders',
            '/thirdparties/{id}/outstandinginvoices':'thirdparties_outstanding_invoices',
            '/products/{id}/stock':'products_stock',
            '/products/{id}/categories':'products_categories',
            '/orders/{id}/shipment':'orders_shipment'}
        targets=[family+'_list']
        if route in origin_routes:targets.append(origin_routes[route])
        for key in targets:
            current=merged.get(key)
            if not current:continue
            for field,kind in observation.get('field_types',{}).items():
                if kind in VALID_KINDS-{'unknown','null'}:
                    current.setdefault('field_types',{})[field]=kind
    return merged


def prepare(discovery:str|Path,swagger:str|Path,outdir:str|Path,*,model_advisor=None, development:str|Path|None=None) -> dict:
    d,dh=_source_report(Path(discovery))
    swagger_raw=Path(swagger).read_bytes()
    catalog=_swagger_ops(json.loads(swagger_raw))
    profiles=_profiles(d)
    development_json = json.loads(Path(development).read_bytes()) if development else {}
    development_sha = sha256(Path(development).read_bytes()).hexdigest() if development else None
    symbolic_profiles=_augment_profile_cache(profiles,development_json)
    contract={q.id:_contract(q,catalog[q.id]) for q in QUERIES}
    symbolic={q.id:_symbolic(q,symbolic_profiles,catalog[q.id]) for q in QUERIES}
    conditioned=ConditionalReadModel(d, development_json)
    conditional={q.id:_check_pred(q, conditioned.predict(q,catalog[q.id])) for q in QUERIES}
    model=None
    if model_advisor is not None:
        safe={'discovery':_allowed_evidence(profiles),'swagger_get_catalog':catalog,
              'development_evidence': [
                  {'family':row.get('family'),'route_template':row.get('route_template'),
                   'observation':row.get('observation')}
                  for row in development_json.get('cases',[])],
              'previous_revealed_query_templates':[
                '/orders?limit=2&properties=id,socid,ref,status',
                '/products?limit=2&properties=id,ref,label,price',
                '/thirdparties?limit=2&properties=id,name,client'],
              'role':'same_readonly_user','scope':'same_tenant'}
        model=model_advisor.predict_many(list(QUERIES),safe)
        if set(model)-BY_ID.keys():raise ValueError('Model invented probe')
        model={key:_check_pred(BY_ID[key],v) for key,v in model.items()}
    result={'format':FORMAT+'-candidate','version':'0.2.6','reviewed':False,
            'behavioral_clone':False,'role_scope':'same_readonly_API_user',
            'source_discovery_sha256':dh,
            'source_swagger_sha256':sha256(swagger_raw).hexdigest(),
            'source_development_sha256':development_sha,
            'conditional_evidence_summary':conditioned.evidence_summary(),
            'matched_evidence_arms':['symbolic','conditional','model'] if model is not None else ['symbolic','conditional'],
            'arms':{'contract':contract,'symbolic':symbolic,'conditional':conditional,
                    **({'model':model} if model is not None else {})},
            'model_provider_calls_during_prepare':int(model is not None)}
    plan={'format':FORMAT+'-plan','query_holdout':True,'fixed_scoring_rubric':True,
          'independent_tenant_holdout':False,'prior_gate2e_cases_excluded':True,
          'cases':[q.public() for q in QUERIES]}
    out=Path(outdir)
    if (out/'PREPARE_FREEZE.json').exists():raise FileExistsError('Existing freeze; use a new directory')
    out.mkdir(parents=True,exist_ok=True)
    rb=_json_bytes(result);pb=_json_bytes(plan)
    (out/'candidate_frozen.json').write_bytes(rb)
    (out/'holdout_plan.json').write_bytes(pb)
    freeze={'format':FORMAT+'-prepare-freeze','phase':'prepare',
            'candidate_sha256':sha256(rb).hexdigest(),
            'plan_sha256':sha256(pb).hexdigest(),
            'source_discovery_sha256':dh,'source_swagger_sha256':sha256(swagger_raw).hexdigest(),
            'source_development_sha256':development_sha,
            'model_provider_calls':int(model is not None),'live_reads':0,'live_writes':0,
            'candidate_reviewed':False,'independent_tenant_holdout':False}
    (out/'PREPARE_FREEZE.json').write_bytes(_json_bytes(freeze))
    return {'planned_cases':len(QUERIES),'arms':list(result['arms']),
            'freeze':freeze,'out':str(out)}


def _read_frozen(folder:Path):
    freeze=json.loads((folder/'PREPARE_FREEZE.json').read_bytes())
    cb=(folder/'candidate_frozen.json').read_bytes()
    pb=(folder/'holdout_plan.json').read_bytes()
    if sha256(cb).hexdigest()!=freeze.get('candidate_sha256') or sha256(pb).hexdigest()!=freeze.get('plan_sha256'):
        raise ValueError('A frozen artifact changed')
    c=json.loads(cb);p=json.loads(pb)
    if (c.get('format')!=FORMAT+'-candidate' or p.get('format')!=FORMAT+'-plan'
            or c.get('reviewed') is not False or c.get('behavioral_clone') is not False
            or p.get('cases') != [q.public() for q in QUERIES]):
        raise ValueError('Unapproved frozen plan or candidate')
    if not isinstance(c.get('arms'),dict) or 'contract' not in c['arms'] or 'symbolic' not in c['arms'] or 'conditional' not in c['arms']:
        raise ValueError('Three frozen comparison arms required')
    if set(c['arms'])-{'contract','symbolic','conditional','model'}:raise ValueError('Unrecognized prediction arm')
    for arm,preds in c['arms'].items():
        if not isinstance(preds,dict) or set(preds)-BY_ID.keys():raise ValueError('Invalid arm predictions')
        for id,pred in preds.items():_check_pred(BY_ID[id],pred)
    return c,p,freeze


def _truth(q:Query,status:int,payload:Any,*,selected_rows:dict) -> dict:
    shape=safe_shape(payload)
    out={'status':status,'kind':shape['kind']}
    if 'length' in shape:out['length']=shape['length']
    if 'first_item_kind' in shape:out['item_kind']=shape['first_item_kind']
    if isinstance(payload,dict):item=payload
    elif isinstance(payload,list) and payload and isinstance(payload[0],dict):item=payload[0]
    else:item={}
    if q.selected:
        out['field_types']={k:type_name(item[k]) for k in q.selected if k in item}
        if shape['kind']=='array' and item:
            out['item_keys']=sorted(k for k in q.selected if k in item)
    if q.special=='reverse':
        rows=selected_rows.get(q.family,[])
        if len(rows)>=2 and isinstance(payload,list) and payload and isinstance(payload[0],dict):
            # Compare actual identities ONLY IN MEMORY. One boolean escapes.
            eligible=rows[:3]
            out['reverse_matches']=str(payload[0].get('id'))==str(eligible[-1].get('id'))
    if q.special=='ref_match':
        rec=selected_rows.get('_current_ref')
        if rec is not None:
            out['ref_matches']=status==200 and isinstance(payload,dict) and payload.get('ref')==rec
    return out


def _score(pred:dict,truth:dict) -> dict:
    """Shared observable rubric and explicit overclaim accounting.

    Type claims are only evaluable if a field actually exists. Extra claims are
    FALSE, not additional observables, so they cannot inflate the denominator.
    A union type counts as correct *membership* but is penalized for ambiguity
    by 1/size in the optional exact-information score.
    """
    targets={k:v for k,v in truth.items() if k!='field_types'}
    targets.update({'field_types.'+k:v for k,v in truth.get('field_types',{}).items()})
    claims={k:v for k,v in pred.items() if k!='field_types'}
    claims.update({'field_types.'+k:v for k,v in pred.get('field_types',{}).items()})
    checked={}
    correct=0
    information=0.
    for k,value in claims.items():
        true=targets.get(k,'__ABSENT__')
        matches=(true in value if k.startswith('field_types.') and isinstance(value,list)
                 else value==true)
        checked[k]=bool(matches)
        if matches:
            correct+=1
            information+=1./len(value) if k.startswith('field_types.') and isinstance(value,list) else 1.
    target_count=len(targets)
    in_scope=sum(k in targets for k in claims)
    predicted_count=len(claims)
    overclaims=predicted_count-in_scope
    incorrect=predicted_count-correct
    return {'predicted_fields':predicted_count,'correct_fields':correct,
            'observable_fields':target_count,'coverage':round(in_scope/target_count,6) if target_count else 0,
            'accuracy':round(correct/predicted_count,6) if predicted_count else None,
            'utility':round((correct-incorrect)/target_count,6) if target_count else 0,
            'information_score':round((information-incorrect)/target_count,6) if target_count else 0,
            'excess_claims':overclaims,'wrong_claims':incorrect,
            'exact_predicted_fields':bool(claims) and incorrect==0,
            'abstained':predicted_count==0,'field_checks':checked}


def _metrics(rows:list[dict],arm:str) -> dict:
    evaluated=[r for r in rows if arm in r['scores']]
    n=sum(r['scores'][arm]['predicted_fields'] for r in evaluated)
    correct=sum(r['scores'][arm]['correct_fields'] for r in evaluated)
    observable=sum(r['scores'][arm]['observable_fields'] for r in evaluated)
    correctly_claimed_targets=sum(r['scores'][arm]['correct_fields'] for r in evaluated)
    in_scope=sum(r['scores'][arm]['predicted_fields']-r['scores'][arm]['excess_claims'] for r in evaluated)
    wrong=n-correct
    return {'cases':len(evaluated),'predicted_fields':n,'correct_fields':correct,
            'accuracy_on_predictions':round(correct/n,5) if n else None,
            'observable_field_coverage':round(in_scope/observable,5) if observable else 0,
            'correct_target_recall':round(correctly_claimed_targets/observable,5) if observable else 0,
            'net_utility':round((correct-wrong)/observable,5) if observable else 0,
            'information_adjusted_net_utility':round(sum(r['scores'][arm]['information_score']*r['scores'][arm]['observable_fields']
                                                        for r in evaluated)/observable,5) if observable else 0,
            'macro_net_utility':round(sum(r['scores'][arm]['utility'] for r in evaluated)/len(evaluated),5) if evaluated else 0,
            'excess_claims':sum(r['scores'][arm]['excess_claims'] for r in evaluated),
            'wrong_claims':wrong,
            'shared_denominator_observable':observable,
            'full_predicted_matches':sum(r['scores'][arm]['exact_predicted_fields'] for r in evaluated),
            'nontrivial_full_matches':sum(r['scores'][arm]['exact_predicted_fields'] and
                          any(k.startswith('field_types.') or k in {'length','item_keys','item_kind','reverse_matches','ref_matches'}
                              for k in r['scores'][arm]['field_checks']) for r in evaluated),
            'abstentions':sum(r['scores'][arm]['abstained'] for r in evaluated),
            'status_correct':sum(r['scores'][arm]['field_checks'].get('status') is True for r in evaluated),
            'status_predictions':sum('status' in r['scores'][arm]['field_checks'] for r in evaluated),
            'special_checks':sum(x in r['scores'][arm]['field_checks'] for r in evaluated
                                  for x in ('reverse_matches','ref_matches')),
            'special_correct':sum(r['scores'][arm]['field_checks'].get(x) is True for r in evaluated
                                  for x in ('reverse_matches','ref_matches'))}


class Evaluator:
    def __init__(self, client:Any, frozen:str|Path, *, budget:int=32):
        if not getattr(client,'loopback',False):raise ValueError('GET-only loopback client required')
        if type(budget) is not int or not 4<=budget<=MAX_BUDGET:raise ValueError('Invalid read budget')
        self.candidate,self.plan,self.freeze=_read_frozen(Path(frozen))
        self.client=client;self.budget=budget;self.reads=0;self.ledger=None

    def _get(self,id:str,template:str,path:str,predictions:dict):
        if self.reads>=self.budget:raise RuntimeError('GET budget exhausted')
        self.ledger.append({'event':'prediction_committed','index':self.reads,'probe_id':id,
                            'path_template':template,'method':'GET','prediction':predictions})
        try:
            status,data=self.client.request('GET',path)
        except Exception as exc:
            self.ledger.append({'event':'probe_transport_error','index':self.reads,
                                'probe_id':id,'error_type':type(exc).__name__})
            self.reads+=1
            raise RuntimeError('local_oracle_transport_error') from None
        self.ledger.append({'event':'observation_recorded','index':self.reads,'probe_id':id,
                            'status':status,'kind':type_name(data)})
        self.reads+=1
        return status,data

    def run(self,outdir:str|Path,*,execute:bool=False):
        out=Path(outdir)
        if (out/'EVAL_FREEZE.json').exists() or (out/'events.jsonl').exists():
            raise FileExistsError('Existing evaluation; choose another folder')
        report={'format':FORMAT+'-report','phase':'holdout' if execute else 'dry_run',
                'source_discovery_sha256':self.freeze['source_discovery_sha256'],
                'candidate_sha256':self.freeze['candidate_sha256'],
                'plan_sha256':self.freeze['plan_sha256'],
                'arms':list(self.candidate['arms']),
                'model_provider_calls_during_evaluation':0,
                'model_provider_calls_during_prepare':self.freeze['model_provider_calls'],
                'query_holdout':True,'fixed_scoring_rubric':True,'independent_tenant_holdout':False,
                'behavioral_clone':False,'live_writes':0,'raw_values_saved':False,
                'credentials_saved':False,'budget_max':self.budget,'operations_executed':0,
                'selector_reads':0,'selector_counts':{},'cases_configured':len(QUERIES),
                'cases':[],'skipped':[],'stop_reason':None}
        if not execute:
            report['plan_preview']=[q.public() for q in QUERIES]
            report['stop_reason']='dry_run_only_zero_ERP_calls'
            self._write(out,report,None)
            return report
        self.ledger=EvidenceLog(out,enabled=True)
        try:
            self.ledger.append({'event':'run_started','budget':self.budget,
                                'candidate_sha256':self.freeze['candidate_sha256'],
                                'plan_sha256':self.freeze['plan_sha256']})
            rows={}
            # Unscored selectors never update any prediction arm.
            for fam in FAMILIES:
                if self.reads>=self.budget:break
                path=f'/{fam}?limit=20'
                status,payload=self._get('selector_'+fam,path,path,{})
                rows[fam]=payload[:20] if status==200 and isinstance(payload,list) else []
                report['selector_reads']+=1
                report['selector_counts'][fam]=len(rows[fam])
            for q in QUERIES:
                if self.reads>=self.budget:
                    report['skipped'].append({'id':q.id,'reason':'budget_exhausted'});continue
                route=q.route;rec=None
                if '{id}' in route or '{ref}' in route:
                    candidates=rows.get(q.family,[])
                    if q.index>=len(candidates) or not isinstance(candidates[q.index],dict):
                        report['skipped'].append({'id':q.id,'reason':'missing_safe_fixture'});continue
                    rec=candidates[q.index]
                    if '{id}' in route:
                        identifier=rec.get('id')
                        if type(identifier) not in (int,str) or not _ID.fullmatch(str(identifier)):
                            report['skipped'].append({'id':q.id,'reason':'unsafe_or_absent_id'});continue
                        route=route.replace('{id}',str(identifier))
                    if '{ref}' in route:
                        ref=rec.get('ref')
                        if type(ref) is not str or not _REF.fullmatch(ref):
                            report['skipped'].append({'id':q.id,'reason':'unsafe_or_absent_ref'});continue
                        route=route.replace('{ref}',ref)
                arms={arm:predictions.get(q.id,{}) for arm,predictions in self.candidate['arms'].items()}
                status,payload=self._get(q.id,q.route,route,arms)
                ephemeral=dict(rows)
                if q.special=='ref_match':ephemeral['_current_ref']=rec['ref']
                truth=_truth(q,status,payload,selected_rows=ephemeral)
                row={'id':q.id,'evidence_class':q.evidence_class,'family':q.family,
                     'route_template':q.route,'method':'GET',
                     'observation':truth, 'predictions':arms,
                     'scores':{arm:_score(pred,truth) for arm,pred in arms.items()}}
                report['cases'].append(row)
                payload=None;rec=None
            report['stop_reason']='plan_complete' if self.reads<self.budget else 'budget_exhausted'
            self.ledger.append({'event':'run_finished','reads':self.reads,
                                'scored':len(report['cases']),
                                'stop_reason':report['stop_reason']})
        except Exception as exc:
            report['stop_reason']='aborted:'+type(exc).__name__
            self.ledger.append({'event':'run_finished','reads':self.reads,
                                'scored':len(report['cases']),
                                'stop_reason':report['stop_reason']})
        finally:
            self.ledger.close()
        report['operations_executed']=self.reads
        self._write(out,report,self.ledger.last)
        return report

    @staticmethod
    def _write(out:Path,report:dict,ledger_hash:str|None):
        report['cases_scored']=len(report['cases'])
        report['scores']={arm:_metrics(report['cases'],arm) for arm in report['arms']}
        report['by_evidence_class']={kind:{arm:_metrics([r for r in report['cases']
                        if r['evidence_class']==kind],arm) for arm in report['arms']}
                        for kind in sorted(set(r['evidence_class'] for r in report['cases']))}
        report['conclusion']='READ_ONLY_SHAPE_BEHAVIOR_NOT_CAUSAL_FIDELITY'
        out.mkdir(parents=True,exist_ok=True)
        data=_json_bytes(report)
        (out/'holdout_report.json').write_bytes(data)
        freeze={'format':FORMAT+'-eval-freeze','phase':report['phase'],
                'report_sha256':sha256(data).hexdigest(),
                'candidate_sha256':report['candidate_sha256'],
                'plan_sha256':report['plan_sha256'],
                'evidence_chain_sha256':ledger_hash,
                'predictions_precommitted':ledger_hash is not None,
                'independent_tenant_holdout':False,'externally_timestamped':False,
                'live_writes':0}
        (out/'EVAL_FREEZE.json').write_bytes(_json_bytes(freeze))


def verify(frozen:str|Path,evaluation:str|Path,discovery:str|Path|None=None) -> dict:
    cand,plan,f=_read_frozen(Path(frozen))
    if discovery is not None and sha256(Path(discovery).read_bytes()).hexdigest()!=f['source_discovery_sha256']:
        raise ValueError('Discovery source changed')
    ev=Path(evaluation)
    ef=json.loads((ev/'EVAL_FREEZE.json').read_bytes())
    data=(ev/'holdout_report.json').read_bytes()
    report=json.loads(data)
    if (sha256(data).hexdigest()!=ef.get('report_sha256') or
        ef.get('candidate_sha256')!=f['candidate_sha256'] or
        ef.get('plan_sha256')!=f['plan_sha256'] or
        report.get('candidate_sha256')!=f['candidate_sha256'] or
        report.get('plan_sha256')!=f['plan_sha256']):
        raise ValueError('Report or freeze digest mismatch')
    if report.get('live_writes')!=0 or report.get('raw_values_saved') is not False:
        raise ValueError('Unapproved evidence provenance')
    if report.get('arms')!=list(cand['arms']):
        raise ValueError('Frozen prediction arms changed')
    seen=set()
    for c in report.get('cases',[]):
        identifier=c.get('id')
        if (identifier not in BY_ID or identifier in seen or
            c.get('method')!='GET' or c.get('route_template')!=BY_ID[identifier].route):
            raise ValueError('Unapproved or duplicate scored case')
        seen.add(identifier)
        frozen_preds={arm:preds.get(identifier,{}) for arm,preds in cand['arms'].items()}
        if c.get('predictions')!=frozen_preds:
            raise ValueError('Evaluation altered the frozen prediction')
        if c.get('scores')!={arm:_score(pred, c['observation'])
                            for arm,pred in frozen_preds.items()}:
            raise ValueError('Per-case score is not reproducible')
    if (report.get('cases_scored')!=len(report['cases']) or
        report.get('scores')!={arm:_metrics(report['cases'],arm) for arm in cand['arms']}):
        raise ValueError('Aggregate score is not reproducible')
    if report['phase']=='dry_run':
        if report.get('operations_executed')!=0:raise ValueError('Dry run may not contact oracle')
        return {'status':'PASS_OFFLINE_PREVIEW','reads_verified':0,'external_timestamped':False}
    previous='0'*64;num=0;index=0;pending=None;starts=ends=0
    valid_ids=set(BY_ID)|{'selector_'+f for f in FAMILIES}
    scored_events=[]
    for line in (ev/'events.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(line);sig=row.pop('sha256')
        if (row.get('seq')!=num+1 or row.get('prev_sha256')!=previous
            or hash_json(row)!=sig):
            raise ValueError('Event chain integrity error')
        previous=sig;num+=1
        if row['event']=='run_started':
            if starts or num!=1 or row.get('candidate_sha256')!=f['candidate_sha256']:
                raise ValueError('Unexpected run start event')
            starts+=1
        elif row['event']=='run_finished':
            if pending is not None or ends or row.get('reads')!=index:
                raise ValueError('Unexpected finish event')
            if row.get('scored')!=len(report['cases']):raise ValueError('Scored case count mismatch')
            ends+=1
        elif row['event']=='prediction_committed':
            if (pending is not None or row['index']!=index or
                row['probe_id'] not in valid_ids or row.get('method')!='GET'
                or not starts or ends):
                raise ValueError('Missing or malformed precommit')
            if row['probe_id'] in BY_ID:
                if row.get('path_template')!=BY_ID[row['probe_id']].route:
                    raise ValueError('Unapproved query path')
                if row.get('prediction')!={arm:preds.get(row['probe_id'],{})
                                           for arm,preds in cand['arms'].items()}:
                    raise ValueError('Not the frozen prediction')
                scored_events.append(row['probe_id'])
            else:
                if (row.get('prediction')!={} or row.get('path_template')!=
                        '/'+row['probe_id'].removeprefix('selector_')+'?limit=20'):
                    raise ValueError('Selector precommit was altered')
            pending=row['probe_id']
        elif row['event'] in ('observation_recorded','probe_transport_error'):
            if row.get('probe_id')!=pending or row['index']!=index:
                raise ValueError('Observation lacks prior prediction')
            if row['event']=='observation_recorded' and pending in BY_ID:
                case=next((c for c in report['cases'] if c['id']==pending),None)
                if (case is None or row.get('status')!=case['observation']['status'] or
                    row.get('kind')!=case['observation']['kind']):
                    raise ValueError('Observed response does not match logged score')
            pending=None;index+=1
        else:
            raise ValueError('Unexpected event type')
    if (pending is not None or previous!=ef.get('evidence_chain_sha256')
        or starts!=1 or ends!=1 or index!=report['operations_executed']
        or scored_events != [c['id'] for c in report['cases']]):
        raise ValueError('Incomplete evaluation chain or scored case ordering')
    return {'status':'PASS_LOCAL_HASH_AND_ORDER_ONLY','reads_verified':index,
            'live_writes':0,'externally_timestamped':False,
            'independent_tenant_holdout':False,'behavioral_clone_verified':False}


def main(argv=None):
    p=argparse.ArgumentParser(description='Gate2G fixed-rubric, frozen, GET-only Dolibarr benchmark')
    s=p.add_subparsers(dest='command',required=True)
    pr=s.add_parser('prepare',help='Offline-only: freeze three-arm candidate')
    pr.add_argument('--discovery',required=True)
    pr.add_argument('--swagger',required=True)
    pr.add_argument('--development',help='Prior revealed Gate2F development report (never current holdout)')
    pr.add_argument('--out',default='build/gate2f_frozen')
    pr.add_argument('--execute-model',action='store_true',help='Optional real model call; no ERP I/O')
    pr.add_argument('--model',default='gpt-6')
    ev=s.add_parser('evaluate',help='Dry-run unless --execute-readonly explicitly provided')
    ev.add_argument('--frozen',required=True)
    ev.add_argument('--out',default='build/gate2f_preview')
    ev.add_argument('--budget',type=int,default=32)
    ev.add_argument('--base-url',default='http://127.0.0.1:8088')
    ev.add_argument('--execute-readonly',action='store_true')
    vv=s.add_parser('verify')
    vv.add_argument('--frozen',required=True)
    vv.add_argument('--evaluation',required=True)
    vv.add_argument('--discovery')
    a=p.parse_args(argv)
    try:
        if a.command=='prepare':
            advisor=OpenAI2FPredictor(a.model) if a.execute_model else None
            result=prepare(a.discovery,a.swagger,a.out,model_advisor=advisor,development=a.development)
        elif a.command=='evaluate':
            u=urlparse(a.base_url)
            if (u.scheme not in ('http','https') or not _local_host(u.hostname or '') or
                u.username or u.password or u.query or u.fragment or
                u.path.rstrip('/') not in ('','/api/index.php')):
                p.error('ERP destination must be root or /api/index.php on loopback host')
            key=os.environ.get('DOLIBARR_API_KEY') if a.execute_readonly else None
            if a.execute_readonly and not key:p.error('DOLIBARR_API_KEY required')
            evaluator=Evaluator(DolibarrClient(a.base_url,key),a.frozen,budget=a.budget)
            report=evaluator.run(a.out,execute=a.execute_readonly)
            result={'phase':report['phase'],'reads':report['operations_executed'],
                    'scored':report['cases_scored'],'scores':report['scores'],
                    'skipped':len(report['skipped']),'stop_reason':report['stop_reason'],
                    'out':a.out}
        else:result=verify(a.frozen,a.evaluation,a.discovery)
    except (ValueError,RuntimeError,FileExistsError,KeyError) as e:
        # No runtime identifiers or URLs in client-error messages.
        p.error(type(e).__name__ if a.command=='evaluate' else type(e).__name__+': '+str(e)[:120])
    print(json.dumps(result,indent=2,sort_keys=True))
    if a.command=='evaluate' and result['stop_reason'].startswith('aborted:'):return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
