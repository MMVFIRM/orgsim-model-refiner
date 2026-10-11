"""Gate 2E: frozen, same-role, read-only ERP response/relationship predictions.

prepare: produces one immutable prediction bundle from existing Gate 2D evidence.
evaluate: selects fixture entities in memory, records precommit before every GET,
          and scores identical held-out queries for symbolic and optional LLM arms.

Not a verified behavioral clone: does not infer unseen writes, fiscal calculations,
or side effects. The holdout uses fresh query variants on the same tenant, not an
independent tenant or an independent business-workflow benchmark.
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
from .gate2d_adaptive import (EvidenceLog, canonical, hash_json, safe_shape,
                                type_name, OpenAIProbeAdvisor)
from .gate2b_readonly import _ID, _REF

FORMAT = 'orgsim-gate2e-predictive-read-v1'
MAX_READS = 40
MAX_SELECTOR_ROWS = 20
FAMILIES = ('thirdparties', 'products', 'orders', 'invoices')
FIELDS = {
    'thirdparties': ('id', 'name', 'client', 'ref', 'country_code', 'status'),
    'products': ('id', 'ref', 'label', 'price', 'type', 'status'),
    'orders': ('id', 'socid', 'ref', 'status', 'total_ht', 'lines'),
    'invoices': ('id', 'socid', 'ref', 'status', 'total_ht', 'paye', 'lines'),
    'orders_lines': ('id', 'fk_product', 'qty', 'subprice', 'total_ht'),
    'invoices_lines': ('id', 'fk_product', 'qty', 'subprice', 'total_ht'),
}
VALID_KINDS = {'object', 'array', 'number', 'string', 'boolean', 'null', 'unknown'}


@dataclass(frozen=True)
class Case:
    id: str
    family: str
    category: str
    source_profile: str
    template: str                 # public path, no identifiers
    identity_index: int = 0      # index 1 = distinct from Gate 2D's first item
    selected_fields: tuple[str, ...] = ()
    relation: str = ''            # '', order_customer, invoice_customer, line_product
    evidence_split: str = 'query_variant'

    def public(self) -> dict:
        row = asdict(self)
        row['selected_fields'] = list(self.selected_fields)
        return row


def _source_report(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get('format') != 'orgsim-gate2d-adaptive-discovery-v1' or data.get('phase') != 'discovery':
        raise ValueError('Need a Gate 2D discovery_report.json')
    if data.get('live_writes') != 0 or data.get('raw_values_saved') is not False:
        raise ValueError('Discovery provenance does not certify read-only, value-free observations')
    if not isinstance(data.get('probes'), list):
        raise ValueError('Malformed discovery evidence')
    return data, sha256(raw).hexdigest()


def _profiles(report: dict) -> dict:
    # Read ONLY the known approved probe IDs. Never load scalar ERP values into output.
    known = {
        f'{fam}_{s}' for fam in FAMILIES for s in ('list', 'detail')
    } | {'products_stock', 'products_categories', 'orders_shipment',
         'orders_lines', 'invoices_lines', 'product_by_ref',
         'thirdparties_outstanding_orders', 'thirdparties_outstanding_invoices'}
    result = {}
    for p in report['probes']:
        if not isinstance(p, dict) or p.get('id') not in known:
            continue
        if p.get('method') != 'GET' or type(p.get('status')) is not int:
            continue
        s = p.get('shape')
        if not isinstance(s, dict) or s.get('kind') not in VALID_KINDS:
            continue
        result[p['id']] = {'status': p['status'], 'kind': s['kind'],
                           'length': s.get('length'),
                           'item_kind': s.get('first_item_kind'),
                           'field_types': {k: v for k, v in (s.get('types') or s.get('first_types') or {}).items()
                                           if isinstance(k, str) and isinstance(v, str)},
                           'field_keys': list(s.get('keys') or s.get('first_keys') or [])}
    return result


def _cases(profiles: dict) -> list[Case]:
    cases = []
    # The baseline Gate 2D uses limit=3 and at most the first record. Gate 2E
    # asks for different projections and (when possible) distinct record IDs.
    for fam in FAMILIES:
        anchor = profiles.get(fam + '_list')
        if not anchor or anchor['status'] != 200 or anchor['kind'] != 'array':
            continue
        fields = {
            'thirdparties': ('id', 'name', 'client'),
            'products': ('id', 'ref', 'label', 'price'),
            'orders': ('id', 'socid', 'ref', 'status'),
            'invoices': ('id', 'socid', 'ref', 'status'),
        }[fam]
        cases.append(Case(f'{fam}_projected_limit2', fam, 'projection', fam+'_list',
                          f'/{fam}?limit=2&properties=' + ','.join(fields),
                          selected_fields=fields))
        # Exact singleton readback repeats the discovery request: do NOT score.
        n = anchor.get('length') or 0
        if n >= 2 and profiles.get(fam+'_detail', {}).get('status') == 200:
            cases.append(Case(f'{fam}_detail_holdout', fam, 'detail', fam+'_detail',
                              f'/{fam}/{{id}}', identity_index=1,
                              selected_fields=FIELDS[fam], evidence_split='new_identity'))
    if (profiles.get('orders_lines', {}).get('status') == 200 and
        (profiles.get('orders_list', {}).get('length') or 0) >= 2):
        cases.append(Case('orders_lines_holdout', 'orders', 'related', 'orders_lines',
                          '/orders/{id}/lines', identity_index=1,
                          selected_fields=FIELDS['orders_lines'], evidence_split='new_identity'))
    if (profiles.get('invoices_lines', {}).get('status') == 200 and
        (profiles.get('invoices_list', {}).get('length') or 0) >= 2):
        cases.append(Case('invoices_lines_holdout', 'invoices', 'related', 'invoices_lines',
                          '/invoices/{id}/lines', identity_index=1,
                          selected_fields=FIELDS['invoices_lines'], evidence_split='new_identity'))
    if (profiles.get('product_by_ref', {}).get('status') == 200 and
        (profiles.get('products_list', {}).get('length') or 0) >= 2):
        cases.append(Case('products_lookup_by_ref', 'products', 'alternate', 'product_by_ref',
                          '/products/ref/{ref}', identity_index=1,
                          selected_fields=FIELDS['products'], evidence_split='new_identity'))
    # Access control predictions are scoped to the same API user and tenant.
    for case_id, fam, key, path in (
        ('products_stock_permission', 'products', 'products_stock',
         '/products/{id}/stock?selected_warehouse_id=0'),
        ('products_categories_permission', 'products', 'products_categories',
         '/products/{id}/categories?limit=2'),
        ('orders_shipment_permission', 'orders', 'orders_shipment',
         '/orders/{id}/shipment'),
    ):
        if profiles.get(key, {}).get('status') == 403:
            cnt = profiles.get(fam+'_list', {}).get('length') or 0
            index = 1 if fam == 'orders' and cnt >= 2 else 0
            cases.append(Case(case_id, fam, 'permission', key, path,
                              index, evidence_split='new_identity' if index else 'query_variant'))
    # Structural hypotheses from API naming, NOT previously verified relationships.
    if profiles.get('orders_list', {}).get('status') == 200:
        cases.append(Case('orders_customer_fk', 'orders', 'relationship', 'thirdparties_detail',
                          '/thirdparties/{socid}', 1 if (profiles['orders_list'].get('length') or 0)>=2 else 0,
                          relation='order_customer', evidence_split='new_identity_relation'))
    if profiles.get('invoices_list', {}).get('status') == 200:
        cases.append(Case('invoices_customer_fk', 'invoices', 'relationship', 'thirdparties_detail',
                          '/thirdparties/{socid}', relation='invoice_customer',
                          evidence_split='relationship_consistency_reuse'))
    if profiles.get('orders_lines', {}).get('status') == 200:
        cases.append(Case('order_line_product_fk', 'orders', 'relationship', 'products_detail',
                          '/products/{fk_product}', 1 if (profiles.get('orders_list',{}).get('length') or 0)>=2 else 0,
                          relation='line_product', evidence_split='relationship_consistency_reuse'))
    return cases


def _known_type_fields(profile: dict, fields: tuple[str,...]) -> dict[str,str]:
    # Null in one sample is not a stable datatype: ABSTAIN instead of overfit.
    return {k: v for k in fields if (v := profile.get('field_types', {}).get(k)) in VALID_KINDS-{'null','unknown'}}


def _symbolic(case: Case, profile: dict, all_profiles: dict) -> dict:
    pred: dict[str,Any] = {}
    if profile:
        pred['status'] = profile['status']
        pred['kind'] = profile['kind']
    if case.category == 'projection':
        pred['status'] = 200
        pred['kind'] = 'array'
        count = profile.get('length')
        if type(count) is int:
            pred['length'] = min(count, 2)
            if count:
                pred['item_kind'] = 'object'
                pred['item_keys'] = sorted(k for k in case.selected_fields
                                          if k in profile.get('field_keys', []))
        pred['field_types'] = _known_type_fields(profile, case.selected_fields)
    elif case.category in {'detail','related','alternate'}:
        # Predict selected, demonstrated non-null type fields. Some may vary by entity.
        if profile.get('item_kind'):
            pred['item_kind'] = profile['item_kind']
        pred['field_types'] = _known_type_fields(profile, case.selected_fields)
    elif case.category == 'permission':
        # Narrow endpoint + same user hypothesis. Never infer global ACL semantics.
        pred['status'] = 403
        pred['kind'] = 'object'
    elif case.category == 'relationship':
        pred['relation_exists'] = True    # HYPOTHESIS: FK points to an accessible record
        pred['status'] = 200            # HYPOTHESIS: related record is GET-accessible
        pred['kind'] = 'object'
    return validate_prediction(pred, case)


def validate_prediction(pred: dict, case: Case) -> dict:
    if not isinstance(pred, dict): raise ValueError('Prediction must be an object')
    allowed = {'status','kind','length','item_kind','item_keys','field_types','relation_exists'}
    if set(pred) - allowed: raise ValueError('Prediction contains unsupported keys')
    out = {}
    if 'status' in pred:
        if type(pred['status']) is not int or pred['status'] < 100 or pred['status'] > 599:
            raise ValueError('Invalid predicted HTTP status')
        out['status'] = pred['status']
    for attr in ('kind','item_kind'):
        if attr in pred:
            if pred[attr] not in VALID_KINDS: raise ValueError('Invalid response kind')
            out[attr] = pred[attr]
    if 'length' in pred:
        if type(pred['length']) is not int or not 0 <= pred['length'] <= 20:
            raise ValueError('Invalid predicted list length')
        out['length'] = pred['length']
    if 'item_keys' in pred:
        if not isinstance(pred['item_keys'], list) or len(pred['item_keys']) > 12 or any(
            type(k) is not str or k not in case.selected_fields for k in pred['item_keys']):
            raise ValueError('Invalid predicted item keys')
        out['item_keys'] = sorted(set(pred['item_keys']))
    if 'field_types' in pred:
        types = pred['field_types']
        if not isinstance(types, dict) or len(types)>12 or any(
            k not in case.selected_fields or v not in VALID_KINDS for k,v in types.items()):
            raise ValueError('Invalid predicted field types')
        out['field_types'] = {k:types[k] for k in sorted(types)}
    if 'relation_exists' in pred:
        if case.category != 'relationship' or type(pred['relation_exists']) is not bool:
            raise ValueError('Invalid relationship prediction')
        out['relation_exists'] = pred['relation_exists']
    return out


def _allowed_evidence(profiles: dict) -> dict:
    """Strict, value-free summary for opt-in remote model. No record identifiers."""
    return {key: {'status': p['status'], 'kind': p['kind'], 'length': p['length'],
                   'item_kind': p['item_kind'],
                   'field_types': {k:v for k,v in p['field_types'].items()
                                   if k in {name for group in FIELDS.values() for name in group}}}
                  for key,p in sorted(profiles.items())}


class OpenAIBlindPredictor:
    """Explicit prepare-time only call. Never contacts the user's ERP."""
    def __init__(self, model: str):
        self.model = model
        self._existing = OpenAIProbeAdvisor(model)

    def predict_many(self, cases: list[Case], evidence: dict) -> dict:
        from urllib.request import Request
        import urllib.error
        system = (
            'Predict held-out read-only ERP API behavior from permitted, non-value evidence. '
            'Output ONE JSON object {"predictions":{"case_id":{...}}} with no other text. '
            'Allowed fields per case: status integer, kind, item_kind, length integer, '
            'item_keys list, field_types object, and relation_exists boolean for relation cases. '
            'Do not invent constraints as facts. Omit any unsupported predictions. '
            'Never output credentials, actual entity IDs, scalar ERP values, or new routes. '
            'A single null type observation is not a stable type.'
        )
        prompt={'evidence': evidence, 'role_scope': 'same Dolibarr API test user',
                'heldout_query_templates': [c.public() for c in cases],
                'target': 'predict schema and permissions, do not infer workflow side effects'}
        body={'model': self.model, 'store': False, 'max_output_tokens': 4000,
              'input':[{'role':'system','content':system},
                       {'role':'user','content':canonical(prompt)}]}
        req=Request('https://api.openai.com/v1/responses', method='POST',
                    data=canonical(body).encode(),
                    headers={'Authorization':'Bearer '+self._existing._key,
                             'Content-Type':'application/json'})
        with self._existing._opener.open(req, timeout=90) as response:
            data=response.read(1_000_001)
        if len(data)>1_000_000:raise ValueError('Model output too large')
        res=json.loads(data)
        message=res.get('output_text') or ''.join(
            p.get('text','') for item in res.get('output',[])
            for p in item.get('content',[]) if p.get('type')=='output_text')
        if message.lstrip().startswith('```'):
            message=message.strip().split('\n',1)[-1].rsplit('```',1)[0].strip()
        parsed=json.loads(message)
        if not isinstance(parsed,dict) or set(parsed)!={'predictions'} or not isinstance(parsed['predictions'],dict):
            raise ValueError('Model output must contain predictions only')
        by_id={c.id:c for c in cases}
        if set(parsed['predictions'])-set(by_id):
            raise ValueError('Model supplied an unauthorized case')
        return {id:validate_prediction(pred,by_id[id]) for id,pred in parsed['predictions'].items()}


def prepare(source_path: str | Path, outdir: str | Path,
            *, model_advisor: Any = None) -> dict:
    """Freeze all cases + predictions BEFORE any evaluation-time network traffic."""
    report, source_hash = _source_report(Path(source_path))
    profiles=_profiles(report)
    cases=_cases(profiles)
    if not cases:
        raise ValueError('Discovery report has no suitable read profiles for Gate 2E')
    symbolic={c.id:_symbolic(c,profiles.get(c.source_profile,{}),profiles) for c in cases}
    llm=None
    if model_advisor is not None:
        llm=model_advisor.predict_many(cases,_allowed_evidence(profiles))
        if not isinstance(llm,dict) or set(llm)-{c.id for c in cases}:
            raise ValueError('Model adapter returned unauthorized prediction IDs')
        by_id={c.id:c for c in cases}
        llm={cid:validate_prediction(value,by_id[cid]) for cid,value in llm.items()}
    candidate={
        'format': FORMAT+'-candidate', 'version':'0.2.4', 'reviewed':False,
        'behavioral_clone':False, 'role_scope':'same_API_user_as_discovery',
        'source_discovery_sha256':source_hash,
        'role_evidence': {k:{'status':p['status'], 'kind':p['kind']}
                          for k,p in profiles.items()},
        'relationship_hypotheses':[
            {'from':'orders.socid','to':'thirdparties.id','verified':False},
            {'from':'invoices.socid','to':'thirdparties.id','verified':False},
            {'from':'orders_lines.fk_product','to':'products.id','verified':False}],
        'constraints': ['no writes', 'null observations do not imply permanent nulls',
                        'no transition or side-effect assumptions',
                        'same-tenant fresh queries are not an independent tenant holdout'],
        'predictors': {'symbolic':symbolic, **({'model':llm} if llm is not None else {})},
        'model_calls_made_during_prepare':int(llm is not None),
    }
    plan={'format':FORMAT+'-plan', 'phase':'frozen_query_holdout',
          'max_selector_rows':MAX_SELECTOR_ROWS,
          'query_holdout':True, 'independent_tenant_holdout':False,
          'source_discovery_sha256':source_hash,
          'cases':[c.public() for c in cases]}
    out=Path(outdir)
    if (out/'PREPARE_FREEZE.json').exists():raise FileExistsError('Preparation is already frozen; use a new directory')
    out.mkdir(parents=True,exist_ok=True)
    cb=(json.dumps(candidate,indent=2,sort_keys=True)+'\n').encode()
    pb=(json.dumps(plan,indent=2,sort_keys=True)+'\n').encode()
    (out/'candidate_frozen.json').write_bytes(cb)
    (out/'holdout_plan.json').write_bytes(pb)
    freeze={'format':FORMAT+'-prepare-freeze','source_discovery_sha256':source_hash,
            'candidate_sha256':sha256(cb).hexdigest(),
            'plan_sha256':sha256(pb).hexdigest(),
            'query_holdout':True, 'independent_tenant_holdout':False,
            'reads_executed':0,'live_writes':0,
            'model_provider_calls':int(llm is not None),
            'candidate_reviewed':False}
    (out/'PREPARE_FREEZE.json').write_text(json.dumps(freeze,indent=2,sort_keys=True)+'\n')
    return {'prepared':len(cases),'model_predictions':len(llm or {}), 'freeze':freeze,
            'out':str(out)}


def _read_frozen(folder: Path) -> tuple[dict, dict, dict]:
    freeze=json.loads((folder/'PREPARE_FREEZE.json').read_text(encoding='utf-8'))
    c_bytes=(folder/'candidate_frozen.json').read_bytes()
    p_bytes=(folder/'holdout_plan.json').read_bytes()
    if sha256(c_bytes).hexdigest()!=freeze.get('candidate_sha256') or sha256(p_bytes).hexdigest()!=freeze.get('plan_sha256'):
        raise ValueError('Gate 2E source freeze mismatch; do not repair frozen predictions after testing')
    candidate=json.loads(c_bytes)
    plan=json.loads(p_bytes)
    if candidate.get('format')!=FORMAT+'-candidate' or plan.get('format')!=FORMAT+'-plan':
        raise ValueError('Unrecognized freeze format')
    if candidate.get('source_discovery_sha256')!=plan.get('source_discovery_sha256'):
        raise ValueError('Source discovery mismatch')
    if candidate.get('reviewed') is not False or candidate.get('behavioral_clone') is not False:
        raise ValueError('Invalid candidate provenance')
    return candidate,plan,freeze


def _case_from_public(row: dict) -> Case:
    allowed=set(Case.__dataclass_fields__)
    if not isinstance(row, dict) or set(row)!=allowed:
        raise ValueError('Unexpected holdout case schema')
    row=dict(row)
    row['selected_fields']=tuple(row['selected_fields'])
    c=Case(**row)
    if not isinstance(c.id,str) or not c.id.isascii() or len(c.id)>80 or not c.id.replace('_','').isalnum():
        raise ValueError('Unsafe case identifier')
    if c.family not in FAMILIES or c.identity_index not in (0,1) or c.category not in {
        'projection','detail','related','alternate','permission','relationship'}:
        raise ValueError('Unexpected holdout selector')
    if set(c.selected_fields)-set(FIELDS.get(c.family,())+FIELDS.get(c.family+'_lines',())):
        raise ValueError('Unexpected prediction fields')
    # Restrict route templates independently of freeze checks. Never let a model
    # insert a URL/method or sensitive API endpoint into the local evaluator.
    allowed_templates={
        'thirdparties': {'/thirdparties/{id}','/thirdparties/{socid}',
                         '/thirdparties?limit=2&properties=id,name,client'},
        'products': {'/products/{id}','/products/ref/{ref}',
                     '/products?limit=2&properties=id,ref,label,price',
                     '/products/{id}/stock?selected_warehouse_id=0',
                     '/products/{id}/categories?limit=2', '/products/{fk_product}'},
        'orders': {'/orders/{id}', '/orders/{id}/lines','/orders/{id}/shipment',
                   '/orders?limit=2&properties=id,socid,ref,status',
                   '/thirdparties/{socid}','/products/{fk_product}'},
        'invoices': {'/invoices/{id}','/invoices/{id}/lines',
                     '/invoices?limit=2&properties=id,socid,ref,status',
                     '/thirdparties/{socid}'},
    }
    if c.template not in allowed_templates[c.family]:
        raise ValueError('Holdout plan contains an unapproved API path')
    return c


def _selector_record(rows: list, index: int) -> dict | None:
    if index >= len(rows): return None
    item=rows[index]
    if not isinstance(item,dict):return None
    if type(item.get('id')) not in (int,str) or not _ID.fullmatch(str(item['id'])):
        return None
    return item


def _materialize_path(c: Case, rows: dict[str,list]) -> tuple[str | None,str | None]:
    if c.category=='projection':return c.template,None
    rec=_selector_record(rows.get(c.family,[]),c.identity_index)
    if rec is None:
        return None,'insufficient_safe_fixture_records_or_ids'
    if '{id}' in c.template:
        return c.template.replace('{id}',str(rec['id'])),None
    if '{ref}' in c.template:
        ref=rec.get('ref')
        if not isinstance(ref,str) or not _REF.fullmatch(ref):
            return None,'unavailable_or_unsafe_product_reference'
        return c.template.replace('{ref}',ref),None
    if '{socid}' in c.template:
        socid=rec.get('socid')
        if type(socid) not in (int,str) or not _ID.fullmatch(str(socid)):
            return None,'missing_or_invalid_customer_fk'
        return c.template.replace('{socid}',str(socid)),None
    return None,'relation_requires_oracle_side_selector'


def _semantic_truth(c: Case, status: int, payload: Any,
                    *, relation_exists: bool | None = None) -> dict:
    shape=safe_shape(payload)
    res={'status':status, 'kind':shape['kind']}
    if 'length' in shape:res['length']=shape['length']
    if 'first_item_kind' in shape:res['item_kind']=shape['first_item_kind']
    if c.category=='projection' and shape['kind']=='array' and payload and isinstance(payload[0],dict):
        res['item_keys']=sorted(k for k in payload[0] if k in c.selected_fields)
    if shape['kind']=='object': target=payload
    elif shape['kind']=='array' and payload and isinstance(payload[0],dict): target=payload[0]
    else: target={}
    res['field_types']={k:type_name(target[k]) for k in c.selected_fields if k in target}
    if relation_exists is not None:
        res['relation_exists']=relation_exists
    return res


def _score_one(c: Case, pred: dict, observed: dict) -> dict:
    """Compare all predictions; abstention never counts as success."""
    checks={}
    for name in ('status','kind','length','item_kind','item_keys','relation_exists'):
        if name in pred:
            checks[name]=pred[name]==observed.get(name)
    for k,v in pred.get('field_types',{}).items():
        checks['field_types.'+k] = v==observed['field_types'].get(k)
    # Denominator includes every observable slot and every attempted slot;
    # missing predictions lower coverage and invented fields lower precision.
    possible=len(set(x for x in ('status','kind','length','item_kind','item_keys','relation_exists')
                     if x in observed) | set(k for k in checks if not k.startswith('field_types.')))
    possible+=len(set(observed['field_types']) | set(pred.get('field_types',{})))
    substantive=any(k not in {'status','kind','item_kind'} for k in checks)
    return {'predicted_fields':len(checks), 'correct_fields':sum(checks.values()),
            'observable_fields':possible, 'coverage':round(len(checks)/possible,4) if possible else 0,
            'full_match':bool(checks) and all(checks.values()),
            'substantive_prediction':substantive,
            'substantive_full_match':substantive and all(checks.values()),
            'fields_scored':checks}


def _metrics(rows:list[dict], arm:str) -> dict:
    predicted=correct=possible=full=substantive=0
    count=0
    perms=[]; relations=[]; status=[]; schema=[]
    for row in rows:
        if row.get('not_scored'):continue
        count+=1
        score=row['scores'][arm]
        predicted+=score['predicted_fields']
        correct+=score['correct_fields']
        possible+=score['observable_fields']
        full+=int(score['full_match'])
        substantive+=int(score['substantive_full_match'])
        checks=score['fields_scored']
        if 'status' in checks:status.append(checks['status'])
        schema.extend(v for k,v in checks.items() if k.startswith('field_types.'))
        if row['category']=='permission':perms.append(checks.get('status',False))
        if row['category']=='relationship':relations.append(checks.get('relation_exists',False))
    return {'cases_scored':count,'field_matches':correct,'fields_predicted':predicted,
            'fields_observable':possible,
            'selective_precision':round(correct/predicted,4) if predicted else None,
            'field_coverage':round(predicted/possible,4) if possible else 0,
            'full_predicted_field_matches':full,'nontrivial_full_matches':substantive,
            'status_accuracy':round(sum(status)/len(status),4) if status else None,
            'selected_type_accuracy':round(sum(schema)/len(schema),4) if schema else None,
            'permission_status_accuracy':round(sum(perms)/len(perms),4) if perms else None,
            'relationship_accuracy':round(sum(relations)/len(relations),4) if relations else None,
            'permission_cases_scored':len(perms), 'relationships_scored':len(relations),
            'abstentions':sum(row['scores'][arm]['predicted_fields']==0 for row in rows
                               if not row.get('not_scored'))}


class Evaluator:
    def __init__(self, client:Any, frozen:str|Path, *, budget:int=32):
        if not getattr(client,'loopback',False):raise ValueError('Gate 2E is loopback-only')
        if type(budget) is not int or not 4<=budget<=MAX_READS:raise ValueError('Invalid GET budget')
        self.client=client
        self.folder=Path(frozen)
        self.candidate,self.plan,self.freeze=_read_frozen(self.folder)
        self.cases=[_case_from_public(c) for c in self.plan['cases']]
        self.by_id={c.id:c for c in self.cases}
        if len(self.by_id)!=len(self.cases):raise ValueError('Duplicate holdout case')
        for arm,preds in self.candidate['predictors'].items():
            if arm not in ('symbolic','model') or not isinstance(preds,dict) or set(preds)-set(self.by_id):
                raise ValueError('Unrecognized prediction arm')
            for cid,pred in preds.items():validate_prediction(pred,self.by_id[cid])
        self.budget=budget
        self.reads=0
        self.ledger=None

    def _oracle_get(self, test_id:str, template:str, path:str, pred:dict) -> tuple[int,Any]:
        if self.reads>=self.budget:raise RuntimeError('read_budget_exhausted')
        # Every oracle operation is logged as precommitted before network I/O.
        # Runtime paths (IDs and foreign keys) are not logged.
        self.ledger.append({'event':'prediction_committed','index':self.reads,
                            'probe_id':test_id, 'path_template':template,
                            'method':'GET','prediction':pred})
        try:
            status,payload=self.client.request('GET',path)
        except Exception as exc:
            self.ledger.append({'event':'probe_transport_error','index':self.reads,
                                'probe_id':test_id,'error_type':type(exc).__name__})
            self.reads+=1
            raise RuntimeError('oracle_transport_error:'+type(exc).__name__) from None
        self.ledger.append({'event':'observation_recorded','index':self.reads,
                            'probe_id':test_id,'status':status,
                            'shape':safe_shape(payload)})
        self.reads+=1
        return status,payload

    def _selectors(self, report:dict) -> dict[str,list]:
        # Extra GETs to choose entities are unscored, and cannot update any model.
        rows={}
        for family in FAMILIES:
            if not any(c.family==family for c in self.cases):continue
            if self.reads>=self.budget:break
            path=f'/{family}?limit={MAX_SELECTOR_ROWS}'
            status,payload=self._oracle_get('selector_'+family, path,path,{})
            if status==200 and isinstance(payload,list):
                rows[family]=payload[:MAX_SELECTOR_ROWS]
            else:rows[family]=[]
            report['selector_reads']+=1
            report['selector_counts'][family] = len(rows[family])
        return rows

    def run(self, outdir:str|Path, *, execute:bool=False) -> dict:
        out=Path(outdir)
        if (out/'events.jsonl').exists() or (out/'EVAL_FREEZE.json').exists():
            raise FileExistsError('Existing Gate 2E result; choose a new folder')
        report={'format':FORMAT+'-report', 'phase':'holdout' if execute else 'dry_run',
                'evidence':'LIVE_DOLIBARR' if execute else 'OFFLINE_PLAN_ONLY',
                'query_holdout':True,'independent_tenant_holdout':False,
                'same_role_required':True, 'candidate_reviewed':False,
                'behavioral_clone':False,'live_writes':0,
                'raw_values_saved':False,'credentials_saved':False,
                'budget_max':self.budget,'operations_executed':0,
                'selector_reads':0,'selector_counts':{},
                'source_discovery_sha256':self.freeze['source_discovery_sha256'],
                'candidate_sha256':self.freeze['candidate_sha256'],
                'plan_sha256':self.freeze['plan_sha256'],
                'model_provider_calls_during_evaluation':0,
                'model_provider_calls_during_prepare':self.freeze['model_provider_calls'],
                'cases_configured':len(self.cases),'cases':[], 'skipped':[],
                'stop_reason':None}
        if not execute:
            out.mkdir(parents=True,exist_ok=True)
            report['plan_preview']=[c.public() for c in self.cases]
            report['stop_reason']='dry_run_only_no_ERP_requests'
            self._write(out,report,None)
            return report
        self.ledger=EvidenceLog(out,enabled=True)
        try:
            self.ledger.append({'event':'run_started','candidate_sha256':self.freeze['candidate_sha256'],
                                'plan_sha256':self.freeze['plan_sha256'],
                                'budget':self.budget,'same_role_required':True})
            selectors=self._selectors(report)
            for c in self.cases:
                if self.reads>=self.budget:
                    report['skipped'].append({'id':c.id,'reason':'read_budget_exhausted'})
                    continue
                if c.relation == 'line_product' and self.reads + 2 > self.budget:
                    report['skipped'].append({'id':c.id,'reason':'insufficient_budget_for_selector_and_target'})
                    continue
                path,reason=_materialize_path(c,selectors)
                expected_fk=None
                if c.relation=='line_product':
                    rec=_selector_record(selectors.get('orders',[]),c.identity_index)
                    if rec is None:
                        reason='no_suitable_order_record'
                    else:
                        # A fresh line-list GET is an unscored oracle selector,
                        # and its observation cannot update frozen predictions.
                        _,line_result=self._oracle_get('selector_order_lines_'+c.id,
                                           '/orders/{id}/lines',f"/orders/{rec['id']}/lines",{})
                        if isinstance(line_result,list) and line_result and isinstance(line_result[0],dict):
                            candidate_fk=line_result[0].get('fk_product')
                            if type(candidate_fk) in (str,int) and _ID.fullmatch(str(candidate_fk)):
                                expected_fk=str(candidate_fk)
                                path='/products/'+expected_fk
                                reason=None
                            else:
                                reason='line_has_no_valid_product_reference'
                        else:reason='no_product_linked_order_line'
                elif c.relation in ('order_customer','invoice_customer'):
                    rec=_selector_record(selectors.get(c.family,[]),c.identity_index)
                    if rec is not None and type(rec.get('socid')) in (str,int) and _ID.fullmatch(str(rec['socid'])):
                        expected_fk=str(rec['socid'])
                if reason or path is None:
                    report['skipped'].append({'id':c.id,'reason':reason or 'no_safe_path'})
                    continue
                arms={arm:preds.get(c.id,{}) for arm,preds in self.candidate['predictors'].items()}
                # One aggregate precommit contains BOTH arms; identical oracle query.
                status,payload=self._oracle_get(c.id,c.template,path,arms)
                relation=None
                if c.category=='relationship':
                    relation=bool(status==200 and isinstance(payload,dict)
                                  and expected_fk is not None
                                  and type(payload.get('id')) in (str,int)
                                  and str(payload['id'])==expected_fk)
                truth=_semantic_truth(c,status,payload,relation_exists=relation)
                scores={arm:_score_one(c,pred,truth) for arm,pred in arms.items()}
                # Persist only field type names, key names, count, and booleans.
                row={'id':c.id,'family':c.family,'category':c.category,
                     'evidence_split':c.evidence_split, 'method':'GET',
                     'path_template':c.template,
                     'observation':truth, 'predictions':arms,'scores':scores}
                report['cases'].append(row)
                payload=None
            report['stop_reason']='budget_exhausted' if self.reads>=self.budget else 'plan_complete'
            self.ledger.append({'event':'run_finished','reads':self.reads,
                                'scored':len(report['cases']),
                                'stop_reason':report['stop_reason']})
        except Exception as exc:
            # Preserve a locally verifiable partial report on transport or other
            # errors, without ever recording URLs, IDs, tokens, or ERP error bodies.
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
    def _write(out:Path,report:dict,chain_sha:str|None):
        report['cases_scored']=len(report['cases'])
        report['scores']={arm:_metrics(report['cases'],arm) for arm in
                          ('symbolic','model') if arm in (report['cases'][0]['scores'] if report['cases'] else
                            {'symbolic':{},**({'model':{}} if report['model_provider_calls_during_prepare'] else {})})}
        report['by_evidence_split'] = {split:{arm:_metrics([r for r in report['cases']
                                                          if r['evidence_split']==split],arm)
                                                for arm in report['scores']}
                                        for split in sorted(set(r['evidence_split'] for r in report['cases']))}
        # Predictions are not meaningful unless both evaluated content and
        # coverage are explicitly reported. There is no automatic live gate pass.
        report['conclusion']='NOT_BEHAVIORAL_FIDELITY_VERIFICATION'
        out.mkdir(parents=True,exist_ok=True)
        raw=(json.dumps(report,indent=2,sort_keys=True)+'\n').encode()
        (out/'holdout_report.json').write_bytes(raw)
        freeze={'format':FORMAT+'-eval-freeze', 'phase':report['phase'],
                'candidate_sha256':report['candidate_sha256'],
                'plan_sha256':report['plan_sha256'],
                'report_sha256':sha256(raw).hexdigest(),
                'evidence_chain_sha256':chain_sha,
                'query_holdout':True,'independent_tenant_holdout':False,
                'live_writes':0,'externally_timestamped':False,
                'predictions_precommitted':chain_sha is not None}
        (out/'EVAL_FREEZE.json').write_text(json.dumps(freeze,indent=2,sort_keys=True)+'\n')


def verify(frozen:str|Path, evaluation:str|Path, discovery:str|Path|None=None) -> dict:
    from .gate2d_adaptive import hash_json
    cand,plan,src_freeze=_read_frozen(Path(frozen))
    if discovery is not None and sha256(Path(discovery).read_bytes()).hexdigest() != src_freeze['source_discovery_sha256']:
        raise ValueError('Original discovery evidence hash does not match preparation')
    out=Path(evaluation)
    ev_freeze=json.loads((out/'EVAL_FREEZE.json').read_text(encoding='utf-8'))
    raw=(out/'holdout_report.json').read_bytes()
    report=json.loads(raw)
    if sha256(raw).hexdigest()!=ev_freeze.get('report_sha256'):
        raise ValueError('Evaluation report hash mismatch')
    if ev_freeze.get('candidate_sha256')!=src_freeze['candidate_sha256'] or ev_freeze.get('plan_sha256')!=src_freeze['plan_sha256']:
        raise ValueError('Preparation and evaluation freeze mismatch')
    if report.get('live_writes')!=0 or any(c.get('method')!='GET' for c in report['cases']):
        raise ValueError('Non-readonly result')
    if report['phase']=='dry_run':
        return {'status':'PASS_DRY_RUN_ARTIFACT_ONLY','oracle_requests_verified':0,
                'external_timestamp_verified':False}
    path=out/'events.jsonl'
    previous='0'*64
    index=0
    pending=None
    count=0
    start=finish=0
    for line in path.read_text(encoding='utf-8').splitlines():
        event=json.loads(line)
        if event.get('seq')!=count+1 or event.get('prev_sha256')!=previous:
            raise ValueError('Event chain numbering mismatch')
        sig=event['sha256']
        unsig=dict(event);del unsig['sha256']
        if sig!=hash_json(unsig):raise ValueError('Event hash mismatch')
        previous=sig;count+=1
        if event['event']=='run_started':start+=1
        if event['event']=='run_finished':finish+=1
        if event['event']=='prediction_committed':
            if pending is not None or event['index']!=index or event.get('method')!='GET':
                raise ValueError('Prediction precommit error')
            if event.get('probe_id') not in {c['id'] for c in report['cases']} | {
                  'selector_'+f for f in FAMILIES} | {'selector_order_lines_order_line_product_fk'}:
                raise ValueError('Unexpected probe in integrity log')
            if event['probe_id'] in {c['id'] for c in report['cases']}:
                frozen_pred={arm:preds.get(event['probe_id'],{}) for arm,preds in cand['predictors'].items()}
                if event.get('prediction')!=frozen_pred:
                    raise ValueError('Recorded prediction differs from frozen candidate')
            pending=event['probe_id']
        if event['event'] in ('observation_recorded','probe_transport_error'):
            if pending!=event.get('probe_id') or event['index']!=index:
                raise ValueError('Observed response lacked matching precommit')
            pending=None;index+=1
    if pending or previous!=ev_freeze.get('evidence_chain_sha256') or start!=1 or finish!=1:
        raise ValueError('Incomplete or altered run')
    if index!=report['operations_executed']:
        raise ValueError('Oracle read count mismatch')
    return {'status':'PASS_LOCAL_INTEGRITY_AND_ORDER_ONLY',
            'verified_precommitted_reads':index,
            'frozen_candidate_sha256':src_freeze['candidate_sha256'],
            'independent_tenant_holdout':False, 'external_timestamp_verified':False,
            'real_behavioral_fidelity_verified':False}


def main(argv:list[str]|None=None) -> int:
    p=argparse.ArgumentParser(description='Gate 2E: freeze predictions, then score unseen read-only queries')
    sub=p.add_subparsers(dest='command',required=True)
    pp=sub.add_parser('prepare',help='Offline only; freeze symbolic and optional OpenAI predictions')
    pp.add_argument('--discovery',required=True,help='Gate2D discovery_report.json')
    pp.add_argument('--out',default='build/gate2e_frozen')
    pp.add_argument('--execute-model',action='store_true',help='Explicit API-model opt-in, NOT Dolibarr')
    pp.add_argument('--model',default='gpt-6')
    pe=sub.add_parser('evaluate',help='Dry run unless --execute-readonly is set')
    pe.add_argument('--frozen',required=True)
    pe.add_argument('--out',default='build/gate2e_preview')
    pe.add_argument('--budget',type=int,default=32)
    pe.add_argument('--base-url',default='http://127.0.0.1:8088')
    pe.add_argument('--execute-readonly',action='store_true')
    pv=sub.add_parser('verify',help='Verify local hashes and precommit order')
    pv.add_argument('--frozen',required=True);pv.add_argument('--evaluation',required=True)
    pv.add_argument('--discovery',help='Optional original discovery evidence to check its hash')
    args=p.parse_args(argv)
    try:
        if args.command=='prepare':
            advisor=OpenAIBlindPredictor(args.model) if args.execute_model else None
            result=prepare(args.discovery,args.out,model_advisor=advisor)
        elif args.command=='evaluate':
            u=urlparse(args.base_url)
            if u.scheme not in ('http','https') or not _local_host(u.hostname or '') or u.username or u.password:
                p.error('Only local HTTP(S) loopback ERP hosts are allowed')
            if u.path.rstrip('/') not in ('','/api/index.php') or u.query or u.fragment:
                p.error('Only base URL at root or /api/index.php, no query or credentials')
            key=os.environ.get('DOLIBARR_API_KEY') if args.execute_readonly else None
            if args.execute_readonly and not key:p.error('DOLIBARR_API_KEY required')
            evaluator=Evaluator(DolibarrClient(args.base_url,key),args.frozen,budget=args.budget)
            result=evaluator.run(args.out,execute=args.execute_readonly)
            result={'phase':result['phase'],'calls':result['operations_executed'],
                    'stop_reason':result['stop_reason'],
                    'cases_scored':result['cases_scored'],'scores':result['scores'],
                    'skipped':len(result['skipped']),'report':str(Path(args.out)/'holdout_report.json')}
        else:
            result=verify(args.frozen,args.evaluation,args.discovery)
    except (ValueError,FileExistsError,RuntimeError,KeyError) as exc:
        # Do not print exception values originating from actual ERP responses.
        p.error(type(exc).__name__+': '+str(exc)[:220] if args.command!='evaluate' else type(exc).__name__)
    print(json.dumps(result,indent=2,sort_keys=True))
    if args.command == 'evaluate' and result.get('stop_reason','').startswith('aborted:'):
        return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
