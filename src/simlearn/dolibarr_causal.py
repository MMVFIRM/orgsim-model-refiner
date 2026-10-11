"""Paired, receipt-classified, GET-only Dolibarr sim-to-real experiment.

All mutations are performed independently by the user in a disposable tenant.
This module has no HTTP-write primitive. Classification of manual responses is
operator attestation, not independent server-side certification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from . import dolibarr_transfer as legacy

FORMAT = 'simlearn-dolibarr-same-status-v3'
RECEIPT_CLASS = ('state_precondition', 'missing_business_required_field',
                 'business_amount_limit', 'duplicate_state_transition',
                 'other_documented_business_rejection','product_not_on_sale')
SAFE_BUSINESS_STATUS = {400, 409, 422}
CONTRAST = "watch_1.status"
EXPECTED_KIND = "order_validate"
SAFE_CODE = re.compile(r'^[A-Za-z][A-Za-z0-9_.-]{2,79}$')
PAIR_KEY = re.compile(r'^[a-z][a-z0-9_-]{4,100}$')
MAX_PAIRS = 150


def _read(path: Path) -> dict:
    return legacy._read(Path(path))


def _save(path: Path, obj: Any) -> None:
    legacy._write_once(Path(path), obj)


def _journal(root: Path, event: str, case: str | None, relative: str, doc: dict) -> None:
    legacy._append_event(root, event, case, relative, doc)


def triage(*, receipt_status:int, witnessed_state_change:bool,
           documented_business_code:bool=False, authorized_writer:bool=True,
           diagnostic_correlated:bool=False)->dict:
    if not 100<=receipt_status<=599:raise ValueError('Provide a real HTTP status')
    if not authorized_writer:
        label='ACCESS_OR_ROLE_UNKNOWN';usable=False
    elif 200<=receipt_status<300 and witnessed_state_change:
        label='SUCCESS_CANDIDATE';usable=True
    elif receipt_status in SAFE_BUSINESS_STATUS and not witnessed_state_change and documented_business_code:
        label='BUSINESS_REJECT_CANDIDATE';usable=True
    elif receipt_status==500 and not witnessed_state_change and documented_business_code and diagnostic_correlated:
        label='POSSIBLE_APPLICATION_VALIDATION_REFUSAL';usable=True
    elif receipt_status==304:
        label='NO_OP_NOT_BUSINESS_REFUSAL';usable=False
    elif receipt_status in (401,403):
        label='ACCESS_DENIED';usable=False
    elif receipt_status==404:
        label='MISSING_RECORD_OR_ROUTE';usable=False
    elif receipt_status>=500:
        label='UNCLASSIFIED_SERVER_ERROR';usable=False
    else:
        label='UNKNOWN';usable=False
    return {'format':FORMAT+'-triage','reported_status':receipt_status,
            'outcome_category':label,'candidate_for_paired_study':usable,
            'warning':'A 500 must contain the application validation error and correlate with local diagnostic evidence. Manual receipts are not independent attestation.',
            'network_calls':0,'automatic_writes':0}

def _public_confidence(k: int, n: int) -> dict:
    """Wilson interval, descriptive only; cases are not independent tenants."""
    if n == 0:
        return {'low': None, 'high': None}
    from math import sqrt
    z = 1.959963984540054
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z*sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return {'low': round(max(0, center-half), 6), 'high': round(min(1,center+half), 6)}


def _assert_protocol(p:dict)->dict:
    if not isinstance(p,dict) or set(p) != {'format','base_plan','pairs','quality_policy','pilot_sha256'} or p['format']!=FORMAT+'-plan':
        raise ValueError('Wrong v0.5 same-status plan format')
    base=p['base_plan'];legacy.validate_plan(base)
    if not isinstance(p['pilot_sha256'],str) or not re.fullmatch(r'[0-9a-f]{64}',p['pilot_sha256']):
        raise ValueError('Qualified pilot digest required')
    rules=p['quality_policy']
    if set(rules)!={'train_pairs_min','validation_pairs_min','holdout_pairs_min'} or any(type(x) is not int or x<1 or x>MAX_PAIRS for x in rules.values()):
        raise ValueError('Invalid pair minima')
    pairs=p['pairs']
    if not isinstance(pairs,list) or not pairs or len(pairs)>MAX_PAIRS:
        raise ValueError('Invalid number of pairs')
    cases={c['id']:c for c in base['cases']}
    assigned=[];names=set()
    for pair in pairs:
        if (set(pair)!={'id','operation','split','members','contrast_keys','match_keys'} or
                not isinstance(pair['id'],str) or not PAIR_KEY.fullmatch(pair['id']) or pair['id'] in names):
            raise ValueError('Invalid pair identifier/keys')
        names.add(pair['id'])
        members=pair['members'];assigned.extend(members)
        if pair['operation']!=EXPECTED_KIND or pair['split'] not in legacy.SPLITS:
            raise ValueError('Only order_validate is in the validated v0.5 study')
        if (not isinstance(members,list) or len(members)!=2 or len(set(members))!=2 or
                any(cid not in cases or cases[cid]['operation']!=EXPECTED_KIND or cases[cid]['split']!=pair['split'] for cid in members)):
            raise ValueError('Pair needs exactly two same-operation, same-split records')
        if pair['contrast_keys']!=[CONTRAST] or pair['match_keys']!=['main.status','main.total_ht','watch_1.type']:
            raise ValueError('Study controls and target contrast are not editable')
        for cid in members:
            if len(cases[cid]['watch'])!=1 or cases[cid]['watch'][0]['family']!='products':
                raise ValueError('Each order must watch the product actually used in its single line')
    if set(assigned)!=set(cases) or len(assigned)!=len(set(assigned)):
        raise ValueError('Every case must belong to exactly one pair')
    for split,minimum in [('train',rules['train_pairs_min']),('validation',rules['validation_pairs_min']),('holdout',rules['holdout_pairs_min'])]:
        if sum(p['split']==split for p in pairs)<minimum:
            raise ValueError('Insufficient '+split+' pair slots')
    return p

def _allowed_feature_keys(operation:str)->set[str]:
    fields={'main.'+k for k in legacy.SAFE_FIELDS[legacy.OPS[operation]]}
    fields.add('main.lines_count')
    if operation=='invoice_payment':fields.add('action.payment_amount')
    return fields


def scaffold(output:Path, *, pilot_file:Path,
             train_pairs:int=5,validation_pairs:int=3,holdout_pairs:int=10,
             base_url:str='http://127.0.0.1:8088')->dict:
    if not 4<=train_pairs<=30 or not 2<=validation_pairs<=20 or not 8<=holdout_pairs<=30:
        raise ValueError('Need >=4 train, >=2 validation, >=8 holdout paired fixtures')
    proof=_read(pilot_file)
    if (proof.get('format')!=FORMAT+'-pilot-confirm' or not proof.get('qualified') or
            proof.get('base_url')!=base_url or not proof.get('same_status_verified') or
            proof.get('business_error_evidence')!='CORRELATED_APP_VALIDATION_RECEIPT'):
        raise ValueError('Use a successful, matching v0.5 same-tenant pilot-confirm first')
    if Path(output).exists() or Path(output).with_name(Path(output).stem+'_IDS.csv').exists():
        raise FileExistsError('Refuse to overwrite scaffold/CSV')
    cases=[];pairs=[]
    for split,n in [('train',train_pairs),('validation',validation_pairs),('holdout',holdout_pairs)]:
        for i in range(1,n+1):
            pair_id=f'{split}-order_validate-{i:03d}'
            members=[f'{pair_id}-a',f'{pair_id}-b']
            for cid in members:
                cases.append({'id':cid,'operation':'order_validate','split':split,'record_id':'FILL_ID',
                              'watch':[{'family':'products','record_id':'FILL_PRODUCT_ID'}],'action':{}})
            pairs.append({'id':pair_id,'operation':'order_validate','split':split,'members':members,
                          'contrast_keys':[CONTRAST],
                          'match_keys':['main.status','main.total_ht','watch_1.type']})
    base={'format':legacy.FORMAT+'-plan','tenant_label':'sandbox_local_v05',
          'snapshot_label':'before_causal_v05','base_url':base_url,'cases':cases}
    p={'format':FORMAT+'-plan','base_plan':base,'pairs':pairs,
       'quality_policy':{'train_pairs_min':4,'validation_pairs_min':2,'holdout_pairs_min':8},
       'pilot_sha256':legacy.digest(proof)}
    _save(output,p)
    csv_path=Path(output).with_name(Path(output).stem+'_IDS.csv')
    with csv_path.open('x',encoding='utf-8',newline='') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(['case_id','pair_id','operation','split','expected_product_sale_status','record_id','product_id'])
        lookup={cid:q['id'] for q in pairs for cid in q['members']}
        for c in cases:
            number=int(lookup[c['id']].rsplit('-',1)[1])
            active_side='a' if number%2 else 'b'
            target='1' if c['id'].endswith('-'+active_side) else '0'
            writer.writerow([c['id'],lookup[c['id']],c['operation'],c['split'],target,'',''])
    return {'format':FORMAT+'-scaffold','output':str(output),'fixture_csv':str(csv_path),
            'pairs':len(pairs),'disposable_orders':len(cases),'pilot_sha256':p['pilot_sha256'],
            'network_calls':0,'automated_writes':0}

def bind(template:Path, fixture_csv:Path, output:Path)->dict:
    plan=_read(template)
    if plan.get('format')!=FORMAT+'-plan':raise ValueError('Wrong plan template')
    if Path(output).exists():raise FileExistsError('Do not overwrite a bound plan')
    cases={c['id']:c for c in plan['base_plan']['cases']}
    members={cid:q['id'] for q in plan['pairs'] for cid in q['members']}
    seen=set();orders=set()
    with Path(fixture_csv).open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['case_id','pair_id','operation','split','expected_product_sale_status','record_id','product_id']:
            raise ValueError('Use the exact generated fixture CSV columns')
        for row in reader:
            cid=row['case_id'];c=cases.get(cid)
            if c is None or cid in seen:raise ValueError('Unknown/duplicate case')
            seen.add(cid)
            if (row['pair_id']!=members[cid] or row['operation']!=c['operation'] or row['split']!=c['split']):
                raise ValueError('Do not edit immutable study assignments')
            number=int(members[cid].rsplit('-',1)[1])
            active_side='a' if number%2 else 'b'
            target='1' if cid.endswith('-'+active_side) else '0'
            if row['expected_product_sale_status']!=target:
                raise ValueError('Intended on-sale assignment must stay unchanged in fixture CSV')
            order=row['record_id'].strip();product=row['product_id'].strip()
            if not legacy.ID.fullmatch(order) or int(order)==0 or not legacy.ID.fullmatch(product) or int(product)==0:
                raise ValueError('Each row needs a positive order and product ID')
            if order in orders:raise ValueError('Distinct disposable order ID required for each action')
            orders.add(order);c['record_id']=order;c['watch'][0]['record_id']=product
    if seen!=set(cases):raise ValueError('Fixture CSV is incomplete')
    _assert_protocol(plan);_save(output,plan)
    return {'format':FORMAT+'-bound','output':str(output),'orders':len(orders),'writes':0}

def _pair_workspace(root:Path)->tuple[dict,dict,dict]:
    root=Path(root)
    base,meta=legacy._workspace(root)
    pp=_read(root/'private'/'PAIRED_PLAN.json')
    _assert_protocol(pp)
    if legacy.digest(pp['base_plan'])!=meta['plan_sha256'] or pp['base_plan']!=base:
        raise ValueError('Paired plan/legacy tenant disagreement')
    return pp,base,meta


def init(plan_path:Path,root:Path)->dict:
    pp=_assert_protocol(_read(plan_path))
    plan_path=Path(plan_path);root=Path(root)
    if root.exists():raise FileExistsError('Refusing to overwrite a paired run')
    # Existing journal initialization is reused; staging contains only the plan,
    # lives next to the user's source and is removed immediately afterwards.
    fd,tmp=tempfile.mkstemp(prefix='.orgsim_paired_',suffix='.json',dir=str(plan_path.resolve().parent))
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as f:
            f.write(legacy.canonical(pp['base_plan']))
        r=legacy.init(Path(tmp),root)
    finally:
        try:Path(tmp).unlink()
        except FileNotFoundError:pass
    _save(root/'private'/'PAIRED_PLAN.json',pp)
    _journal(root,'paired_init',None,'private/PAIRED_PLAN.json',pp)
    return {**r,'format':FORMAT+'-initialized','pairs':len(pp['pairs']),
            'read_requests':0,'write_requests':0}


def _find_pair(pp:dict,pair_id:str)->dict:
    for pair in pp['pairs']:
        if pair['id']==pair_id:return pair
    raise ValueError('Unknown pair ID')


def _pair_for_case(pp:dict,case_id:str)->dict:
    for pair in pp['pairs']:
        if case_id in pair['members']:return pair
    raise ValueError('Unknown paired case ID')


def _equal(x:Any,y:Any)->bool:
    if isinstance(x,(str,int,float)) and isinstance(y,(str,int,float)):
        xx=legacy._decimal(x);yy=legacy._decimal(y)
        if xx is not None and yy is not None:return xx==yy
    return x==y


def _pair_difference(pair:dict,left:dict,right:dict,left_action:dict,right_action:dict)->dict:
    if (left['main'].get('http_status')!=200 or right['main'].get('http_status')!=200 or
            left['main'].get('kind')!='object' or right['main'].get('kind')!='object'):
        raise ValueError('Both order snapshots must be readable HTTP 200 objects')
    if (left.get('watch_1',{}).get('http_status')!=200 or right.get('watch_1',{}).get('http_status')!=200):
        raise ValueError('Both watched products must be readable 200 objects')
    lf=legacy._features(left,left_action);rf=legacy._features(right,right_action)
    if 'main.status' not in lf or 'main.status' not in rf:
        raise ValueError('Missing required order lifecycle status; cannot enforce same status')
    if not _equal(lf['main.status'],rf['main.status']) or str(lf['main.status']).lower() not in ('0','draft'):
        raise ValueError('Both order records must be DRAFT and have the SAME lifecycle status')
    for k in ('main.statut','main.fk_statut'):
        if k in lf and k in rf and not _equal(lf[k],rf[k]):
            raise ValueError('Both orders must have the same secondary status '+k)
    diff=[k for k in sorted(set(lf)|set(rf)) if k not in lf or k not in rf or not _equal(lf[k],rf[k])]
    # Dolibarr versions may expose an exact alias for product on-sale status.
    # Permit it to vary only if both sides agree semantically with status.
    aliases=set()
    for alias in ('watch_1.tosell','watch_1.statut'):
        if alias in diff and alias in lf and alias in rf and CONTRAST in lf and CONTRAST in rf:
            active=lambda v:str(v).lower() in ('1','true','yes')
            if active(lf[alias])==active(lf[CONTRAST]) and active(rf[alias])==active(rf[CONTRAST]):
                aliases.add(alias)
    unexpected=sorted(set(diff)-set(pair['contrast_keys'])-aliases)
    if unexpected:raise ValueError(f'Same-status pair not matched; unexpected features differ: {unexpected[:8]}')
    for k in pair['contrast_keys']:
        if k not in lf or k not in rf or _equal(lf[k],rf[k]):
            raise ValueError('The watched-product status contrast was not actually observed')
    for k in pair['match_keys']:
        if k not in lf or k not in rf or not _equal(lf[k],rf[k]):
            raise ValueError('Mandatory matched condition absent/different: '+k)
    if ('main.lines_count' in lf or 'main.lines_count' in rf) and (str(lf.get('main.lines_count'))!='1' or str(rf.get('main.lines_count'))!='1'):
        raise ValueError('Both draft orders must contain exactly one line each')
    if {str(lf[CONTRAST]),str(rf[CONTRAST])}!={'0','1'}:
        raise ValueError('The watched products must contrast on-sale (1) vs not-on-sale (0)')
    if pair.get('id'):
        expected_first='1' if int(pair['id'].rsplit('-',1)[1])%2 else '0'
        if str(lf[CONTRAST])!=expected_first:
            raise ValueError('The watched-product status does not match randomized CSV assignment')
    return {'verified':True,'same_draft_status':True,'contrast_observed':[CONTRAST],
            'unexpected_differences':0,'shared_features':len(set(lf)|set(rf)),
            'status_only_baseline_not_discriminating':True,
            'caution':'Snapshot-only comparison; separately verify actual order line links to watched product.'}

def _verify_line_binding(client,case:dict)->bool:
    """Read raw order lines IN MEMORY; persist only booleans, never product or customer IDs."""
    if case['operation']!='order_validate' or len(case['watch'])!=1 or case['watch'][0]['family']!='products':
        raise ValueError('An order with exactly one watched product is required')
    status,raw=client.request('GET',f"/orders/{case['record_id']}?withLines=true")
    if status!=200 or not isinstance(raw,dict):
        raise ValueError('Cannot independently inspect order lines')
    lines=raw.get('lines')
    if not isinstance(lines,list) or len(lines)!=1 or not isinstance(lines[0],dict):
        raise ValueError('Order must have exactly one actual line to isolate a product condition')
    product=lines[0].get('fk_product')
    if product is None or str(product)!=case['watch'][0]['record_id']:
        raise ValueError('Watched product does not match the real line product; fix the CSV')
    return True


def start_pair(root:Path,pair_id:str,*,client=None,execute_model=False,
               share_financials=False,model_name='gpt-6-luna',model_backend=None)->dict:
    root=Path(root);pp,base,meta=_pair_workspace(root)
    pair=_find_pair(pp,pair_id)
    if pair['split']=='holdout':
        if not (root/'private'/'FROZEN_POLICY.json').exists():
            raise ValueError('Do not open holdouts before a qualified frozen policy')
    elif (root/'private'/'FROZEN_POLICY.json').exists():
        raise ValueError('Training/validation must not be opened after freeze')
    for cid in pair['members']:
        if (root/'private'/'cases'/(cid+'.start.json')).exists():
            raise FileExistsError('This matched pair has already been started')
    case_a=legacy._case_lookup(base,pair['members'][0]);case_b=legacy._case_lookup(base,pair['members'][1])
    # No file is committed until both pre-intervention snapshots pass comparator.
    if client is None:client=legacy._default_client(base)
    before_a=legacy._capture(client,case_a);before_b=legacy._capture(client,case_b)
    diag=_pair_difference(pair,before_a,before_b,case_a['action'],case_b['action'])
    binding_a=_verify_line_binding(client,case_a)
    binding_b=_verify_line_binding(client,case_b)
    # Commit both predictions before returning control to the operator.
    result_a=legacy.start(root,case_a['id'],client=client,execute_model=execute_model,
                          share_financials=share_financials,model_name=model_name,model_backend=model_backend)
    result_b=legacy.start(root,case_b['id'],client=client,execute_model=execute_model,
                          share_financials=share_financials,model_name=model_name,model_backend=model_backend)
    persisted_a=_read(root/'private'/'cases'/(case_a['id']+'.start.json'))
    persisted_b=_read(root/'private'/'cases'/(case_b['id']+'.start.json'))
    actual=_pair_difference(pair,persisted_a['before'],persisted_b['before'],case_a['action'],case_b['action'])
    record={'format':FORMAT+'-pair-start','pair_id':pair_id,'split':pair['split'],
            'preflight_snapshot_hashes':[legacy.digest(before_a),legacy.digest(before_b)],
            'committed_start_hashes':[legacy.digest(persisted_a),legacy.digest(persisted_b)],
            'contrast_observed':actual['contrast_observed'],'shared_features':actual['shared_features'],
            'line_binding_verified':[binding_a,binding_b],
            'commit_precedes_manual_intervention':True,'operator_action_executed':False}
    rel='private/pairs/'+pair_id+'.start.json'
    _save(root/rel,record);_journal(root,'paired_start',pair_id,rel,record)
    return {'pair_id':pair_id,'split':pair['split'],'record_ids_local_only':[case_a['record_id'],case_b['record_id']],
            'cases':pair['members'],'quality':actual,'preflight_reads':6,'committed_reads':4,
            'model_calls':4*len(pair['members']) if execute_model else 0,
            'automated_writes':0,
            'instruction':'Both immutable before-states and holdout predictions are committed. Now perform the two actions independently; finish each case.'}


def _check_app_validation_500(receipt_file:Path,diagnostic_file:Path)->tuple[str,str]:
    """Both sources remain local; store only hashes and the classification flag."""
    for fp in (receipt_file,diagnostic_file):
        if not isinstance(fp,Path) or not fp.is_file() or fp.stat().st_size>2_000_000 or fp.stat().st_size==0:
            raise ValueError('Require two nonempty local redacted evidence files (<=2MB each)')
    receipt=receipt_file.read_text('utf-8',errors='replace')
    diagnostic=diagnostic_file.read_text('utf-8',errors='replace')
    if not re.search(r'Error when validating Order\s*:',receipt,re.I):
        raise ValueError('HTTP 500 receipt is missing Dolibarr application validation-error prefix')
    if not (re.search(r'checkActiveProductInLines',diagnostic,re.I) or
            re.search(r'product[^\n]{0,100}(?:not.{0,30}(?:sell|sale|active)|inactive|disabled)',diagnostic,re.I) or
            re.search(r'(?:not.{0,30}(?:sell|sale|active)|inactive|disabled)[^\n]{0,100}product',diagnostic,re.I)):
        raise ValueError('Local diagnostic must specifically corroborate product on-sale validation, not just any HTTP 500')
    return _file_hash(receipt_file),_file_hash(diagnostic_file)


def _pilot_case(name:str,order_id:str,product_id:str)->dict:
    if not legacy.ID.fullmatch(str(order_id)) or not legacy.ID.fullmatch(str(product_id)) or int(order_id)==0 or int(product_id)==0:
        raise ValueError('Pilot requires positive numeric order/product IDs')
    return {'id':name,'operation':EXPECTED_KIND,'split':'train','record_id':str(order_id),
            'watch':[{'family':'products','record_id':str(product_id)}],'action':{}}


def pilot_start(root:Path,*,order_a:str,order_b:str,product_a:str,product_b:str,
                base_url:str='http://127.0.0.1:8088',client=None)->dict:
    """Minimal two-order pilot; network is GET-only and mutations remain manual."""
    root=Path(root)
    if root.exists():raise FileExistsError('Pilot directory already exists')
    if order_a==order_b or product_a==product_b:
        raise ValueError('Pilot pairs need distinct orders and distinct products')
    cases=[_pilot_case('pilot-a',order_a,product_a),_pilot_case('pilot-b',order_b,product_b)]
    base={'format':legacy.FORMAT+'-plan','tenant_label':'pilot_same_status_v05',
          'snapshot_label':'pilot_snapshot_v05','base_url':base_url,'cases':[
              {**c,'split':'train'} for c in cases]}
    # Validates host before any request; no secrets or arbitrary URLs accepted.
    if not re.fullmatch(r'http://127\.0\.0\.1:[0-9]{2,5}',base_url):
        raise ValueError('Pilot must use explicit loopback http://127.0.0.1:PORT')
    port=int(base_url.rsplit(':',1)[1])
    if not 1024<=port<=65535:raise ValueError('Unsafe pilot port')
    if client is None:client=legacy._default_client(base)
    a=legacy._capture(client,cases[0]);b=legacy._capture(client,cases[1])
    pair={'contrast_keys':[CONTRAST],
          'match_keys':['main.status','main.total_ht','watch_1.type']}
    comparison=_pair_difference(pair,a,b,{},{});
    links=[_verify_line_binding(client,c) for c in cases]
    sides=[str(legacy._features(x,{})[CONTRAST]) for x in (a,b)]
    active='pilot-a' if sides[0]=='1' else 'pilot-b'
    inactive='pilot-b' if active=='pilot-a' else 'pilot-a'
    root.mkdir(parents=True)
    private={'format':FORMAT+'-pilot-start','base_url':base_url,'cases':cases,
             'before':{'pilot-a':a,'pilot-b':b},'line_bindings':links,
             'same_status_verified':True,'contrast':comparison['contrast_observed'],
             'active_case':active,'inactive_case':inactive,'automatic_writes':0}
    _save(root/'private'/'PILOT_START.json',private)
    return {'format':FORMAT+'-pilot-start-receipt','active_case':active,
            'inactive_case':inactive,'same_status_verified':True,'line_bindings_verified':True,
            'private_plan_hash':legacy.digest(private),
            'instruction':'Now manually validate BOTH disposable draft orders, one using the sale-active product and one using the sale-disabled product. Save the redacted HTTP 500 receipt and relevant local diagnostic if a business failure occurs.',
            'GET_calls':6,'automatic_writes':0}


def pilot_finish(root:Path,*,active_status:int,inactive_status:int,
                 receipt_file:Path,diagnostic_file:Path,client=None)->dict:
    root=Path(root)
    source=_read(root/'private'/'PILOT_START.json')
    if (root/'public'/'PILOT_CONFIRM.json').exists():
        raise FileExistsError('Pilot confirmation immutable')
    if active_status not in (200,201) or inactive_status!=500:
        raise ValueError('Pilot requires an observed 2xx success and an attributable application-level 500 refusal')
    h1,h2=_check_app_validation_500(Path(receipt_file),Path(diagnostic_file))
    plan={'base_url':source['base_url']}
    if client is None:client=legacy._default_client(plan)
    cases=source['cases']
    after={c['id']:legacy._capture(client,c) for c in cases}
    changes={c['id']:legacy._change(source['before'][c['id']],after[c['id']]) for c in cases}
    active=source['active_case'];inactive=source['inactive_case']
    if not changes[active] or changes[inactive]:
        raise ValueError('Before/after GETs do not corroborate active success AND inactive refusal')
    before=source['before']
    preactive=legacy._features(before[active],{})
    if preactive.get(CONTRAST) not in ('1',1):raise ValueError('Incorrect active side')
    qualified={'format':FORMAT+'-pilot-confirm','base_url':source['base_url'],
               'qualified':True,'same_status_verified':True,'business_error_evidence':'CORRELATED_APP_VALIDATION_RECEIPT',
               'initial_snapshots_sha256':legacy.digest(source['before']),
               'receipt_sha256':h1,'diagnostic_sha256':h2,
               'active_changed_fields':len(changes[active]),'inactive_changed_fields':0,
               'manual_outcomes_not_independently_authenticated':True,'automatic_writes':0}
    _save(root/'public'/'PILOT_CONFIRM.json',qualified)
    _save(root/'private'/'PILOT_FINISH.json',{
        'format':FORMAT+'-pilot-private-after','after':after,'changed_fields':changes,
        'public_confirm_sha256':legacy.digest(qualified)})
    return {'format':FORMAT+'-pilot-result','qualified':True,
            'pilot_proof':str(root/'public'/'PILOT_CONFIRM.json'),
            'manual_outcomes_not_independently_authenticated':True,
            'automatic_writes':0}


def _file_hash(path:Path|None)->str|None:
    if path is None:return None
    p=Path(path)
    if not p.is_file() or p.stat().st_size>2_000_000 or p.stat().st_size==0:
        raise ValueError('Receipt evidence file must be a non-empty local file <=2MB')
    return hashlib.sha256(p.read_bytes()).hexdigest()


def finish_case(root:Path,case_id:str,*,outcome:str,source:str,
                receipt_status:int|None=None,operator_role:str='unknown',
                business_reason:str|None=None,business_code:str|None=None,
                receipt_file:Path|None=None,diagnostic_file:Path|None=None,client=None)->dict:
    root=Path(root);pp,base,_=_pair_workspace(root)
    pair=_pair_for_case(pp,case_id)
    if not (root/'private'/'pairs'/(pair['id']+'.start.json')).exists():
        raise ValueError('Use start-pair before manual interventions')
    if not (root/'private'/'cases'/(case_id+'.start.json')).exists():
        raise ValueError('Missing before-state')
    if (root/'private'/'cases'/(case_id+'.finish.json')).exists():
        raise FileExistsError('Case already finished')
    if source not in ('UI','MANUAL_API') or outcome not in legacy.KINDS:
        raise ValueError('Invalid outcome/source')
    if operator_role not in ('authorized_writer','read_only','unknown'):
        raise ValueError('Invalid intervention role')
    if source=='MANUAL_API' and (receipt_status is None or not 100<=receipt_status<=599):
        raise ValueError('Manual API attempt requires the actual HTTP status')
    if operator_role!='authorized_writer' and outcome in ('SUCCESS','BUSINESS_REJECTED','NO_OP'):
        raise ValueError('Non-writer identity cannot attest business results')
    if source=='MANUAL_API':
        if receipt_status in (401,403) and outcome!='ACCESS_DENIED':
            raise ValueError('401/403 are access denial, never a business-rule label')
        if receipt_status==304 and outcome!='NO_OP':
            raise ValueError('304 must be NO_OP; cannot count as a business rejection')
        if receipt_status==404 and outcome not in ('UNKNOWN','OTHER_ERROR'):
            raise ValueError('404 indicates missing route or fixture')
        if receipt_status>=500 and not (receipt_status==500 and outcome=='BUSINESS_REJECTED') and outcome!='OTHER_ERROR':
            raise ValueError('Unhandled 5xx is OTHER_ERROR, not a business rule')
        if 200<=receipt_status<300 and outcome not in ('SUCCESS','UNKNOWN'):
            raise ValueError('2xx can only indicate success or unknown')
    hreceipt=_file_hash(receipt_file)
    hdiagnostic=None;checked500=False
    if outcome=='BUSINESS_REJECTED':
        if source!='MANUAL_API' or receipt_status not in SAFE_BUSINESS_STATUS|{500}:
            raise ValueError('Business refusals require a writer API response and documented error')
        if business_reason not in RECEIPT_CLASS:
            raise ValueError('Documented business reason is mandatory')
        if receipt_status in SAFE_BUSINESS_STATUS and (not isinstance(business_code,str) or not SAFE_CODE.fullmatch(business_code)):
            raise ValueError('4xx business receipt requires a nonsecret actual server error code')
        if business_code is not None and (not isinstance(business_code,str) or not SAFE_CODE.fullmatch(business_code)):
            raise ValueError('Nonsecret business code must be a safe literal token')
        if hreceipt is None:raise ValueError('Redacted actual server receipt file required')
        if receipt_status==500:
            if business_reason!='product_not_on_sale' or diagnostic_file is None:
                raise ValueError('500 classification requires product_not_on_sale and correlated diagnostic file')
            hreceipt,hdiagnostic=_check_app_validation_500(Path(receipt_file),Path(diagnostic_file));checked500=True
    elif business_reason is not None or business_code is not None or diagnostic_file is not None:
        raise ValueError('Business reason/code/diagnostic only applies to BUSINESS_REJECTED')
    if outcome=='NO_OP' and (source!='MANUAL_API' or receipt_status!=304 or hreceipt is None):
        raise ValueError('NO_OP needs 304 receipt; it does NOT count as business refusal')
    if outcome=='SUCCESS' and source=='MANUAL_API' and (receipt_status is None or not 200<=receipt_status<300):
        raise ValueError('Manual API success requires 2xx')
    # The legacy evidence recorder rejects all 5xx business labels. Once the
    # v0.5 wrapper has checked BOTH independent local evidence files, pass a
    # null status to the legacy recorder; keep the actual 500 in the separately
    # chained v0.5 receipt. This avoids changing inherited v0.4 behavior.
    delegated_status=None if checked500 else receipt_status
    result=legacy.finish(root,case_id,outcome=outcome,source=source,
                         receipt_status=delegated_status,client=client)
    receipt={'format':FORMAT+'-operator-receipt','case_id':case_id,'pair_id':pair['id'],
             'status':receipt_status,'source':source,'operator_role':operator_role,
             'outcome_attested':outcome,'business_reason':business_reason,
             'business_code':business_code,'receipt_sha256':hreceipt,
             'diagnostic_sha256':hdiagnostic,'app_validation_500_correlated':checked500,
             'manual_attestation_not_independent':True}
    rel='private/receipts/'+case_id+'.json'
    _save(root/rel,receipt);_journal(root,'paired_receipt',case_id,rel,receipt)
    return {**result,'label':_effective_label(root,case_id),
            'application_500_with_correlated_diagnostic':checked500,
            'automatic_writes':0}

def _effective_label(root:Path,case_id:str)->str:
    ff=root/'private'/'cases'/(case_id+'.finish.json')
    rp=root/'private'/'receipts'/(case_id+'.json')
    if not ff.exists() or not rp.exists():return 'MISSING'
    finish=_read(ff);receipt=_read(rp)
    if receipt['operator_role']!='authorized_writer':return 'UNKNOWN'
    if finish['effective_truth']=='ACCEPT' and receipt['outcome_attested']=='SUCCESS':return 'ACCEPT'
    if finish['effective_truth']!='REJECT' or receipt['outcome_attested']!='BUSINESS_REJECTED':return 'UNKNOWN'
    if (receipt['source']!='MANUAL_API' or not receipt['receipt_sha256'] or
            receipt['business_reason'] not in RECEIPT_CLASS):
        return 'UNKNOWN'
    if receipt['status'] in SAFE_BUSINESS_STATUS and receipt['business_code']:return 'REJECT'
    if (receipt['status']==500 and receipt.get('app_validation_500_correlated') and
            receipt.get('diagnostic_sha256') and receipt['business_reason']=='product_not_on_sale'):
        return 'REJECT'
    return 'UNKNOWN'

def _qualified_pairs(root:Path,pp:dict,split:str)->tuple[list[dict],list[dict]]:
    good=[];bad=[]
    for pair in pp['pairs']:
        if pair['split']!=split:continue
        labels=[_effective_label(root,cid) for cid in pair['members']]
        rec={'pair_id':pair['id'],'operation':pair['operation'],'split':split,
             'cases':pair['members'],'labels':labels}
        if set(labels)=={'ACCEPT','REJECT'}:
            good.append(rec)
        else:
            rec['reason']='missing' if 'MISSING' in labels else 'unknown_or_no_op' if 'UNKNOWN' in labels else 'same_class'
            bad.append(rec)
    return good,bad

def fit(root:Path)->dict:
    root=Path(root);pp,base,_=_pair_workspace(root)
    if (root/'private'/'FROZEN_POLICY.json').exists():
        raise FileExistsError('Frozen policy already exists')
    if any((root/'private'/'cases'/(c['id']+'.start.json')).exists() for c in base['cases'] if c['split']=='holdout'):
        raise ValueError('Holdout opened before a qualified fit; abandon this workspace')
    incomplete=[pair['id'] for pair in pp['pairs'] if pair['split'] in ('train','validation')
                and not all((root/'private'/'cases'/(cid+'.finish.json')).exists() and
                            (root/'private'/'receipts'/(cid+'.json')).exists()
                            for cid in pair['members'])]
    if incomplete:
        raise ValueError(f'Finish all train/validation pairs before freezing; {len(incomplete)} unfinished')
    qualified={};unqualified={}
    for split in ('train','validation'):
        qualified[split],unqualified[split]=_qualified_pairs(root,pp,split)
    for op in sorted({p['operation'] for p in pp['pairs']}):
        for split,label in [('train','train_pairs_min'),('validation','validation_pairs_min')]:
            count=sum(q['operation']==op for q in qualified[split])
            if count<pp['quality_policy'][label]:
                reasons={}
                for x in unqualified[split]:
                    if x['operation']==op:reasons[x['reason']]=reasons.get(x['reason'],0)+1
                raise ValueError(f'{op} {split}: {count} qualifying ACCEPT/REJECT pairs; '
                                 f'require {pp["quality_policy"][label]}. Blocked before holdout. '
                                 f'Incomplete/non-discriminating pairs: {reasons}')
    good_train={c for pair in qualified['train'] for c in pair['cases']}
    good_validation={c for pair in qualified['validation'] for c in pair['cases']}
    r=legacy.fit(root,min_train=8,train_ids=good_train,validation_ids=good_validation,
                 min_train_per_class=2*pp['quality_policy']['train_pairs_min']//2,
                 min_validation_per_class=pp['quality_policy']['validation_pairs_min'])
    frozen=_read(root/'private'/'FROZEN_POLICY.json')
    if any(frozen['operations'][op].get('status')!='trained' for op in {p['operation'] for p in pp['pairs']}):
        raise RuntimeError('Internal fit unexpectedly failed after preflight')
    result={'format':FORMAT+'-freeze-summary','qualified_train_pairs':len(qualified['train']),
            'qualified_validation_pairs':len(qualified['validation']),
            'unqualified_train_pairs':len(unqualified['train']),
            'unqualified_validation_pairs':len(unqualified['validation']),
            'policy_sha256':legacy.digest(frozen),
            'all_required_operations_trained':True,
            'holdout_unopened_at_freeze':True}
    _save(root/'private'/'PAIR_FIT.json',result)
    _journal(root,'paired_fit',None,'private/PAIR_FIT.json',result)
    return {**r,**result,'read_requests':0,'provider_calls':0,'automated_writes':0}


def verify(root:Path)->dict:
    root=Path(root);pp,base,meta=_pair_workspace(root)
    result=legacy.verify(root)
    # Event authenticity and order checked by legacy.verify; now validate pair
    # atomicity, pair matching, receipt presence and freeze ordering as well.
    events=[json.loads(ln) for ln in (root/'private'/'EVENTS.jsonl').read_text('utf-8').splitlines() if ln.strip()]
    event_idx={(e['phase'],e['case']):i for i,e in enumerate(events)}
    if ('paired_init',None) not in event_idx:raise ValueError('Missing paired plan commitment')
    for pair in pp['pairs']:
        pair_path=root/'private'/'pairs'/(pair['id']+'.start.json')
        starts=[root/'private'/'cases'/(cid+'.start.json') for cid in pair['members']]
        finishes=[root/'private'/'cases'/(cid+'.finish.json') for cid in pair['members']]
        if pair_path.exists():
            if not all(f.exists() for f in starts):raise ValueError('Partial pair has a start commitment')
            data_a=_read(starts[0]);data_b=_read(starts[1]); pair_doc=_read(pair_path)
            _pair_difference(pair,data_a['before'],data_b['before'],
                             legacy._case_lookup(base,pair['members'][0])['action'],
                             legacy._case_lookup(base,pair['members'][1])['action'])
            if pair_doc['committed_start_hashes']!=[legacy.digest(data_a),legacy.digest(data_b)]:
                raise ValueError('Committed paired prediction hash changed')
            pair_start_idx=event_idx.get(('paired_start',pair['id']))
            member_start_idx=[event_idx.get(('start',cid)) for cid in pair['members']]
            if pair_start_idx is None or any(i is None or i>=pair_start_idx for i in member_start_idx):
                raise ValueError('Paired start does not follow both predictor commitments')
            for cid,fin in zip(pair['members'],finishes):
                if fin.exists():
                    if not (root/'private'/'receipts'/(cid+'.json')).exists():
                        raise ValueError('Missing paired operator receipt')
                    i_finish=event_idx.get(('finish',cid));i_receipt=event_idx.get(('paired_receipt',cid))
                    if (i_finish is None or i_receipt is None or
                            i_finish<=pair_start_idx or i_receipt<=i_finish):
                        raise ValueError('Invalid paired evidence ordering')
        elif any(f.exists() for f in starts+finishes):
            raise ValueError('Unpaired case was opened outside start-pair')
    if ('fit',None) in event_idx:
        if ('paired_fit',None) not in event_idx or event_idx[('paired_fit',None)]<=event_idx[('fit',None)]:
            raise ValueError('Missing verified paired qualification seal')
        _,unq_train=_qualified_pairs(root,pp,'train')
        # A finished but nonqualifying pair is permitted, but never used in fit.
        frozen=_read(root/'private'/'FROZEN_POLICY.json')
        qualifying_train,_=_qualified_pairs(root,pp,'train')
        eligible={c for p in qualifying_train for c in p['cases']}
        if not set(frozen['train_case_hashes'])<=eligible:
            raise ValueError('Unqualified training receipt used by policy')
    return {**result,'format':FORMAT+'-verification','pairs_planned':len(pp['pairs']),
            'paired_start_commitments':sum((root/'private'/'pairs'/(p['id']+'.start.json')).exists() for p in pp['pairs']),
            'automatic_writes':0}


def status(root:Path)->dict:
    root=Path(root);pp,base,_=_pair_workspace(root)
    frozen=(root/'private'/'FROZEN_POLICY.json').exists()
    info={};next_pair=None;next_action=None
    for split in legacy.SPLITS:
        done=ongoing=unopened=0
        for pair in pp['pairs']:
            if pair['split']!=split:continue
            start=(root/'private'/'pairs'/(pair['id']+'.start.json')).exists()
            finished=all((root/'private'/'cases'/(cid+'.finish.json')).exists() for cid in pair['members'])
            if finished:done+=1
            elif start:ongoing+=1
            else:unopened+=1
            if next_pair is None and not finished and (split!='holdout' or frozen):
                next_pair=pair['id'];next_action='finish members' if start else 'start-pair'
        good,bad=_qualified_pairs(root,pp,split)
        info[split]={'pairs_finished':done,'pairs_started_incomplete':ongoing,'pairs_unopened':unopened,
                     'qualifying_opposite_outcome_pairs':len(good),
                     'nonqualifying_pairs':{reason:sum(p['reason']==reason for p in bad)
                                             for reason in sorted({p['reason'] for p in bad})}}
    requirements=[]
    for op in sorted({pair['operation'] for pair in pp['pairs']}):
        for split,min_key in [('train','train_pairs_min'),('validation','validation_pairs_min')]:
            count=sum(pair['operation']==op for pair in _qualified_pairs(root,pp,split)[0])
            required=pp['quality_policy'][min_key]
            requirements.append({'operation':op,'split':split,'qualified_pairs':count,
                                 'minimum':required,'met':count>=required})
    finished_training=all(info[s]['pairs_unopened']==0 and info[s]['pairs_started_incomplete']==0
                          for s in ('train','validation'))
    if not frozen and finished_training:
        next_action='fit' if all(x['met'] for x in requirements) else 'BLOCKED_INSUFFICIENT_DISCRIMINATING_EVIDENCE'
        next_pair=None
    return {'format':FORMAT+'-operator-status','frozen':frozen,'splits':info,
            'minimum_evidence_requirements':requirements,
            'ready_to_freeze':finished_training and all(x['met'] for x in requirements),
            'next_action':next_action,'next_pair':next_pair,'network_calls':0,
            'automatic_writes':0,'privacy':'Local console details only; share public reports.'}


def report(root:Path)->dict:
    root=Path(root);pp,base,meta=_pair_workspace(root)
    if not (root/'private'/'FROZEN_POLICY.json').exists():
        raise ValueError('Cannot score before policy freeze')
    validation=verify(root)
    incomplete=[p['id'] for p in pp['pairs'] if p['split']=='holdout' and
                not all((root/'private'/'cases'/(cid+'.finish.json')).exists() for cid in p['members'])]
    if incomplete:
        raise ValueError(f'Finish all held-out cases before report; {len(incomplete)} incomplete pairs')
    usable,other=_qualified_pairs(root,pp,'holdout')
    good_ids={cid for p in usable for cid in p['cases']}
    lookup={c['id']:c for c in base['cases']}
    rows=[];provider={}
    for cid in sorted(good_ids):
        case=lookup[cid]
        before=_read(root/'private'/'cases'/(cid+'.start.json'))
        finish=_read(root/'private'/'cases'/(cid+'.finish.json'))
        rows.append({'id':cid,'operation':case['operation'],'truth':_effective_label(root,cid),
                     'predictions':before['predictions'],'actual_effects':finish['changed_fields']})
    arms={}
    named={'always_accept','always_reject','majority_baseline','structured_tree'}
    for row in rows:
        named.update(n for n in row['predictions'] if n.startswith('llm_'))
    for arm in sorted(named):
        truth_pred=[]
        for r in rows:
            pred='ACCEPT' if arm=='always_accept' else 'REJECT' if arm=='always_reject' else r['predictions'].get(arm,'ABSTAIN')
            if isinstance(pred,dict):pred=pred.get('decision','ABSTAIN')
            truth_pred.append((r['truth'],pred))
        score=legacy._scores(truth_pred,[])
        score['per_operation']={op:legacy._scores([x for x,r in zip(truth_pred,rows) if r['operation']==op],[])
                                  for op in sorted({r['operation'] for r in rows})}
        score['wilson95']=_public_confidence(score['correct'],score['cases'])
        score['accuracy_over_all_planned_cases']=round(score['correct']/sum(2 for p in pp['pairs'] if p['split']=='holdout'),6)
        arms[arm]=score
    # Same-draft-status makes a status-only model unable to distinguish pairs.
    # Its precommitted deterministic prediction is ACCEPT for every draft order.
    arms['status_only']={**legacy._scores([(r['truth'],'ACCEPT') for r in rows],[]),
         'definition':'Same-status status-only classifier, tied training labels use ACCEPT',
         'accuracy_over_all_planned_cases':round(sum(r['truth']=='ACCEPT' for r in rows)/max(1,sum(2 for p in pp['pairs'] if p['split']=='holdout')),6)}
    llm_present=all('llm_baseline' in r['predictions'] and 'llm_policy' in r['predictions'] for r in rows) and bool(rows)
    incremental=(bool(llm_present) and
                 arms['llm_policy']['accuracy_all_cases']>arms['llm_baseline']['accuracy_all_cases'])
    # Distinguish trivial status effects from non-status/cross-entity effects.
    non_status_effect_cases=sum(r['truth']=='ACCEPT' and any(
        k not in ('main.status','main.statut','main.fk_statut') for k in r['actual_effects']) for r in rows)
    cross_entity_effect_cases=sum(r['truth']=='ACCEPT' and any(
        k.startswith('watch_') for k in r['actual_effects']) for r in rows)
    effects=[r for r in rows if r['truth']=='ACCEPT' and isinstance(r['predictions'].get('structured_effects'),dict)]
    effect_exact=sum(r['predictions']['structured_effects']==legacy._effect_observed_after(r['actual_effects']) for r in effects)
    categories={}
    for pair in other:
        for cid in pair['cases']:
            receipt=_read(root/'private'/'receipts'/(cid+'.json'))
            status=receipt.get('status')
            label=('http_5xx' if isinstance(status,int) and status>=500 else
                   'http_permission_denied' if status in (401,403) else
                   'http_not_found' if status==404 else
                   'not_business_evidence' if _effective_label(root,cid)=='UNKNOWN' else
                   'nondiscriminating_pair')
            categories[label]=categories.get(label,0)+1
    per_op={}
    for op in sorted({p['operation'] for p in pp['pairs']}):
        n=[p for p in pp['pairs'] if p['split']=='holdout' and p['operation']==op]
        v=[p for p in usable if p['operation']==op]
        negative_cases=[cid for q in v for cid in q['cases'] if _effective_label(root,cid)=='REJECT']
        actual_business=sum(_read(root/'private'/'receipts'/(cid+'.json'))['outcome_attested']=='BUSINESS_REJECTED'
                            for cid in negative_cases)
        app500=sum(_read(root/'private'/'receipts'/(cid+'.json')).get('app_validation_500_correlated',False)
                   for cid in negative_cases)
        nontransition=sum(_read(root/'private'/'receipts'/(cid+'.json'))['outcome_attested']=='NO_OP'
                          for cid in negative_cases)
        per_op[op]={'pairs_planned':len(n),'discriminating_pairs':len(v),
                    'class_counts':{'ACCEPT':len(v),'REJECT':len(v)},
                    'negative_evidence_types':{'documented_business_rejections':actual_business,
                                               'http_304_no_state_transition':nontransition,
                                               'correlated_application_500_refusals':app500},
                    'sufficient_for_transition_quality_claim':len(v)>=pp['quality_policy']['holdout_pairs_min'],
                    'sufficient_for_business_rejection_claim':actual_business>=pp['quality_policy']['holdout_pairs_min']}
    truth_n=len(rows)
    quality=(all(v['sufficient_for_transition_quality_claim'] for v in per_op.values()) and
             bool(rows) and arms['structured_tree']['coverage']>=.95 and
             arms['structured_tree']['accuracy_all_cases']>=.95 and
             arms['structured_tree']['false_accepts']==0 and
             arms['structured_tree']['accuracy_all_cases']>arms['always_accept']['accuracy_all_cases'])
    calls=[p for r in rows for name,p in r['predictions'].items() if name.startswith('llm_') and isinstance(p,dict)]
    public={'format':FORMAT+'-public-report','conclusion':'SAME_STATUS_PAIRED_BUSINESS_RULE_EVIDENCE_OPERATOR_ATTESTED',
            'planned_holdout_pairs':sum(p['split']=='holdout' for p in pp['pairs']),
            'discriminating_holdout_pairs':len(usable),
            'holdout_unqualified_pairs':len(other),
            'usable_labeled_cases':truth_n,
            'all_planned_holdout_cases':sum(2 for p in pp['pairs'] if p['split']=='holdout'),
            'holdout_truth_complete_fraction':round(truth_n/max(1,sum(2 for p in pp['pairs'] if p['split']=='holdout')),6),
            'per_operation':per_op,'error_categories_on_unqualified_pairs':categories,
            'evidence_not_independently_attested':True,
            'read_only_watcher':True,'automatic_dolibarr_writes':0,
            'model_provider_usage_on_scored_cases':{'logical_calls':sum(int(p.get('logical_calls') or 0) for p in calls),
                                                    'input_tokens':sum(int(p.get('input_tokens') or 0) for p in calls),
                                                    'output_tokens':sum(int(p.get('output_tokens') or 0) for p in calls)},
            'scores':arms,'effect_exact_scored':effect_exact,'effect_cases_scored':len(effects),
            'effect_exact_rate':round(effect_exact/len(effects),6) if effects else None,
            'non_status_effect_cases':non_status_effect_cases,
            'cross_entity_effect_cases':cross_entity_effect_cases,
            'quality_gate_passed':quality,
            'incremental_model_refiner_advantage':incremental if llm_present else None,
            'llm_comparison_executed_on_every_scored_case':bool(llm_present),
            'quality_gate_vs_unadapted_llm_passed':bool(quality and incremental),
            'business_rejection_quality_gate_passed':bool(quality and all(
                x['sufficient_for_business_rejection_claim'] for x in per_op.values())),
            'quality_requirements':{'min_decision_accuracy':.95,'min_coverage':.95,
                                    'false_accepts':0,'minimum_valid_pairs_per_operation':pp['quality_policy']['holdout_pairs_min'],
                                    'better_than_always_accept':True,'better_than_unadapted_llm_required_for_adaptation_claim':True},
            'evidence_chain_sha256':validation['evidence_chain_sha256'],
            'plan_sha256':meta['plan_sha256'],
            'frozen_policy_sha256':legacy.digest(_read(root/'private'/'FROZEN_POLICY.json')),
            'notes':['Qualifying pairs contain one observed state change and a separately evidenced business-rule refusal.',
                     'HTTP 304 NO_OP is excluded from both training and business-rejection holdout labels.',
                     'Every unqualified pair reduces the planned-case coverage numerator.',
                     'A Dolibarr 500 may be classified as a business refusal only with a validation-error receipt, correlated local diagnostic, and unchanged state. Labels remain operator attested.',
                     'No real ERP authorization or autonomous writes follow from this research result.']}
    _save(root/'public'/'PAIRED_REPORT.json',public)
    freeze={'format':FORMAT+'-public-freeze','report_sha256':legacy.digest(public),
            'evidence_chain_sha256':validation['evidence_chain_sha256'],
            'plan_sha256':meta['plan_sha256'],'frozen_policy_sha256':public['frozen_policy_sha256'],
            'real_writes_by_harness':0,'externally_timestamped':False,
            'manual_outcome_receipts_not_independently_authenticated':True}
    _save(root/'public'/'PAIRED_AUDIT_FREEZE.json',freeze)
    return public


def main(argv:list[str]|None=None)->None:
    p=argparse.ArgumentParser(description='v0.5 same-status real Dolibarr business-rule learning; GET-only runner')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('pilot-start',help='GET-only same-status pilot: no mutations')
    a.add_argument('--root',type=Path,required=True)
    for name in ('order-a','order-b','product-a','product-b'):
        a.add_argument('--'+name,required=True)
    a.add_argument('--base-url',default='http://127.0.0.1:8088')
    a=sub.add_parser('pilot-finish',help='Manual pilot outcomes + new GET snapshots')
    a.add_argument('--root',type=Path,required=True)
    a.add_argument('--active-status',type=int,required=True)
    a.add_argument('--inactive-status',type=int,required=True)
    a.add_argument('--receipt-file',type=Path,required=True)
    a.add_argument('--diagnostic-file',type=Path,required=True)
    a=sub.add_parser('scaffold',help='Prepare study using successful pilot proof')
    a.add_argument('--pilot',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True)
    a.add_argument('--base-url',default='http://127.0.0.1:8088')
    for name,default in [('train-pairs',5),('validation-pairs',3),('holdout-pairs',10)]:
        a.add_argument('--'+name,type=int,default=default)
    a=sub.add_parser('bind');a.add_argument('--template',type=Path,required=True)
    a.add_argument('--csv',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('triage');a.add_argument('--receipt-status',type=int,required=True)
    a.add_argument('--state-changed',action='store_true')
    a.add_argument('--documented-business-code',action='store_true')
    a.add_argument('--writer-authorized',action='store_true')
    a.add_argument('--diagnostic-correlated',action='store_true')
    a=sub.add_parser('init');a.add_argument('--plan',type=Path,required=True);a.add_argument('--root',type=Path,required=True)
    a=sub.add_parser('start-pair');a.add_argument('--root',type=Path,required=True);a.add_argument('--pair',required=True)
    a.add_argument('--execute-model',action='store_true');a.add_argument('--share-sanitized-financials',action='store_true')
    a.add_argument('--model',default='gpt-6-luna')
    a=sub.add_parser('finish-case');a.add_argument('--root',type=Path,required=True);a.add_argument('--case',required=True)
    a.add_argument('--outcome',required=True,choices=legacy.KINDS)
    a.add_argument('--source',required=True,choices=('UI','MANUAL_API'))
    a.add_argument('--receipt-status',type=int)
    a.add_argument('--operator-role',default='unknown',choices=('authorized_writer','read_only','unknown'))
    a.add_argument('--business-reason',choices=RECEIPT_CLASS);a.add_argument('--business-code')
    a.add_argument('--receipt-file',type=Path);a.add_argument('--diagnostic-file',type=Path)
    for command in ('fit','status','verify','report'):
        a=sub.add_parser(command);a.add_argument('--root',type=Path,required=True)
    args=p.parse_args(argv)
    if args.command=='pilot-start':
        result=pilot_start(args.root,order_a=args.order_a,order_b=args.order_b,
                           product_a=args.product_a,product_b=args.product_b,base_url=args.base_url)
    elif args.command=='pilot-finish':
        result=pilot_finish(args.root,active_status=args.active_status,
                            inactive_status=args.inactive_status,
                            receipt_file=args.receipt_file,diagnostic_file=args.diagnostic_file)
    elif args.command=='scaffold':
        result=scaffold(args.out,pilot_file=args.pilot,train_pairs=args.train_pairs,
                        validation_pairs=args.validation_pairs,holdout_pairs=args.holdout_pairs,
                        base_url=args.base_url)
    elif args.command=='bind':result=bind(args.template,args.csv,args.out)
    elif args.command=='triage':result=triage(receipt_status=args.receipt_status,
                        witnessed_state_change=args.state_changed,
                        documented_business_code=args.documented_business_code,
                        authorized_writer=args.writer_authorized,
                        diagnostic_correlated=args.diagnostic_correlated)
    elif args.command=='init':result=init(args.plan,args.root)
    elif args.command=='start-pair':result=start_pair(args.root,args.pair,
                    execute_model=args.execute_model,share_financials=args.share_sanitized_financials,
                    model_name=args.model)
    elif args.command=='finish-case':result=finish_case(args.root,args.case,
                    outcome=args.outcome,source=args.source,receipt_status=args.receipt_status,
                    operator_role=args.operator_role,business_reason=args.business_reason,
                    business_code=args.business_code,receipt_file=args.receipt_file,
                    diagnostic_file=args.diagnostic_file)
    elif args.command=='fit':result=fit(args.root)
    elif args.command=='status':result=status(args.root)
    elif args.command=='verify':result=verify(args.root)
    else:result=report(args.root)
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__':main()
