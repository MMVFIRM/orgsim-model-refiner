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

FORMAT = 'simlearn-dolibarr-paired-v2'
RECEIPT_CLASS = ('state_precondition', 'missing_business_required_field',
                 'business_amount_limit', 'duplicate_state_transition',
                 'other_documented_business_rejection')
SAFE_BUSINESS_STATUS = {400, 409, 422}
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
           documented_business_code:bool=False, authorized_writer:bool=True)->dict:
    """Offline diagnostics before investing in fixture construction.

    Never claims that an HTTP error alone proves a business-rule decision.
    """
    if not 100<=receipt_status<=599:raise ValueError('Provide a real HTTP status')
    if not authorized_writer:
        label='ACCESS_OR_ROLE_UNKNOWN';usable=False
        advice='Use a separately authorized sandbox write identity. Read-only 403 is not business-rule evidence.'
    elif 200<=receipt_status<300 and witnessed_state_change:
        label='SUCCESS_CANDIDATE';usable=True
        advice='Eligible only after committed before/after snapshots corroborate the changed fields.'
    elif receipt_status==304 and not witnessed_state_change:
        label='NO_OP_CANDIDATE';usable=True
        advice='A 304 can label a non-transition, not a business-rule rejection. Preserve a redacted receipt.'
    elif receipt_status in SAFE_BUSINESS_STATUS and not witnessed_state_change and documented_business_code:
        label='BUSINESS_REJECT_CANDIDATE';usable=True
        advice='Requires a locally retained redacted receipt, a real error code, and correct permissions.'
    elif receipt_status in (401,403):
        label='ACCESS_DENIED';usable=False
        advice='Do not turn an authorization denial into a business-rule label.'
    elif receipt_status==404:
        label='MISSING_ROUTE_OR_RECORD';usable=False
        advice='Check endpoint, fixture ID, required modules, and role configuration.'
    elif receipt_status>=500:
        label='SERVER_ERROR_UNLABELED';usable=False
        advice='Examine local Dolibarr/PHP logs; 500 alone is not evidence of business rejection.'
    elif witnessed_state_change:
        label='MISMATCHED_RECEIPT_AND_STATE';usable=False
        advice='Investigate out-of-band changes before using the fixture.'
    else:
        label='UNKNOWN_UNLABELED';usable=False
        advice='No observed state change and no supported rejection/no-op evidence.'
    return {'format':FORMAT+'-operator-preflight','reported_status':receipt_status,
            'outcome_category':label,'eligible_candidate':usable,
            'note':'Offline triage only. This neither contacts Dolibarr nor independently attests the event.',
            'next_step':advice,'network_calls':0,'automatic_writes':0}


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


def _assert_protocol(p: dict) -> dict:
    if not isinstance(p, dict) or set(p) != {'format','base_plan','pairs','quality_policy'} or p['format'] != FORMAT+'-plan':
        raise ValueError('Wrong v0.4 paired plan format')
    base=p['base_plan']
    legacy.validate_plan(base)
    rules=p['quality_policy']
    if set(rules) != {'train_pairs_min','validation_pairs_min','holdout_pairs_min'}:
        raise ValueError('Invalid evidence policy')
    if any(type(v) is not int or v < 1 or v > MAX_PAIRS for v in rules.values()):
        raise ValueError('Invalid pair minima')
    pairs=p['pairs']
    if not isinstance(pairs, list) or not pairs or len(pairs) > MAX_PAIRS:
        raise ValueError('Invalid number of paired fixtures')
    case_index={c['id']:c for c in base['cases']}
    assigned=[]
    ids=set()
    for pair in pairs:
        if (not isinstance(pair,dict) or set(pair) != {'id','operation','split','members','contrast_keys','match_keys'}
                or not isinstance(pair['id'],str) or not PAIR_KEY.fullmatch(pair['id'])
                or pair['id'] in ids):
            raise ValueError('Pair has invalid identifier or keys')
        ids.add(pair['id'])
        if pair['split'] not in legacy.SPLITS or pair['operation'] not in legacy.OPS:
            raise ValueError('Invalid pair operation/split')
        members=pair['members']
        if not isinstance(members,list) or len(members)!=2 or len(set(members))!=2:
            raise ValueError('Every pair needs exactly two distinct members')
        if any(mid not in case_index for mid in members):
            raise ValueError('Pair refers to unknown case')
        if any(case_index[mid]['operation']!=pair['operation'] or case_index[mid]['split']!=pair['split'] for mid in members):
            raise ValueError('Pair contains mismatched operations/splits')
        contrast=pair['contrast_keys'];match=pair['match_keys']
        if (not isinstance(contrast,list) or not 1<=len(contrast)<=3 or
                any(not isinstance(x,str) or x not in _allowed_feature_keys(pair['operation']) for x in contrast) or
                len(set(contrast))!=len(contrast)):
            raise ValueError('Invalid contrast_keys; use allowlisted snapshot features')
        if (not isinstance(match,list) or len(match)>5 or any(not isinstance(x,str) or
                x not in _allowed_feature_keys(pair['operation']) or x in contrast for x in match)
                or len(set(match))!=len(match)):
            raise ValueError('Invalid match_keys')
        assigned.extend(members)
    if set(assigned)!=set(case_index) or len(assigned)!=len(set(assigned)):
        raise ValueError('Every case must belong to exactly one pair')
    for op in {pair['operation'] for pair in pairs}:
        for split,limit in [('train',rules['train_pairs_min']),('validation',rules['validation_pairs_min']),
                            ('holdout',rules['holdout_pairs_min'])]:
            if sum(pair['operation']==op and pair['split']==split for pair in pairs)<limit:
                raise ValueError(f'{op} has insufficient {split} pair slots')
    return p


def _allowed_feature_keys(operation:str)->set[str]:
    fields={'main.'+k for k in legacy.SAFE_FIELDS[legacy.OPS[operation]]}
    fields.add('main.lines_count')
    if operation=='invoice_payment':fields.add('action.payment_amount')
    return fields


def scaffold(output:Path, *, operations:tuple[str,...]=('order_validate',),
             train_pairs:int=6,validation_pairs:int=3,holdout_pairs:int=10,
             base_url:str='http://127.0.0.1:8088')->dict:
    if not operations or len(set(operations))!=len(operations) or any(x not in legacy.OPS for x in operations):
        raise ValueError('Unknown or repeated operation')
    if not 4<=train_pairs<=30 or not 2<=validation_pairs<=20 or not 2<=holdout_pairs<=30:
        raise ValueError('Pairs must provide at least four train, two validation and two holdout pairs')
    output=Path(output)
    csv_path=output.with_name(output.stem+'_IDS.csv')
    if output.exists() or csv_path.exists():
        raise FileExistsError('Refusing to overwrite an existing scaffold/CSV')
    cases=[];pairs=[]
    for op in operations:
        for split,n in [('train',train_pairs),('validation',validation_pairs),('holdout',holdout_pairs)]:
            for i in range(1,n+1):
                pair_id=f'{split}-{op}-{i:03d}'
                members=[f'{pair_id}-a',f'{pair_id}-b']
                for c_id in members:
                    cases.append({'id':c_id,'operation':op,'split':split,'record_id':'FILL_ID',
                                  'watch':[],
                                  'action':{'payment_amount':'FILL_AMOUNT'} if op=='invoice_payment' else {}})
                contrast=['action.payment_amount'] if op=='invoice_payment' else ['main.status','main.statut']
                pairs.append({'id':pair_id,'operation':op,'split':split,'members':members,
                              'contrast_keys':contrast,'match_keys':[]})
    base={'format':legacy.FORMAT+'-plan','tenant_label':'sandbox_local_002',
          'snapshot_label':'before_paired_trial_002','base_url':base_url,'cases':cases}
    p={'format':FORMAT+'-plan','base_plan':base,'pairs':pairs,
       'quality_policy':{'train_pairs_min':4,'validation_pairs_min':2,
                         'holdout_pairs_min':min(8,holdout_pairs)}}
    _save(output,p)
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    with csv_path.open('x',encoding='utf-8',newline='') as f:
        wr=csv.writer(f,lineterminator='\n')
        wr.writerow(['case_id','pair_id','operation','split','record_id','payment_amount'])
        membership={cid:pair['id'] for pair in pairs for cid in pair['members']}
        for c in cases:
            wr.writerow([c['id'],membership[c['id']],c['operation'],c['split'],'',''])
    return {'output':str(output),'fixture_id_csv':str(csv_path),
            'planned_pairs':len(pairs),'manual_interventions':len(cases),
            'operations':list(operations),'note':'Fill distinct disposable record IDs and construct matched pairs. Preview only; no ERP requests.'}


def bind(template:Path, fixture_csv:Path, output:Path)->dict:
    """Offline import of user-seeded disposable IDs. No actual ERP writes."""
    p=_read(Path(template))
    if p.get('format')!=FORMAT+'-plan':raise ValueError('Wrong paired fixture template')
    base=p['base_plan']
    all_cases={c['id']:c for c in base['cases']}
    with Path(fixture_csv).open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['case_id','pair_id','operation','split','record_id','payment_amount']:
            raise ValueError('Use the generated fixture CSV columns unchanged')
        rows=list(reader)
    membership={cid:pair['id'] for pair in p['pairs'] for cid in pair['members']}
    seen=set();ids=set()
    for row in rows:
        name=row['case_id']
        if name not in all_cases or name in seen:raise ValueError('Unknown/duplicate fixture CSV case')
        seen.add(name)
        c=all_cases[name]
        if (row['pair_id']!=membership[name] or row['operation']!=c['operation'] or row['split']!=c['split']):
            raise ValueError('Fixture row attempts to alter immutable assignment')
        record=row['record_id'].strip()
        if not legacy.ID.fullmatch(record) or record=='0':
            raise ValueError(f'Fill a safe positive numeric record_id for {name}')
        key=(legacy.OPS[c['operation']],record)
        if key in ids:raise ValueError('Duplicate Dolibarr record ID; every fixture must be independent')
        ids.add(key);c['record_id']=record
        payment=row['payment_amount'].strip()
        if c['operation']=='invoice_payment':
            if not legacy.NUM.fullmatch(payment) or legacy._decimal(payment)<=0:
                raise ValueError(f'Provide positive payment_amount for {name}')
            c['action']['payment_amount']=payment
        elif payment:
            raise ValueError('payment_amount must remain blank for non-payment operations')
    if seen!=set(all_cases):raise ValueError('Fixture ID CSV is incomplete')
    _assert_protocol(p)
    _save(Path(output),p)
    return {'format':FORMAT+'-bound','output':str(output),'cases_bound':len(rows),
            'distinct_fixture_records':len(ids),'erp_requests':0,'model_requests':0,'automated_writes':0}



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


def _pair_difference(pair:dict, left:dict, right:dict,
                     left_action:dict, right_action:dict)->dict:
    lf=legacy._features(left,left_action)
    rf=legacy._features(right,right_action)
    if left['main'].get('http_status')!=200 or right['main'].get('http_status')!=200 or \
       left['main'].get('kind')!='object' or right['main'].get('kind')!='object':
        raise ValueError('Both matched records must be readable 200 object snapshots')
    keys=set(lf)|set(rf)
    diff=[k for k in sorted(keys) if (k not in lf or k not in rf or not _equal(lf[k],rf[k]))]
    unexpected=sorted(set(diff)-set(pair['contrast_keys']))
    if unexpected:
        raise ValueError(f'Pair not matched: unexpected fields differ {unexpected[:8]}. Re-seed fixtures; do not reinterpret the contrast.')
    observed=[k for k in pair['contrast_keys'] if k in lf and k in rf and not _equal(lf[k],rf[k])]
    if not observed:
        raise ValueError('Pair did not actually differ on an observed contrast field')
    for k in pair['match_keys']:
        if k not in lf or k not in rf or not _equal(lf[k],rf[k]):
            raise ValueError(f'Mandatory match feature {k} missing/different')
    # Relationship watches must also remain matched; otherwise isolation is unclear.
    if set(left)!=set(right):
        raise ValueError('Pair watches are not symmetric')
    for scope in left:
        if scope=='main':continue
        if left[scope]!=right[scope]:
            raise ValueError('Watched related record changed across matched fixtures')
    return {'verified':True,'contrast_observed':observed,'unexpected_differences':0,
            'shared_features':len(keys),'snapshot_limited':True}


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
            'commit_precedes_manual_intervention':True,'operator_action_executed':False}
    rel='private/pairs/'+pair_id+'.start.json'
    _save(root/rel,record);_journal(root,'paired_start',pair_id,rel,record)
    return {'pair_id':pair_id,'split':pair['split'],'record_ids_local_only':[case_a['record_id'],case_b['record_id']],
            'cases':pair['members'],'quality':actual,'preflight_reads':2,'committed_reads':2,
            'model_calls':4*len(pair['members']) if execute_model else 0,
            'automated_writes':0,
            'instruction':'Both immutable before-states and holdout predictions are committed. Now perform the two actions independently; finish each case.'}


def _file_hash(path:Path|None)->str|None:
    if path is None:return None
    p=Path(path)
    if not p.is_file() or p.stat().st_size>2_000_000 or p.stat().st_size==0:
        raise ValueError('Receipt evidence file must be a non-empty local file <=2MB')
    return hashlib.sha256(p.read_bytes()).hexdigest()


def finish_case(root:Path,case_id:str,*,outcome:str,source:str,
                receipt_status:int|None=None,operator_role:str='unknown',
                business_reason:str|None=None,business_code:str|None=None,
                receipt_file:Path|None=None,client=None)->dict:
    root=Path(root);pp,base,_=_pair_workspace(root)
    pair=_pair_for_case(pp,case_id)
    if not (root/'private'/'pairs'/(pair['id']+'.start.json')).exists():
        raise ValueError('Use start-pair to freeze both pair members BEFORE interventions')
    if not (root/'private'/'cases'/(case_id+'.start.json')).exists():
        raise ValueError('Missing pre-action snapshot')
    if (root/'private'/'cases'/(case_id+'.finish.json')).exists():
        raise FileExistsError('Case already finished')
    if source not in ('UI','MANUAL_API') or outcome not in legacy.KINDS:
        raise ValueError('Invalid outcome/source')
    if operator_role not in ('authorized_writer','read_only','unknown'):
        raise ValueError('Invalid intervention role')
    if source=='MANUAL_API' and (receipt_status is None or not 100<=receipt_status<=599):
        raise ValueError('Manual API attempt requires its real observed HTTP status')
    if operator_role!='authorized_writer' and outcome in ('SUCCESS','BUSINESS_REJECTED','NO_OP'):
        raise ValueError('Read-only/unknown operator identity cannot supply positive business-decision labels')
    if source=='MANUAL_API':
        if receipt_status in (401,403) and outcome!='ACCESS_DENIED':
            raise ValueError('401/403 must be ACCESS_DENIED, not a business-rule rejection')
        if receipt_status>=500 and outcome!='OTHER_ERROR':
            raise ValueError('5xx must be OTHER_ERROR, not a business-rule rejection')
        if receipt_status==304 and outcome!='NO_OP':
            raise ValueError('304 Not Modified must be classified as NO_OP')
        if receipt_status==404 and outcome not in ('OTHER_ERROR','UNKNOWN'):
            raise ValueError('404 requires investigation as missing route/fixture, not a business decision')
        if 200<=receipt_status<=299 and outcome!='SUCCESS':
            raise ValueError('2xx must be SUCCESS, unless inspection is explicitly marked UNKNOWN')
    if outcome=='NO_OP':
        if source!='MANUAL_API' or receipt_status!=304 or receipt_file is None:
            raise ValueError('NO_OP requires a verified operator-attested HTTP 304 and local redacted receipt file')
    if outcome=='BUSINESS_REJECTED':
        if source!='MANUAL_API' or receipt_status not in SAFE_BUSINESS_STATUS:
            raise ValueError('Usable business rejects need an independent authorized writer API response with HTTP 400/409/422')
        if business_reason not in RECEIPT_CLASS:
            raise ValueError('Classify the documented business-rule failure using --business-reason')
        if not isinstance(business_code,str) or not SAFE_CODE.fullmatch(business_code):
            raise ValueError('Provide a nonsecret server error code token via --business-code')
        if receipt_file is None:
            raise ValueError('Provide a local redacted response/receipt file via --receipt-file for negative evidence')
    elif business_reason is not None or business_code is not None:
        raise ValueError('Business explanation/code is only for real BUSINESS_REJECTED receipts')
    receipt_digest=_file_hash(receipt_file)
    result=legacy.finish(root,case_id,outcome=outcome,source=source,
                         receipt_status=receipt_status,client=client)
    # UI rejects remain ineligible by design; BUSINESS_REJECTED can only be
    # attributed to the business layer with the extra receipt/role evidence.
    receipt={'format':FORMAT+'-operator-receipt','case_id':case_id,'pair_id':pair['id'],
             'status':receipt_status,'source':source,'operator_role':operator_role,
             'outcome_attested':outcome,'business_reason':business_reason,
             'business_code':business_code,'receipt_sha256':receipt_digest,
             'business_evidence_is_operator_classified_not_cryptographically_authenticated':True}
    rel='private/receipts/'+case_id+'.json'
    _save(root/rel,receipt);_journal(root,'paired_receipt',case_id,rel,receipt)
    return {**result,'receipt_classification':_effective_label(root,case_id),
            'real_server_receipt_local_only':receipt_digest is not None,
            'automated_writes':0}


def _effective_label(root:Path,case_id:str)->str:
    ff=root/'private'/'cases'/(case_id+'.finish.json')
    rp=root/'private'/'receipts'/(case_id+'.json')
    if not ff.exists() or not rp.exists():return 'MISSING'
    finished=_read(ff);receipt=_read(rp)
    if finished['effective_truth']=='ACCEPT' and receipt['operator_role']=='authorized_writer':
        return 'ACCEPT'
    if (finished['effective_truth']=='REJECT' and receipt['outcome_attested']=='NO_OP' and
            receipt['operator_role']=='authorized_writer' and receipt['source']=='MANUAL_API' and
            receipt['status']==304 and receipt['receipt_sha256']):
        return 'REJECT'
    if (finished['effective_truth']=='REJECT' and receipt['operator_role']=='authorized_writer'
            and receipt['source']=='MANUAL_API' and receipt['status'] in SAFE_BUSINESS_STATUS
            and receipt['business_reason'] in RECEIPT_CLASS and receipt['business_code']
            and receipt['receipt_sha256']):
        return 'REJECT'
    return 'UNKNOWN'


def _qualified_pairs(root:Path,pp:dict,split:str)->tuple[list[dict],list[dict]]:
    qualified=[];other=[]
    for pair in pp['pairs']:
        if pair['split']!=split:continue
        labels=[_effective_label(root,case_id) for case_id in pair['members']]
        item={'pair_id':pair['id'],'operation':pair['operation'],'split':split,
              'cases':pair['members'],'labels':labels}
        if set(labels)=={'ACCEPT','REJECT'}:
            qualified.append(item)
        else:
            item['reason']=('unfinished_or_missing_receipt' if 'MISSING' in labels else
                            'unknown_error_or_access_control' if 'UNKNOWN' in labels else
                            'nondiscriminating_same_class')
            other.append(item)
    return qualified,other


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
        nontransition=sum(_read(root/'private'/'receipts'/(cid+'.json'))['outcome_attested']=='NO_OP'
                          for cid in negative_cases)
        per_op[op]={'pairs_planned':len(n),'discriminating_pairs':len(v),
                    'class_counts':{'ACCEPT':len(v),'REJECT':len(v)},
                    'negative_evidence_types':{'documented_business_rejections':actual_business,
                                               'http_304_no_state_transition':nontransition},
                    'sufficient_for_transition_quality_claim':len(v)>=pp['quality_policy']['holdout_pairs_min'],
                    'sufficient_for_business_rejection_claim':actual_business>=pp['quality_policy']['holdout_pairs_min']}
    truth_n=len(rows)
    quality=(all(v['sufficient_for_transition_quality_claim'] for v in per_op.values()) and
             bool(rows) and arms['structured_tree']['coverage']>=.95 and
             arms['structured_tree']['accuracy_all_cases']>=.95 and
             arms['structured_tree']['false_accepts']==0 and
             arms['structured_tree']['accuracy_all_cases']>arms['always_accept']['accuracy_all_cases'])
    calls=[p for r in rows for name,p in r['predictions'].items() if name.startswith('llm_') and isinstance(p,dict)]
    public={'format':FORMAT+'-public-report','conclusion':'MANUAL_ATTESTATION_PAIRED_SIM_TO_REAL_NOT_INDEPENDENT_CERTIFICATION',
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
            'quality_gate_passed':quality,
            'business_rejection_quality_gate_passed':bool(quality and all(
                x['sufficient_for_business_rejection_claim'] for x in per_op.values())),
            'quality_requirements':{'min_decision_accuracy':.95,'min_coverage':.95,
                                    'false_accepts':0,'minimum_valid_pairs_per_operation':pp['quality_policy']['holdout_pairs_min'],
                                    'better_than_always_accept':True},
            'evidence_chain_sha256':validation['evidence_chain_sha256'],
            'plan_sha256':meta['plan_sha256'],
            'frozen_policy_sha256':legacy.digest(_read(root/'private'/'FROZEN_POLICY.json')),
            'notes':['Qualifying holdout pairs contain one observed state change and one receipt-supported non-transition.',
                     'An HTTP 304 NO_OP is a non-transition, not proof of a business-rule refusal.',
                     'Every unqualified pair reduces the planned-case coverage numerator.',
                     'Labels and rejection categories remain operator attested; no independent transaction audit.',
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
    p=argparse.ArgumentParser(description='v0.4 paired, manual-write-only Dolibarr transfer test')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('scaffold');a.add_argument('--out',type=Path,required=True)
    a.add_argument('--base-url',default='http://127.0.0.1:8088')
    a.add_argument('--operation',action='append',choices=sorted(legacy.OPS))
    for x,default in [('train-pairs',6),('validation-pairs',3),('holdout-pairs',10)]:
        a.add_argument('--'+x,type=int,default=default)
    a=sub.add_parser('bind');a.add_argument('--template',type=Path,required=True)
    a.add_argument('--csv',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('triage');a.add_argument('--receipt-status',required=True,type=int)
    a.add_argument('--state-changed',action='store_true');a.add_argument('--documented-business-code',action='store_true')
    a.add_argument('--writer-authorized',action='store_true')
    a=sub.add_parser('init');a.add_argument('--plan',required=True,type=Path);a.add_argument('--root',required=True,type=Path)
    a=sub.add_parser('start-pair');a.add_argument('--root',required=True,type=Path);a.add_argument('--pair',required=True)
    a.add_argument('--execute-model',action='store_true');a.add_argument('--share-sanitized-financials',action='store_true')
    a.add_argument('--model',default='gpt-6-luna')
    a=sub.add_parser('finish-case');a.add_argument('--root',required=True,type=Path);a.add_argument('--case',required=True)
    a.add_argument('--outcome',required=True,choices=legacy.KINDS)
    a.add_argument('--source',required=True,choices=('UI','MANUAL_API'))
    a.add_argument('--receipt-status',type=int);a.add_argument('--operator-role',default='unknown',
                                                            choices=('authorized_writer','read_only','unknown'))
    a.add_argument('--business-reason',choices=RECEIPT_CLASS);a.add_argument('--business-code')
    a.add_argument('--receipt-file',type=Path)
    for command in ('fit','status','verify','report'):
        a=sub.add_parser(command);a.add_argument('--root',type=Path,required=True)
    args=p.parse_args(argv)
    if args.command=='scaffold':
        result=scaffold(args.out,operations=tuple(args.operation or ['order_validate']),base_url=args.base_url,
                        train_pairs=args.train_pairs,validation_pairs=args.validation_pairs,holdout_pairs=args.holdout_pairs)
    elif args.command=='bind':result=bind(args.template,args.csv,args.out)
    elif args.command=='triage':result=triage(receipt_status=args.receipt_status,witnessed_state_change=args.state_changed,
                    documented_business_code=args.documented_business_code,authorized_writer=args.writer_authorized)
    elif args.command=='init':result=init(args.plan,args.root)
    elif args.command=='start-pair':result=start_pair(args.root,args.pair,execute_model=args.execute_model,
                    share_financials=args.share_sanitized_financials,model_name=args.model)
    elif args.command=='finish-case':result=finish_case(args.root,args.case,outcome=args.outcome,
                    source=args.source,receipt_status=args.receipt_status,
                    operator_role=args.operator_role,business_reason=args.business_reason,
                    business_code=args.business_code,receipt_file=args.receipt_file)
    elif args.command=='fit':result=fit(args.root)
    elif args.command=='status':result=status(args.root)
    elif args.command=='verify':result=verify(args.root)
    else:result=report(args.root)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':main()
