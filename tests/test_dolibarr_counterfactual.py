from __future__ import annotations

import copy
import json
from pathlib import Path
import pytest

from simlearn import dolibarr_counterfactual as paired
from simlearn import dolibarr_transfer as old


class FakeGET:
    def __init__(self):
        self.rows={}
        self.calls=[]

    def request(self,method,path,data=None):
        assert method=='GET' and data is None
        self.calls.append((method,path))
        scope,ident=path.strip('/').split('/')[:2]
        return 200,copy.deepcopy(self.rows[(scope,ident)])


def make_workspace(tmp_path, operations=('order_validate',),training=4,validation=2,holdout=2):
    template=tmp_path/'fill.json'
    paired.scaffold(template,operations=operations,train_pairs=training,
                    validation_pairs=validation,holdout_pairs=holdout)
    p=json.loads(template.read_text('utf-8'))
    for index,c in enumerate(p['base_plan']['cases']):
        c['record_id']=str(1000+index)
    template.unlink();template.write_text(json.dumps(p))
    root=tmp_path/'workspace'
    paired.init(template,root)
    fake=FakeGET()
    for pair in p['pairs']:
        for side,cid in enumerate(pair['members']):
            c=old._case_lookup(p['base_plan'],cid)
            fake.rows[(old.OPS[c['operation']],c['record_id'])]={
                'status':str(side), 'total_ht':'50.00', 'total_ttc':'60.00',
                'name':'PRIVATE CLIENT','webservices_key':'PRIVATE_API_KEY'
            }
    return p,root,fake


def action_and_receipt(tmp_path,root,fake,pair,*,a_success=True,b_rejected=True):
    paired.start_pair(root,pair['id'],client=fake)
    case=old._case_lookup(paired._pair_workspace(root)[1],pair['members'][0])
    fam=old.OPS[case['operation']]
    if a_success:
        fake.rows[(fam,case['record_id'])]['status']='2'
        paired.finish_case(root,case['id'],outcome='SUCCESS',source='UI',operator_role='authorized_writer',client=fake)
    else:
        paired.finish_case(root,case['id'],outcome='OTHER_ERROR',source='MANUAL_API',
                           receipt_status=500,client=fake)
    other=pair['members'][1]
    if b_rejected:
        raw=tmp_path/f'{other}.txt'
        raw.write_text('Redacted test receipt: business rule; NO credentials here',encoding='utf-8')
        paired.finish_case(root,other,outcome='BUSINESS_REJECTED',source='MANUAL_API',
                           receipt_status=409,operator_role='authorized_writer',
                           business_reason='state_precondition',
                           business_code='ErrorWrongStatus',receipt_file=raw,client=fake)
    else:
        case_b=old._case_lookup(paired._pair_workspace(root)[1],other)
        fake.rows[(old.OPS[case_b['operation']],case_b['record_id'])]['status']='3'
        paired.finish_case(root,other,outcome='SUCCESS',source='UI',operator_role='authorized_writer',client=fake)


def test_scaffold_and_identity_pair_rules(tmp_path):
    path=tmp_path/'p.json'
    info=paired.scaffold(path,operations=('order_validate','invoice_validate'),
                         train_pairs=4,validation_pairs=2,holdout_pairs=2)
    assert info['manual_interventions']==32
    doc=json.loads(path.read_text())
    assert all(len(p['members'])==2 for p in doc['pairs'])
    for i,c in enumerate(doc['base_plan']['cases']):c['record_id']=str(1000+i)
    paired._assert_protocol(doc)
    doc['pairs'][0]['members'][0]=doc['pairs'][1]['members'][0]
    with pytest.raises(ValueError,match='(exactly one pair|mismatched)'):
        paired._assert_protocol(doc)


def test_unmatched_pair_refused_before_commit(tmp_path):
    p,root,fake=make_workspace(tmp_path)
    pair=p['pairs'][0];second=old._case_lookup(p['base_plan'],pair['members'][1])
    fake.rows[('orders',second['record_id'])]['total_ht']='100.00'
    with pytest.raises(ValueError,match='Pair not matched'):
        paired.start_pair(root,pair['id'],client=fake)
    assert not (root/'private'/'cases'/(pair['members'][0]+'.start.json')).exists()
    assert all(m=='GET' for m,_ in fake.calls)


def test_all_accept_dataset_blocks_freeze_and_holdout(tmp_path):
    p,root,fake=make_workspace(tmp_path)
    for pair in p['pairs']:
        if pair['split']=='holdout':continue
        action_and_receipt(tmp_path,root,fake,pair,b_rejected=False)
    with pytest.raises(ValueError,match='(qualifying|Block)'):
        paired.fit(root)
    assert not (root/'private'/'FROZEN_POLICY.json').exists()
    hold=next(p for p in p['pairs'] if p['split']=='holdout')
    call_count=len(fake.calls)
    with pytest.raises(ValueError,match='before a qualified frozen policy'):
        paired.start_pair(root,hold['id'],client=fake)
    assert len(fake.calls)==call_count


def test_403_and_500_cannot_become_business_rule_rejections(tmp_path):
    p,root,fake=make_workspace(tmp_path)
    pair=p['pairs'][0];paired.start_pair(root,pair['id'],client=fake)
    cid=pair['members'][1]
    receipt=tmp_path/'redacted.txt';receipt.write_text('Local-only sanitized log')
    with pytest.raises(ValueError,match='401/403'):
        paired.finish_case(root,cid,outcome='BUSINESS_REJECTED',source='MANUAL_API',
                           receipt_status=403,operator_role='authorized_writer',business_reason='state_precondition',
                           business_code='ErrorWrongStatus',receipt_file=receipt,client=fake)
    with pytest.raises(ValueError,match='5xx'):
        paired.finish_case(root,cid,outcome='BUSINESS_REJECTED',source='MANUAL_API',
                           receipt_status=500,operator_role='authorized_writer',business_reason='state_precondition',
                           business_code='ErrorWrongStatus',receipt_file=receipt,client=fake)
    with pytest.raises(ValueError,match='401/403'):
        paired.finish_case(root,cid,outcome='UNKNOWN',source='MANUAL_API',
                           receipt_status=403,client=fake)
    with pytest.raises(ValueError,match='5xx'):
        paired.finish_case(root,cid,outcome='UNKNOWN',source='MANUAL_API',
                           receipt_status=500,client=fake)
    result=paired.finish_case(root,cid,outcome='ACCESS_DENIED',source='MANUAL_API',
                              receipt_status=403,operator_role='read_only',client=fake)
    assert result['receipt_classification']=='UNKNOWN'
    assert all(m=='GET' for m,_ in fake.calls)


def test_business_evidence_requirements(tmp_path):
    p,root,fake=make_workspace(tmp_path)
    pair=p['pairs'][0];paired.start_pair(root,pair['id'],client=fake)
    cid=pair['members'][1]
    local=tmp_path/'receipt.txt';local.write_text('Sanitized status 409')
    with pytest.raises(ValueError,match='independent authorized writer'):
        paired.finish_case(root,cid,outcome='BUSINESS_REJECTED',source='UI',operator_role='authorized_writer',client=fake)
    with pytest.raises(ValueError,match='documented business'):
        paired.finish_case(root,cid,outcome='BUSINESS_REJECTED',source='MANUAL_API',
                           receipt_status=409,operator_role='authorized_writer',client=fake)
    with pytest.raises(ValueError,match='nonsecret server error code'):
        paired.finish_case(root,cid,outcome='BUSINESS_REJECTED',source='MANUAL_API',
                           receipt_status=409,operator_role='authorized_writer',business_reason='state_precondition',
                           business_code='x',receipt_file=local,client=fake)
    with pytest.raises(ValueError,match='Read-only/unknown'):
        paired.finish_case(root,cid,outcome='BUSINESS_REJECTED',source='MANUAL_API',
                           operator_role='read_only',receipt_status=409,
                           business_reason='state_precondition',
                           business_code='ErrorWrongStatus',receipt_file=local,client=fake)
    assert not (root/'private'/'cases'/(cid+'.finish.json')).exists()


def test_end_to_end_paired_holdout_class_balance_and_evidence_integrity(tmp_path):
    p,root,fake=make_workspace(tmp_path,operations=('order_validate','invoice_validate'))
    for pair in p['pairs']:
        if pair['split']=='holdout':continue
        action_and_receipt(tmp_path,root,fake,pair)
    outcome=paired.fit(root)
    assert outcome['qualified_train_pairs']==8
    assert outcome['qualified_validation_pairs']==4
    assert outcome['all_required_operations_trained']
    for pair in p['pairs']:
        if pair['split']!='holdout':continue
        action_and_receipt(tmp_path,root,fake,pair)
    verification=paired.verify(root)
    assert verification['valid'] and verification['paired_start_commitments']==len(p['pairs'])
    report=paired.report(root)
    assert report['usable_labeled_cases']==8
    assert report['discriminating_holdout_pairs']==4
    assert report['holdout_truth_complete_fraction']==1.0
    assert report['scores']['always_accept']['accuracy_all_cases']==0.5
    assert report['scores']['always_reject']['accuracy_all_cases']==0.5
    assert report['scores']['structured_tree']['accuracy_all_cases']>=0.75
    assert all(v['class_counts']=={'ACCEPT':2,'REJECT':2} for v in report['per_operation'].values())
    public=(root/'public'/'PAIRED_REPORT.json').read_text()
    for c in p['base_plan']['cases']:
        # A private ID must not appear as a standalone value/token. A coincidental
        # decimal substring inside a SHA-256 digest is not an ID disclosure.
        import re
        assert re.search(r'(?<![0-9A-Za-z])'+re.escape(c['record_id'])+r'(?![0-9A-Za-z])',public) is None
    assert 'PRIVATE' not in public and 'ErrorWrongStatus' not in public
    assert all(method=='GET' for method,_ in fake.calls)
    with pytest.raises(FileExistsError):paired.report(root)


def test_tampered_manual_receipt_rejected_by_chain(tmp_path):
    p,root,fake=make_workspace(tmp_path)
    pair=p['pairs'][0]
    action_and_receipt(tmp_path,root,fake,pair)
    cid=pair['members'][1]
    path=root/'private'/'receipts'/(cid+'.json')
    doc=json.loads(path.read_text())
    doc['business_reason']='other_documented_business_rejection'
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='modified'):
        paired.verify(root)


def test_extra_outcome_categories_and_insufficient_holdout_report(tmp_path):
    p,root,fake=make_workspace(tmp_path)
    for pair in p['pairs']:
        if pair['split']=='holdout':continue
        action_and_receipt(tmp_path,root,fake,pair)
    paired.fit(root)
    pair=next(q for q in p['pairs'] if q['split']=='holdout')
    action_and_receipt(tmp_path,root,fake,pair,a_success=False,b_rejected=True)
    for rest in [q for q in p['pairs'] if q['split']=='holdout' and q['id']!=pair['id']]:
        action_and_receipt(tmp_path,root,fake,rest)
    summary=paired.report(root)
    assert summary['holdout_unqualified_pairs']==1
    assert summary['error_categories_on_unqualified_pairs']['http_5xx']==1
    assert summary['holdout_truth_complete_fraction']==0.5
    assert not summary['quality_gate_passed']


def test_credential_never_recorded_and_no_writes(tmp_path,monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    class Handler(BaseHTTPRequestHandler):
        seen=[]
        def do_GET(self):
            Handler.seen.append((self.command,self.path,self.headers.get('DOLAPIKEY')))
            k=self.path.split('/')[-1]
            raw=json.dumps({'status':str(int(k)%2),'total_ht':'50.00',
                           'name':'SECRET_NAME','webservices_key':'SECRET_KEY'}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def do_POST(self):raise AssertionError('No POST should reach the test server')
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        template=tmp_path/'fill.json'
        paired.scaffold(template,train_pairs=4,validation_pairs=2,holdout_pairs=2,
                        base_url=f'http://127.0.0.1:{server.server_port}')
        p=json.loads(template.read_text('utf-8'))
        for i,c in enumerate(p['base_plan']['cases']):c['record_id']=str(2000+i)
        template.write_text(json.dumps(p))
        root=tmp_path/'workspace';paired.init(template,root)
        monkeypatch.setenv('DOLIBARR_API_KEY','SECRETS_NEVER_PERSISTED')
        pair=p['pairs'][0]
        paired.start_pair(root,pair['id'])
        assert Handler.seen and all(m=='GET' for m,_,_ in Handler.seen)
        assert all(k=='SECRETS_NEVER_PERSISTED' for _,_,k in Handler.seen)
        source=''.join(x.read_text('utf-8') for x in (root/'private'/'cases').glob('*.json'))
        assert 'SECRET_NAME' not in source and 'SECRET_KEY' not in source
        assert 'SECRETS_NEVER_PERSISTED' not in source
    finally:
        server.shutdown();server.server_close();thread.join(timeout=5)


def test_http_304_no_op_counts_as_negative_transition_not_business_reject(tmp_path):
    p,root,fake=make_workspace(tmp_path)
    receipt=tmp_path/'evidence.txt';receipt.write_text('Sanitized Dolibarr 304 Not Modified')
    for pair in p['pairs']:
        if pair['split']=='holdout':continue
        paired.start_pair(root,pair['id'],client=fake)
        ca=old._case_lookup(p['base_plan'],pair['members'][0])
        fake.rows[('orders',ca['record_id'])]['status']='2'
        paired.finish_case(root,ca['id'],outcome='SUCCESS',source='UI',operator_role='authorized_writer',client=fake)
        cb=pair['members'][1]
        got=paired.finish_case(root,cb,outcome='NO_OP',source='MANUAL_API',receipt_status=304,
                               operator_role='authorized_writer',receipt_file=receipt,client=fake)
        assert got['receipt_classification']=='REJECT'
    paired.fit(root)
    for pair in p['pairs']:
        if pair['split']!='holdout':continue
        paired.start_pair(root,pair['id'],client=fake)
        ca=old._case_lookup(p['base_plan'],pair['members'][0])
        fake.rows[('orders',ca['record_id'])]['status']='2'
        paired.finish_case(root,ca['id'],outcome='SUCCESS',source='UI',operator_role='authorized_writer',client=fake)
        paired.finish_case(root,pair['members'][1],outcome='NO_OP',source='MANUAL_API',
                           receipt_status=304,operator_role='authorized_writer',receipt_file=receipt,client=fake)
    result=paired.report(root)
    breakdown=result['per_operation']['order_validate']['negative_evidence_types']
    assert breakdown['http_304_no_state_transition']==2
    assert breakdown['documented_business_rejections']==0
    assert not result['business_rejection_quality_gate_passed']
    assert result['scores']['always_accept']['accuracy_all_cases']==.5


def test_triage_blocks_server_error_and_distinguishes_noops():
    assert paired.triage(receipt_status=500,witnessed_state_change=False,authorized_writer=True)['eligible_candidate'] is False
    assert paired.triage(receipt_status=403,witnessed_state_change=False,authorized_writer=True)['outcome_category']=='ACCESS_DENIED'
    assert paired.triage(receipt_status=304,witnessed_state_change=False,authorized_writer=True)['outcome_category']=='NO_OP_CANDIDATE'
    assert paired.triage(receipt_status=400,witnessed_state_change=False,authorized_writer=True)['eligible_candidate'] is False
    assert paired.triage(receipt_status=400,witnessed_state_change=False,documented_business_code=True,
                         authorized_writer=True)['outcome_category']=='BUSINESS_REJECT_CANDIDATE'
    assert paired.triage(receipt_status=200,witnessed_state_change=True,authorized_writer=True)['outcome_category']=='SUCCESS_CANDIDATE'


def test_csv_fixture_binding_and_tamper_rejection(tmp_path):
    t=tmp_path/'editable.json';paired.scaffold(t,train_pairs=4,validation_pairs=2,holdout_pairs=2)
    cp=tmp_path/'editable_IDS.csv'
    import csv
    with cp.open('r',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    for idx,row in enumerate(rows):row['record_id']=str(idx+500)
    with cp.open('w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=['case_id','pair_id','operation','split','record_id','payment_amount'])
        wr.writeheader();wr.writerows(rows)
    ready=tmp_path/'ready.json'
    out=paired.bind(t,cp,ready)
    assert out['cases_bound']==len(rows)
    assert all(c['record_id']!='FILL_ID' for c in json.loads(ready.read_text())['base_plan']['cases'])
    with pytest.raises(FileExistsError):paired.bind(t,cp,ready)
    rows[0]['split']='holdout'
    with cp.open('w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=['case_id','pair_id','operation','split','record_id','payment_amount'])
        wr.writeheader();wr.writerows(rows)
    with pytest.raises(ValueError,match='immutable assignment'):
        paired.bind(t,cp,tmp_path/'bad.json')
