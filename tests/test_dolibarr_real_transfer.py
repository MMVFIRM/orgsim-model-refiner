from __future__ import annotations

import copy
import json
from pathlib import Path
import pytest

from simlearn import dolibarr_transfer as rt


class ReadOnlyFake:
    """Fake Dolibarr. Mutations are modeled OUTSIDE the GET-only bridge."""
    def __init__(self):
        self.rows={}
        self.calls=[]
    def request(self,method,path,data=None):
        assert method=='GET' and data is None
        self.calls.append((method,path))
        assert path.startswith(('/orders/','/invoices/','/thirdparties/','/products/'))
        family,idx=path.strip('/').split('/')[:2]
        if path.endswith('/payments'):return 200,[]
        record=self.rows.get((family,idx))
        return (200,copy.deepcopy(record)) if record else (404,{'error':'missing'})


def _plan(tmp_path:Path, operations=('order_validate','invoice_validate'),tr=8,va=4,te=12):
    path=tmp_path/'scaffold.json'
    rt.scaffold(path,operations=operations,train=tr,validation=va,holdout=te)
    manifest=json.loads(path.read_text())
    # Update the template before initializing the immutable plan.
    for j,c in enumerate(manifest['cases']):
        c['record_id']=str(100+j)
    path.unlink(); path.write_text(json.dumps(manifest))
    return manifest,path


def test_full_real_journal_with_fake_manual_actions(tmp_path):
    plan,loc=_plan(tmp_path)
    root=tmp_path/'session';rt.init(loc,root)
    client=ReadOnlyFake()
    for i,c in enumerate(plan['cases']):
        fam=rt.OPS[c['operation']]
        # Deliberately include a trainable numerical veto independent of simple status.
        valid=(i%4<2)
        amount=40 if valid else 150
        client.rows[(fam,c['record_id'])]={'status':'0','total_ht':str(amount),'ref':'SECRET-CUSTOMER-REF',
                                             'webservices_key':'EXTRA_SECRET', 'lines':[{'id':'13','desc':'sensitive'}]}
    train=[c for c in plan['cases'] if c['split']!='holdout']
    hold=[c for c in plan['cases'] if c['split']=='holdout']
    for c in train:
        rt.start(root,c['id'],client=client)
        fam=rt.OPS[c['operation']]
        allowed=int(client.rows[(fam,c['record_id'])]['total_ht'])<=100
        if allowed:client.rows[(fam,c['record_id'])]['status']='1'
        rt.finish(root,c['id'],client=client,outcome='SUCCESS' if allowed else 'BUSINESS_REJECTED',source='UI')
    answer=rt.fit(root,min_train=6)
    assert answer['training_examples']==16
    assert answer['validation_examples']==8
    for c in hold:
        preview=rt.start(root,c['id'],client=client)
        assert 'majority_baseline' in preview['predictions']
        assert 'structured_tree' in preview['predictions']
        fam=rt.OPS[c['operation']]
        allowed=int(client.rows[(fam,c['record_id'])]['total_ht'])<=100
        if allowed:client.rows[(fam,c['record_id'])]['status']='1'
        rt.finish(root,c['id'],client=client,outcome='SUCCESS' if allowed else 'BUSINESS_REJECTED',source='UI')
    summary=rt.report(root)
    assert summary['holdout_completed']==24
    assert summary['holdout_labeled']==24
    assert set(summary['scores'])=={'majority_baseline','structured_tree'}
    assert summary['scores']['structured_tree']['accuracy_all_cases'] >= .5
    verification=rt.verify(root)
    assert verification['valid'] and verification['events']==(len(plan['cases'])*2+2)
    assert all(method=='GET' for method,_ in client.calls)
    # Raw ERP content and IDs remain private; excluded from public report.
    assert 'EXTRA_SECRET' not in (root/'private'/'cases'/(plan['cases'][0]['id']+'.start.json')).read_text()
    assert 'SECRET-CUSTOMER-REF' not in (root/'private'/'cases'/(plan['cases'][0]['id']+'.start.json')).read_text()
    public=(root/'public'/'REPORT.json').read_text()
    assert 'ref' not in public and plan['cases'][0]['record_id'] not in public


def test_reject_raw_host_and_invalid_path(tmp_path):
    plan,path=_plan(tmp_path)
    plan['base_url']='http://198.51.100.7:8088'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='loopback'):rt.init(path,tmp_path/'bad')
    plan['base_url']='http://127.0.0.1:8088'
    plan['cases'][0]['record_id']='../../foo'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='record_id'):rt.init(path,tmp_path/'bad2')


def test_holdout_freeze_required_and_zero_mutations(tmp_path):
    plan,loc=_plan(tmp_path)
    root=tmp_path/'session';rt.init(loc,root)
    c=next(c for c in plan['cases'] if c['split']=='holdout')
    fake=ReadOnlyFake();fake.rows[(rt.OPS[c['operation']],c['record_id'])]={'status':'0'}
    with pytest.raises(ValueError,match='Fit and freeze'):rt.start(root,c['id'],client=fake)
    assert not fake.calls


def test_verify_detects_tampering_and_immutable_cases(tmp_path):
    plan,loc=_plan(tmp_path)
    root=tmp_path/'session';rt.init(loc,root)
    c=plan['cases'][0]
    fake=ReadOnlyFake();fake.rows[(rt.OPS[c['operation']],c['record_id'])]={'status':'0'}
    rt.start(root,c['id'],client=fake)
    with pytest.raises(FileExistsError):rt.start(root,c['id'],client=fake)
    p=root/'private'/'cases'/(c['id']+'.start.json')
    doc=json.loads(p.read_text());doc['before']['main']['fields']['status']='not a real value'
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='modified'):rt.verify(root)


def test_uncorroborated_success_is_not_training_truth(tmp_path):
    plan,loc=_plan(tmp_path)
    root=tmp_path/'session';rt.init(loc,root)
    c=plan['cases'][0]
    fake=ReadOnlyFake();fake.rows[(rt.OPS[c['operation']],c['record_id'])]={'status':'0'}
    rt.start(root,c['id'],client=fake)
    result=rt.finish(root,c['id'],client=fake,outcome='SUCCESS',source='UI')
    assert result['label']=='UNKNOWN'
    assert not rt._get_train_rows(root,plan,'train')


def test_no_cross_tenant_artifact_usage(tmp_path):
    plan,loc=_plan(tmp_path)
    root=tmp_path/'session';rt.init(loc,root)
    root2=tmp_path/'session2'
    plan['snapshot_label']='totally_different_clone_001'
    clone=tmp_path/'changed.json';clone.write_text(json.dumps(plan))
    rt.init(clone,root2)
    left=json.loads((root/'private'/'WORKSPACE.json').read_text())
    right=json.loads((root2/'private'/'WORKSPACE.json').read_text())
    assert left['tenant_fingerprint']!=right['tenant_fingerprint']


def test_model_requires_explicit_financial_data_sharing(tmp_path):
    plan,loc=_plan(tmp_path)
    root=tmp_path/'session';rt.init(loc,root)
    c=plan['cases'][0]
    with pytest.raises(ValueError,match='Model calls require'):rt.start(root,c['id'],client=ReadOnlyFake(),execute_model=True)


def test_income_and_bodies_are_sanitized():
    raw={'status':'0','total_ht':'19.90','email':'customer@example.com',
         'webservices_key':'secret','name':'Patient John','lines':[{'label':'sensitive'}]}
    x=rt._normalize_response('invoices',200,raw)
    assert x['fields']['status']=='0'
    assert x['fields']['total_ht']=='19.90'
    assert x['fields']['lines_count']=='1'
    assert 'email' not in x['fields'] and 'webservices_key' not in x['fields']
    assert '19.90' not in rt.canonical({'format':'public_summary'})


def test_end_to_end_real_http_get_client_no_mutations(tmp_path,monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        actions=[]
        status='0'
        def do_GET(self):
            Handler.actions.append((self.command,self.path,self.headers.get('DOLAPIKEY')))
            assert self.path.startswith('/api/index.php/orders/')
            payload={'status':Handler.status,'total_ht':'14.25',
                     'name':'PRIVATE CUSTOMER','webservices_key':'PRIVATE_KEY'}
            body=json.dumps(payload).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
        def do_POST(self):raise AssertionError('No mutation is permitted')
        def log_message(self,*args):pass

    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    try:
        plan,loc=_plan(tmp_path,operations=('order_validate',),tr=2,va=0,te=2)
        plan['base_url']=f'http://127.0.0.1:{server.server_port}'
        loc.write_text(json.dumps(plan))
        root=tmp_path/'session';rt.init(loc,root)
        monkeypatch.setenv('DOLIBARR_API_KEY','DO_NOT_SAVE_THIS_KEY')
        c=plan['cases'][0]
        rt.start(root,c['id'])
        Handler.status='1' # operator-only simulated external transaction
        result=rt.finish(root,c['id'],outcome='SUCCESS',source='UI')
        assert result['label']=='ACCEPT'
        assert Handler.actions and all(method=='GET' for method,_,_ in Handler.actions)
        assert all(v=='DO_NOT_SAVE_THIS_KEY' for _,_,v in Handler.actions)
        records=''.join((root/'private'/'cases'/filename).read_text() for filename in
                        [f"{c['id']}.start.json",f"{c['id']}.finish.json"])
        assert 'DO_NOT_SAVE_THIS_KEY' not in records and 'PRIVATE' not in records
    finally:
        server.shutdown();server.server_close();worker.join(timeout=3)


def test_duplicate_target_record_across_splits_forbidden(tmp_path):
    plan,path=_plan(tmp_path,operations=('order_validate',),tr=2,va=0,te=2)
    plan['cases'][-1]['record_id']=plan['cases'][0]['record_id']
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='unique'):rt.init(path,tmp_path/'session')


def test_effect_rule_training_and_validation():
    rows=[]
    for i in range(3):
        row={'decision':'ACCEPT','before':{'main':{'http_status':200,'kind':'object','fields':{'status':'0'}}},
             'action':{},'changed_fields':{'main.status':{'from':'0','to':'1'}}}
        rows.append(row)
    effect=rt._effect_program(rows,rows[:1]);assert effect['ready']
    assert rt._effect_predict(effect,rows[0]['before'],{})=={'main.status':'1'}
    broken=copy.deepcopy(rows[:1]);broken[0]['changed_fields']['main.status']['to']='2'
    assert not rt._effect_program(rows,broken)['ready']


def test_model_refiner_four_arm_provider_mock_and_rehearsal_provenance(tmp_path):
    from model_refiner.backend import ModelBackend
    from model_refiner.types import ModelResult

    class FakeModel(ModelBackend):
        def __init__(self):self.messages=[]
        def generate(self,*,model,messages):
            self.messages.append(messages)
            return ModelResult(text='ACCEPT',input_tokens=40,output_tokens=1,
                               metadata={'status':'completed'})

    plan,loc=_plan(tmp_path,operations=('order_validate',),tr=8,va=4,te=2)
    root=tmp_path/'session';rt.init(loc,root)
    client=ReadOnlyFake()
    for i,c in enumerate(plan['cases']):
        client.rows[('orders',c['record_id'])]={'status':'0','total_ht':str(40 if i%4<2 else 150)}
    for c in [row for row in plan['cases'] if row['split']!='holdout']:
        rt.start(root,c['id'],client=client)
        allowed=int(client.rows[('orders',c['record_id'])]['total_ht'])<=100
        if allowed:client.rows[('orders',c['record_id'])]['status']='1'
        rt.finish(root,c['id'],outcome='SUCCESS' if allowed else 'BUSINESS_REJECTED',source='UI',client=client)
    r=rt.fit(root,min_train=6)
    assert r['real_retrieval_examples']==8
    assert r['simulator_rehearsal_pseudolabels']>0
    banks=rt._read(root/'private'/'MEMORY_BANKS.json')
    assert all(x['metadata']['provenance']=='SIMULATOR_PSEUDO_LABEL_UNVERIFIED'
               for x in banks['simulated_rehearsal_examples'])
    c=next(x for x in plan['cases'] if x['split']=='holdout')
    model=FakeModel()
    preview=rt.start(root,c['id'],client=client,execute_model=True,share_financials=True,model_backend=model)
    assert len(model.messages)==4
    assert {'llm_baseline','llm_real_retrieval','llm_sim_rehearsal','llm_policy'} <= set(preview['predictions'])
    assert preview['predictions']['llm_sim_rehearsal']['retrieved_examples']==4
    assert preview['predictions']['llm_real_retrieval']['retrieved_examples']==4
    assert any('SIMULATOR' not in str(m) for m in model.messages)
    tampered=copy.deepcopy(banks);tampered['tenant_fingerprint']='not_this_tenant'
    with pytest.raises(ValueError,match='another tenant'):
        rt._real_model_predict(rt._read(root/'private'/'FROZEN_POLICY.json'),c,
                               rt._capture(client,c),'gpt-6-luna',backend=model,memory_banks=tampered)
    assert len(model.messages)==4


def test_status_guides_operator_without_erp_reads(tmp_path):
    plan,loc=_plan(tmp_path,operations=('order_validate',),tr=2,va=0,te=2)
    root=tmp_path/'root';rt.init(loc,root)
    s=rt.status(root)
    assert s['next_action']=='start' and s['next_case']['split']=='train'
    assert s['read_requests']==0 and s['model_calls']==0
    fake=ReadOnlyFake();c=plan['cases'][0]
    fake.rows[('orders',c['record_id'])]={'status':'0'}
    rt.start(root,c['id'],client=fake)
    s=rt.status(root)
    assert s['next_action']=='finish' and s['next_case']['case_id']==c['id']
