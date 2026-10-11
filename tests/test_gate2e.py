"""Gate 2E offline tests: real staged freeze, unknown query results, privacy and tampering."""
from __future__ import annotations
import json
from pathlib import Path
from urllib.parse import urlsplit,parse_qs

import pytest
from orgsim.gate2d_adaptive import Investigator
from orgsim.gate2e_blind import (
    prepare, verify, Evaluator, _profiles, _cases, _source_report, _score_one,
    Case, validate_prediction, _read_frozen, _case_from_public, _known_type_fields,
)


class FauxDolibarr:
    loopback=True
    def __init__(self, *, break_acl=False, break_foreign_key=False, sparse=False):
        self.calls=[]
        self.break_acl=break_acl
        self.break_foreign_key=break_foreign_key
        self.data={
            'thirdparties': [
                {'id':str(123456780+i),'name':f'SECRET_NAME_{i}', 'client':'1',
                 'ref':'CU'+str(i),'country_code':'US',
                 'email':None, 'status':'1',
                 **{'extra'+str(k):None for k in range(12)}}
                for i in range(3)],
            'products': [
                {'id':'987654321','ref':'SECRET-PRODUCT','label':'HIDDEN_PRODUCT_LABEL',
                 'price':'10.50','type':'0','status':'1',
                 **{'extra'+str(k):None for k in range(15)}}],
            'orders': [
                {'id':str(555552222+i),'socid':str(123456780+i), 'ref':'HIDDEN_ORDER_'+str(i),
                 'status':'1','total_ht':'55.00','lines':[],
                 **{'extra'+str(k):None for k in range(20)}}
                for i in range(2)] if not sparse else [],
            'invoices': [
                {'id':'555553333','socid':'123456780','ref':'HIDDEN_INVOICE',
                 'status':'1','total_ht':'55.00','paye':'0',
                 **{'extra'+str(k):None for k in range(20)}}] if not sparse else [],
        }
    def request(self, method, path):
        assert method=='GET', 'Only read requests are permitted'
        assert not path.startswith('//')
        self.calls.append((method,path))
        u=urlsplit(path)
        bits=u.path.strip('/').split('/')
        fam=bits[0]
        assert fam in self.data, path
        opts=parse_qs(u.query)
        if len(bits)==1:
            limit=int(opts.get('limit',['20'])[0]); page=int(opts.get('page',['0'])[0])
            rows=self.data[fam][page*limit:(page+1)*limit]
            if opts.get('ids_only')==['1']:return 200,[r['id'] for r in rows]
            if 'properties' in opts:
                props=opts['properties'][0].split(',')
                rows=[{key:r[key] for key in props if key in r} for r in rows]
            return 200,rows
        if fam=='products' and bits[1]=='ref':
            record=next((r for r in self.data[fam] if r['ref']==bits[2]),None)
            return (200,record) if record else (404,{'error':'INTERNAL_ONLY'})
        record=next((r for r in self.data[fam] if r['id']==bits[1]),None)
        if record is None:return 404,{'error':{'secret':'DO_NOT_LOG'}}
        if len(bits)==2:return 200,record
        if fam=='products' and bits[2] in ('stock','categories'):
            if not self.break_acl:return 403,{'error':{'message':'INTERNAL'}}
            return 200,[]
        if fam=='orders' and bits[2]=='shipment':
            return 403,{'error':{'message':'INTERNAL'}}
        if fam=='orders' and bits[2]=='lines':
            return 200,[{'id':'444441111','fk_product':'987654321' if not self.break_foreign_key
                             else '777777777', 'qty':'2','subprice':'10.50','total_ht':'21.00'}]
        if fam=='invoices' and bits[2]=='lines':
            return 200,[{'id':'444442222','fk_product':'987654321', 'qty':'1','total_ht':'10.50'}]
        if fam=='invoices' and bits[2]=='payments':return 200,[]
        if fam=='thirdparties' and bits[2] in ('outstandingorders','outstandinginvoices'):
            return 200,{'opened':0,'refs':{}}
        raise AssertionError('Unexpected route '+path)


def make_discovery(tmp_path, *, sparse=False):
    origin=tmp_path/'gate2d'
    client=FauxDolibarr(sparse=sparse)
    report=Investigator(client,budget=20).run(execute=True,outdir=origin)
    assert 4 <= report['operations_executed'] <= 20
    return origin/'discovery_report.json'


def staged(tmp_path, *, sparse=False, model_advisor=None, oracle=None, budget=32):
    discovery=make_discovery(tmp_path,sparse=sparse)
    freeze=tmp_path/'frozen'
    pre=prepare(discovery,freeze,model_advisor=model_advisor)
    client=oracle or FauxDolibarr(sparse=sparse)
    evaluator=Evaluator(client,freeze,budget=budget)
    report=evaluator.run(tmp_path/'holdout',execute=True)
    return freeze,client,report,tmp_path/'holdout'


def test_freeze_is_prepared_before_any_holdout_calls(tmp_path):
    source=make_discovery(tmp_path)
    target=tmp_path/'freeze'
    prepared=prepare(source,target)
    assert prepared['freeze']['reads_executed']==0
    assert prepared['freeze']['model_provider_calls']==0
    assert prepared['freeze']['candidate_reviewed'] is False
    candidate,plan,_=_read_frozen(target)
    assert len(plan['cases'])>=12
    assert candidate['behavioral_clone'] is False
    assert set(candidate['predictors'])=={'symbolic'}
    assert 'orders_customer_fk' in {c['id'] for c in plan['cases']}
    assert 'products_stock_permission' in {c['id'] for c in plan['cases']}
    assert all(c['template'].startswith('/') for c in plan['cases'])


def test_full_offline_holdout_safe_and_separates_metrics(tmp_path):
    freeze,client,report,output=staged(tmp_path)
    assert report['phase']=='holdout'
    assert report['independent_tenant_holdout'] is False
    assert report['operations_executed']<=32
    assert report['cases_scored']>=13
    assert report['live_writes']==0
    assert all(m=='GET' for m,_ in client.calls)
    assert report['model_provider_calls_during_evaluation']==0
    assert report['scores']['symbolic']['status_accuracy'] is not None
    assert report['scores']['symbolic']['fields_observable']>=report['scores']['symbolic']['fields_predicted']
    assert report['scores']['symbolic']['relationship_accuracy'] is not None
    assert report['scores']['symbolic']['permission_cases_scored']==3
    assert report['scores']['symbolic']['permission_status_accuracy']==1.0
    assert report['scores']['symbolic']['relationships_scored']==3
    audit=verify(freeze,output)
    assert audit['verified_precommitted_reads']==len(client.calls)
    assert audit['status']=='PASS_LOCAL_INTEGRITY_AND_ORDER_ONLY'


def test_persist_no_ids_or_customer_values_even_in_event_chain(tmp_path):
    freeze,client,report,folder=staged(tmp_path)
    for base in (freeze,folder):
        for fp in base.glob('*.json*'):
            string=fp.read_text(encoding='utf-8')
            for secret in ('123456780','123456781','555552222','987654321',
                           'HIDDEN_ORDER','SECRET_NAME','SECRET-PRODUCT','DO_NOT_LOG'):
                assert secret not in string, (fp, secret)
    assert report['raw_values_saved'] is False


def test_model_and_symbolic_use_same_oracle_reads(tmp_path):
    class StubModel:
        def __init__(self):self.calls=0
        def predict_many(self, cases, evidence):
            self.calls+=1
            assert 'SECRET' not in json.dumps(evidence)
            return {c.id:{'status':200} for c in cases}
    advisor=StubModel()
    freeze,client,report,output=staged(tmp_path,model_advisor=advisor)
    assert advisor.calls==1
    assert set(report['scores'])=={'symbolic','model'}
    assert report['model_provider_calls_during_prepare']==1
    assert report['model_provider_calls_during_evaluation']==0
    assert report['scores']['model']['permission_status_accuracy']==0
    assert report['scores']['symbolic']['permission_status_accuracy']==1
    assert verify(freeze,output)['status'].startswith('PASS_')


def test_permission_drift_is_a_real_error_not_hidden(tmp_path):
    freeze,client,report,output=staged(tmp_path,oracle=FauxDolibarr(break_acl=True))
    s=report['scores']['symbolic']
    assert s['permission_status_accuracy']<1
    errors={x['id']:x for x in report['cases']}
    assert errors['products_stock_permission']['scores']['symbolic']['fields_scored']['status'] is False
    assert errors['products_categories_permission']['scores']['symbolic']['fields_scored']['status'] is False


def test_broken_fk_is_scored_instead_of_claiming_relationship_pass(tmp_path):
    freeze,client,report,output=staged(tmp_path,oracle=FauxDolibarr(break_foreign_key=True))
    rel={x['id']:x for x in report['cases'] if x['category']=='relationship'}
    assert rel['order_line_product_fk']['observation']['relation_exists'] is False
    assert rel['order_line_product_fk']['scores']['symbolic']['fields_scored']['relation_exists'] is False
    assert verify(freeze,output)['status'].startswith('PASS_')


def test_sparse_tenant_skips_absent_workflows_without_making_up_accuracy(tmp_path):
    freeze,client,report,output=staged(tmp_path,sparse=True)
    assert report['cases_scored']>=4
    assert any(row['id']=='orders_customer_fk' for row in report['skipped'])
    assert report['scores']['symbolic']['relationships_scored']==0
    assert verify(freeze,output)['status'].startswith('PASS_')


def test_budget_ceiling_and_skip_reporting(tmp_path):
    freeze,client,report,output=staged(tmp_path,budget=7)
    assert len(client.calls)==7
    assert len(report['skipped'])>0
    assert report['stop_reason']=='budget_exhausted'
    assert verify(freeze,output)['verified_precommitted_reads']==7


def test_dry_run_never_calls_client(tmp_path):
    path=make_discovery(tmp_path)
    prepare(path,tmp_path/'frozen')
    class NoNetwork:
        loopback=True
        def request(self,*args): raise AssertionError('Dry run must not make requests')
    r=Evaluator(NoNetwork(),tmp_path/'frozen').run(tmp_path/'preview',execute=False)
    assert r['phase']=='dry_run'
    assert verify(tmp_path/'frozen',tmp_path/'preview')['status']=='PASS_DRY_RUN_ARTIFACT_ONLY'


def test_tamper_detection_on_candidate_report_and_events(tmp_path):
    freeze,client,report,out=staged(tmp_path)
    target=out/'events.jsonl'
    before=target.read_text()
    target.write_text(before.replace('"kind":"array"','"kind":"string"',1))
    with pytest.raises(ValueError):verify(freeze,out)
    target.write_text(before)
    assert verify(freeze,out)['status'].startswith('PASS_')
    fp=freeze/'candidate_frozen.json'
    fp.write_text(fp.read_text().replace('"behavioral_clone": false','"behavioral_clone": true'))
    with pytest.raises(ValueError):verify(freeze,out)


def test_malicious_case_paths_or_predictions_blocked(tmp_path):
    source=make_discovery(tmp_path)
    prepare(source,tmp_path/'frozen')
    candidate,plan,_=_read_frozen(tmp_path/'frozen')
    original=plan['cases'][0]
    original['template']='//attacker.example/login'
    with pytest.raises(ValueError): _case_from_public(original)
    case=Case('x','products','detail','products_detail','/products/{id}',
              selected_fields=('id','ref'))
    for pred in ({'url':'https://bad.example'}, {'status':'200'},
                 {'field_types':{'secret':'string'}},
                 {'relation_exists':True}, {'item_keys':['password']}):
        with pytest.raises(ValueError): validate_prediction(pred,case)


def test_null_type_abstained_and_denominator_penalizes_incomplete_guesses():
    profile={'field_types':{'id':'string','email':'null','price':'number'}}
    assert _known_type_fields(profile,('id','email','price'))=={'id':'string','price':'number'}
    c=Case('x','products','detail','products_detail','/products/{id}',
           selected_fields=('id','ref','price'))
    observed={'status':200,'kind':'object','field_types':{'id':'string','ref':'string','price':'number'}}
    poor=_score_one(c,{'status':200},observed)
    assert poor['full_match'] is True
    assert poor['substantive_full_match'] is False
    assert poor['coverage']<0.5
    bad=_score_one(c,{'status':200,'field_types':{'price':'string'}},observed)
    assert bad['correct_fields']==1 and bad['predicted_fields']==2


def test_non_local_client_rejected(tmp_path):
    source=make_discovery(tmp_path)
    prepare(source,tmp_path/'freeze')
    class Remote:
        loopback=False
    with pytest.raises(ValueError):Evaluator(Remote(),tmp_path/'freeze')


def test_every_live_get_has_a_durable_precommit_before_call(tmp_path):
    src=make_discovery(tmp_path)
    root=tmp_path/'frozen'
    prepare(src,root)
    out=tmp_path/'holdout'
    seen=[]
    class Spy(FauxDolibarr):
        def request(self,method,path):
            lines=(out/'events.jsonl').read_text().splitlines()
            assert lines, 'No evidence log before remote call'
            event=json.loads(lines[-1])
            assert event['event']=='prediction_committed'
            assert event['method']=='GET'
            seen.append(event['probe_id'])
            return super().request(method,path)
    client=Spy()
    report=Evaluator(client,root,budget=32).run(out,execute=True)
    assert len(seen)==report['operations_executed']
    assert verify(root,out,src)['status'].startswith('PASS_')


def test_budget_enough_for_one_but_not_two_last_reads_skips_relation(tmp_path):
    src=make_discovery(tmp_path)
    root=tmp_path/'freeze'
    prepare(src,root)
    # 4 selector GETs + 12 case GETs = 16. The 13th case needs TWO reads.
    # It must be skipped rather than committing one and then crashing.
    client=FauxDolibarr()
    result=Evaluator(client,root,budget=17).run(tmp_path/'holdout',execute=True)
    assert result['operations_executed']<=17
    assert any(row['id']=='order_line_product_fk' and 'budget' in row['reason']
               for row in result['skipped'])
    assert verify(root,tmp_path/'holdout')['status'].startswith('PASS_')


def test_openai_model_adapter_mock_no_erp_values_and_strict_validation(tmp_path):
    from orgsim.gate2e_blind import OpenAIBlindPredictor
    from types import SimpleNamespace
    class Reply:
        def __init__(self, b):self.b=b
        def read(self,n):return self.b
        def __enter__(self):return self
        def __exit__(self,*a):pass
    class FakeOpener:
        def __init__(self):self.last=None
        def open(self,request,timeout):
            self.last=request
            b=json.dumps({'output_text':json.dumps({
                'predictions':{'thirdparties_projected_limit2':{
                    'status':200,'kind':'array','field_types':{'id':'string'}}}})}).encode()
            return Reply(b)
    source=make_discovery(tmp_path)
    report,_=_source_report(source)
    profiles=_profiles(report)
    cases=_cases(profiles)
    instance=OpenAIBlindPredictor.__new__(OpenAIBlindPredictor)
    instance.model='gpt-6'
    transport=FakeOpener()
    instance._existing=SimpleNamespace(_key='TEST_TOKEN_DO_NOT_SAVE',_opener=transport)
    from orgsim.gate2e_blind import _allowed_evidence
    result=instance.predict_many(cases,_allowed_evidence(profiles))
    assert result['thirdparties_projected_limit2']['field_types']['id']=='string'
    req=transport.last
    assert req.full_url=='https://api.openai.com/v1/responses'
    body=json.loads(req.data)
    assert body['store'] is False
    assert 'HIDDEN_ORDER' not in json.dumps(body)
    assert 'SECRET-PRODUCT' not in json.dumps(body)
    assert '123456780' not in json.dumps(body)
    assert 'TEST_TOKEN_DO_NOT_SAVE' not in json.dumps(body)


def test_transport_exception_yields_partial_auditable_failure(tmp_path):
    src=make_discovery(tmp_path)
    freeze=tmp_path/'frozen'
    prepare(src,freeze)
    class NetworkFails(FauxDolibarr):
        def request(self, method, path):
            if path.startswith('/products?limit=20'):
                raise TimeoutError('PRIVATE_APIKEY_123456780 not for logs')
            return super().request(method,path)
    client=NetworkFails()
    report=Evaluator(client,freeze,budget=32).run(tmp_path/'failed',execute=True)
    assert report['stop_reason']=='aborted:RuntimeError'
    assert 'PRIVATE_APIKEY' not in (tmp_path/'failed/holdout_report.json').read_text()
    assert '123456780' not in (tmp_path/'failed/events.jsonl').read_text()
    assert verify(freeze,tmp_path/'failed')['status'].startswith('PASS_')
