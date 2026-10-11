"""Gate 2F: real staged code paths with a local fake, plus synthetic causal controls."""
from __future__ import annotations
import json
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
import pytest

from orgsim.gate2f_readonly import (prepare, Evaluator, verify, QUERIES, BY_ID,
                                    _read_frozen, _check_pred, _swagger_ops)
from orgsim.gate2f_causal import run as run_causal

ROOT=Path(__file__).resolve().parents[1]
SWAGGER=ROOT/'gate_dolibarr/gate2f_inputs/sanitized_live_swagger.json'
DISCOVERY=ROOT/'gate_dolibarr/gate2f_inputs/gate2d_discovery_report.json'


class FakeERP:
    loopback=True
    def __init__(self, *, reverse_broken=False):
        self.calls=[];self.reverse_broken=reverse_broken
        self.data={
            'thirdparties':[{'id':str(343000+i),'name':f'SECRET_CUSTOMER_{i}',
                             'client':'1','ref':f'CUSTOMER_SECRET_{i}',
                             'country_code':'US','status':'1'} for i in range(3)],
            'products':[{'id':'492100','ref':'SECRET_PROD','label':'SECRET_LABEL',
                         'type':'0','status':'1','price':'10.00'}],
            'orders':[{'id':str(561000+i),'ref':f'SECRET_ORDER_{i}',
                       'socid':str(343000+i),'status':str(i),
                       'total_ht':'15.00','lines':[]} for i in range(2)],
            'invoices':[{'id':'762000','ref':'SECRET_INVOICE','socid':'343000',
                         'status':'1','lines':[],'total_ht':'12.00'}],
        }

    def request(self,method,path):
        assert method=='GET' and path.startswith('/') and not path.startswith('//')
        self.calls.append((method,path))
        bits=urlsplit(path);segments=bits.path.strip('/').split('/')
        fam=segments[0];assert fam in self.data,path
        opts=parse_qs(bits.query)
        if len(segments)==1:
            records=list(self.data[fam])
            if opts.get('status')==['draft']:
                records=[r for r in records if r.get('status')=='0']
            if opts.get('status')==['unpaid']:
                records=[r for r in records if r.get('status')=='1']
            if opts.get('mode')==['1']:
                records=[r for r in records if r.get('type')=='0']
            if opts.get('sortorder')==['DESC'] and not self.reverse_broken:
                records.reverse()
            limit=int(opts.get('limit',['20'])[0]);page=int(opts.get('page',['0'])[0])
            records=records[page*limit:(page+1)*limit]
            if opts.get('ids_only')==['1']:return 200,[r['id'] for r in records]
            if opts.get('withLines')==['false']:
                records=[{k:v for k,v in r.items() if k!='lines'} for r in records]
            if 'properties' in opts:
                cols=opts['properties'][0].split(',')
                records=[{k:r[k] for k in cols if k in r} for r in records]
            return 200,records
        if len(segments)>2 and segments[1]=='ref':
            record=next((r for r in self.data[fam] if r.get('ref')==segments[2]),None)
            return (200,record) if record else (404,{'error':{'secret':'NO_REVEAL'}})
        rec=next((r for r in self.data[fam] if r['id']==segments[1]),None)
        if rec is None:return 404,{'error':'NO_REVEAL'}
        if len(segments)==2:return 200,rec
        end=segments[2]
        if end=='outstandingorders':return 200,{'opened':0,'refs':{}}
        if end=='outstandinginvoices':return 200,{'opened':0,'refs':{},'refsopened':[]}
        if end=='contacts':return 200,[]
        if end=='discount':return 200,{}
        if end=='categories':return 403,{'error':{'private':'ACL_SECRET'}}
        if end=='purchase_prices':return 403,{'error':{'private':'ACL_SECRET'}}
        if end=='variants':return 200,[]
        raise AssertionError('unexpected route '+path)


def staged(tmp_path,*,model=None,client=None,budget=32):
    frozen=tmp_path/'frozen'
    prep=prepare(DISCOVERY,SWAGGER,frozen,model_advisor=model)
    fake=client or FakeERP()
    result=Evaluator(fake,frozen,budget=budget).run(tmp_path/'evaluation',execute=True)
    return frozen,fake,result,tmp_path/'evaluation'


def test_new_templates_exist_in_real_sanitized_swagger():
    doc=json.loads(SWAGGER.read_text())
    found=_swagger_ops(doc)
    assert len(found)==len(QUERIES)==18
    assert len({q.route for q in QUERIES})==len(QUERIES)
    gate2e=json.loads((ROOT/'gate_dolibarr/gate2e_seed_frozen/holdout_plan.json').read_text())
    old={row['template'] for row in gate2e['cases']}
    assert not old.intersection(q.route for q in QUERIES)


def test_prepare_is_offline_freeze_and_source_matched(tmp_path):
    frozen=tmp_path/'frozen'
    result=prepare(DISCOVERY,SWAGGER,frozen)
    assert result['planned_cases']==18
    assert result['freeze']['live_reads']==result['freeze']['live_writes']==0
    assert result['freeze']['source_discovery_sha256']=='136025e9432f687605691473d515ad1dccdb435d227220e9146c04c77eb51d2c'
    c,p,f=_read_frozen(frozen)
    assert set(c['arms'])=={'contract','symbolic'}
    assert c['reviewed'] is False and c['behavioral_clone'] is False
    assert len(p['cases'])==18
    with pytest.raises(FileExistsError):prepare(DISCOVERY,SWAGGER,frozen)


def test_dry_run_never_contacts_api(tmp_path):
    frozen=tmp_path/'f';prepare(DISCOVERY,SWAGGER,frozen)
    class NoNetwork:
        loopback=True
        def request(self,*args):raise AssertionError('dryrun unexpectedly sent a request')
    evaluator=Evaluator(NoNetwork(),frozen,budget=30)
    report=evaluator.run(tmp_path/'dry',execute=False)
    assert report['operations_executed']==0
    assert verify(frozen,tmp_path/'dry')['status']=='PASS_OFFLINE_PREVIEW'


def test_full_local_experiment_three_labels_and_integrity(tmp_path):
    frozen,client,report,out=staged(tmp_path)
    assert report['operations_executed']==22
    assert report['selector_reads']==4
    assert report['cases_scored']==18
    assert len(client.calls)==22 and all(x[0]=='GET' for x in client.calls)
    assert report['live_writes']==0
    assert report['model_provider_calls_during_evaluation']==0
    assert set(report['scores'])=={'contract','symbolic'}
    assert report['scores']['symbolic']['predicted_fields']>report['scores']['contract']['predicted_fields']
    assert report['scores']['symbolic']['special_checks']>=3
    assert verify(frozen,out)['reads_verified']==len(client.calls)


def test_model_is_frozen_with_same_queries_and_never_called_during_evaluation(tmp_path):
    class Stub:
        calls=0
        def predict_many(self,cases,evidence):
            self.calls+=1
            assert 'SECRET_CUSTOMER' not in json.dumps(evidence)
            assert 'SECRET_PROD' not in json.dumps(evidence)
            assert len(cases)==18
            return {q.id:{'status':200} for q in cases}
    model=Stub()
    frozen,client,report,out=staged(tmp_path,model=model)
    assert model.calls==1
    assert len(client.calls)==22
    assert set(report['scores'])=={'contract','symbolic','model'}
    assert report['model_provider_calls_during_evaluation']==0
    assert report['model_provider_calls_during_prepare']==1
    assert verify(frozen,out)['reads_verified']==22


def test_no_customer_identifiers_or_raw_values_written(tmp_path):
    frozen,client,report,out=staged(tmp_path)
    for folder in (frozen,out):
        for file in folder.iterdir():
            if file.suffix in ('.json','.jsonl'):
                content=file.read_text(encoding='utf-8')
                for secret in ('SECRET_CUSTOMER','SECRET_ORDER','SECRET_INVOICE',
                               'SECRET_PROD','343000','561000','492100','762000',
                               'ACL_SECRET','NO_REVEAL'):
                    assert secret not in content, (file,secret)


def test_freeze_and_log_tampering_rejected(tmp_path):
    frozen,client,report,out=staged(tmp_path)
    path=frozen/'candidate_frozen.json'
    raw=path.read_bytes();path.write_bytes(raw+b' ')
    with pytest.raises(ValueError):verify(frozen,out)
    path.write_bytes(raw)
    events=out/'events.jsonl'
    events.write_bytes(events.read_bytes().replace(b'prediction_committed',b'prediction_uncommitted',1))
    with pytest.raises(ValueError):verify(frozen,out)


def test_budget_ceiling_and_skips_not_concealed(tmp_path):
    frozen,client,report,out=staged(tmp_path,budget=7)
    assert report['operations_executed']==7
    assert report['cases_scored']==3
    assert len(report['skipped'])==15
    assert verify(frozen,out)['reads_verified']==7


def test_permission_and_ordering_mismatch_is_scored(tmp_path):
    frozen,client,report,out=staged(tmp_path,client=FakeERP(reverse_broken=True))
    first=next(x for x in report['cases'] if x['id']=='customers_desc')
    assert first['observation']['reverse_matches'] is False
    assert first['scores']['symbolic']['field_checks']['reverse_matches'] is False
    assert verify(frozen,out)['status']=='PASS_LOCAL_HASH_AND_ORDER_ONLY'


def test_unsafe_predictions_and_plan_rejected(tmp_path):
    frozen=tmp_path/'f';prepare(DISCOVERY,SWAGGER,frozen)
    with pytest.raises(ValueError):_check_pred(BY_ID['orders_by_ref'],{'path':'//attacker.com'})
    with pytest.raises(ValueError):_check_pred(BY_ID['orders_by_ref'],{'ref_matches':'true'})
    with pytest.raises(ValueError):Evaluator(object(),frozen)
    with pytest.raises(ValueError):Evaluator(FakeERP(),frozen,budget=999)


def test_causal_gate_matches_budgets_and_exposes_coverage_failure(tmp_path):
    result=run_causal(seeds=2,budgets=(20,32),holdout=80,
                      outdir=tmp_path/'causal')
    assert result['live_calls']==0 and result['live_writes']==0
    for family in ('simple','interaction'):
        for budget in (20,32):
            rows=[x for x in result['results'] if x['oracle_family']==family
                  and x['budget']==budget]
            assert set(x['arm'] for x in rows)=={'uniform','counterexample'}
            assert len({x['cases'] for x in rows})==1
            assert len({x['discovery_observations'] for x in rows})==1
    simple=next(x for x in result['results'] if x['oracle_family']=='simple'
                and x['budget']==32 and x['arm']=='counterexample')
    hard=next(x for x in result['results'] if x['oracle_family']=='interaction'
              and x['budget']==32 and x['arm']=='counterexample')
    assert simple['coverage']>=.8 and simple['false_accepts']==0
    assert hard['coverage']<.8 and hard['causal_gate']=='FAIL'


def test_api_model_adapter_frozen_json_shape_and_transport(monkeypatch):
    from orgsim.gate2f_readonly import OpenAI2FPredictor,QUERIES
    from orgsim.gate2d_adaptive import canonical
    class Answer:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def read(self,n):
            return json.dumps({'output':[{'content':[{'type':'output_text','text':
                json.dumps({'predictions':{'customers_desc':{'status':200,'kind':'array'}}})}]}]}).encode()
    class Opener:
        def open(self,request,timeout):
            assert timeout==120
            request_body=json.loads(request.data)
            assert request_body['model']=='test-model'
            assert request_body['store'] is False
            assert request.full_url=='https://api.openai.com/v1/responses'
            assert 'SECRET' not in canonical(request_body)
            return Answer()
    adapter=OpenAI2FPredictor.__new__(OpenAI2FPredictor)
    adapter.model='test-model'
    adapter._advisor=type('Creds',(),{'_key':'SAMPLE_TEST_ONLY','_opener':Opener()})()
    result=adapter.predict_many(list(QUERIES), {'previous_shapes':{'orders':'array'}})
    assert result=={'customers_desc':{'status':200,'kind':'array'}}


def test_causal_report_freeze_is_reproducible(tmp_path):
    import hashlib
    run_causal(seeds=2,budgets=(20,),holdout=80,outdir=tmp_path/'causal')
    base=tmp_path/'causal'
    freeze=json.loads((base/'CAUSAL_FREEZE.json').read_text())
    assert freeze['report_sha256']==hashlib.sha256((base/'causal_report.json').read_bytes()).hexdigest()
    assert freeze['records_sha256']==hashlib.sha256((base/'per_operation.json').read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):run_causal(seeds=2,budgets=(20,),holdout=80,outdir=base)
